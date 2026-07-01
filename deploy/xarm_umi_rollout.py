"""LingBot-VLA rollout on a real xArm 7 + UMI (Damiao 4310) gripper.

Interactive, chunk-at-a-time deployment client — modeled on
`/home/pierre/Desktop/repo/xrobo/gr00t_rollout_async.py`:

  * async background camera reader (latest-frame-in-memory, ~1 ms reads),
  * preview each predicted action chunk and PROMPT [E]xecute / [S]kip / [Q]uit
    before any motion is sent (or `--auto` to run unattended),
  * per-chunk latency accounting and a summary at the end.

It talks to the LingBot-VLA **websocket policy server** (INFERENCE.md §2), so
model loading, normalization, feature-transform and de-normalization all happen
server-side. Start the server first, e.g.:

    export QWEN25_PATH=/home/pierre/models/Qwen2.5-VL-3B-Instruct
    CKPT=/path/to/umi_real_depth/global_step_4000/hf_ckpt
    python -m deploy.lingbot_vla_policy \
        --model_path $CKPT --use_length 25 --port 8000 \
        --norm_path assets/norm_stats/umi.json

then run this client:

    python -m deploy.xarm_umi_rollout \
        --host 127.0.0.1 --port 8000 \
        --arm-ip 192.168.1.226 \
        --camera 0 \
        --task "Pick up the small circular item and put it in the cup on the left."

What this client sends the model each step (the UMI 10-dim contract):
  observation.state = [x, y, z, rot6d(0..5), gripper]         (raw / un-normalized)
  observation.images.camera_top         = live RGB frame
  observation.images.camera_wrist_left  = SAME frame (replicated)
  observation.images.camera_wrist_right = SAME frame (replicated)
  task = the language instruction

The server returns a de-normalized action chunk `action` of shape (T, 10) in the
same [xyz, rot6d, gripper] space. We convert rot6d -> RPY, m -> mm, and stream it
to the arm with `set_servo_cartesian`; the gripper channel is mapped to motor rad.

NOTE — assumptions you should validate (see the end-of-run summary):
  * rot6d convention (first two ROWS of R; Gram-Schmidt inverse) matches training.
  * the model's pose frame == the xArm base frame, in meters, absolute pose.
  * gripper linear map (model 0.27..1.0  <->  your calibrated close..open rad).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# --- websocket client to the LingBot-VLA policy server (same package) ---------
try:
    from .websocket_client_policy import WebsocketClientPolicy
except ImportError:  # allow running as a plain script, not just `-m`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from deploy.websocket_client_policy import WebsocketClientPolicy


# =============================================================================
# rot6d <-> rotation helpers  (Pierre's stack convention: first two ROWS of R)
# =============================================================================
# Mirrors gr00t_rollout.py's quat_xyzw_to_rot6d / rot6d_to_quat_xyzw exactly, but
# expressed on rotation matrices so we never have to pick a quaternion frame.
# Flip ROT6D_ROWS to False if open-loop eval shows the training script used the
# first two COLUMNS (the "standard" Zhou et al. convention) instead.

ROT6D_ROWS = True


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> rot6d (6,)."""
    return (R[:2, :] if ROT6D_ROWS else R[:, :2].T).reshape(-1).astype(np.float64)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """rot6d (6,) -> 3x3 rotation matrix via Gram-Schmidt."""
    a, b = np.asarray(rot6d, dtype=np.float64).reshape(2, 3)
    r1 = a / (np.linalg.norm(a) + 1e-9)
    b = b - np.dot(r1, b) * r1
    r2 = b / (np.linalg.norm(b) + 1e-9)
    r3 = np.cross(r1, r2)
    rows = np.stack([r1, r2, r3], axis=0)          # rows == the two we exported
    return rows if ROT6D_ROWS else rows.T


def rpy_to_rot6d(rpy: np.ndarray) -> np.ndarray:
    """xArm [roll, pitch, yaw] (rad) -> rot6d (6,)."""
    return matrix_to_rot6d(Rotation.from_euler("xyz", rpy).as_matrix())


def rot6d_to_rpy(rot6d: np.ndarray) -> np.ndarray:
    """rot6d (6,) -> xArm [roll, pitch, yaw] (rad)."""
    return Rotation.from_matrix(rot6d_to_matrix(rot6d)).as_euler("xyz")


# =============================================================================
# gripper: model channel (0.27..1.0) <-> motor radians (calibrated)
# =============================================================================

class GripperMap:
    """Linear two-point map between the model's continuous gripper channel and
    the Damiao motor angle (rad). Calibrate close_rad / open_rad once with
    umi.py (it prints live motor position while you move the jaws by hand)."""

    def __init__(self, model_close: float, model_open: float,
                 close_rad: float, open_rad: float):
        self.mc, self.mo = float(model_close), float(model_open)
        self.rc, self.ro = float(close_rad), float(open_rad)

    def model_to_rad(self, g: float) -> float:
        g = float(np.clip(g, min(self.mc, self.mo), max(self.mc, self.mo)))
        frac = (g - self.mc) / (self.mo - self.mc + 1e-9)
        return self.rc + frac * (self.ro - self.rc)

    def rad_to_model(self, rad: float) -> float:
        frac = (rad - self.rc) / (self.ro - self.rc + 1e-9)
        return self.mc + frac * (self.mo - self.mc)


# =============================================================================
# async camera (cv2.VideoCapture + background reader thread) -- from gr00t_async
# =============================================================================

class AsyncCamera:
    """cv2.VideoCapture + bg thread holding the latest frame; read_rgb() is a
    memcpy + BGR->RGB (~1 ms). Rejects stub /dev/videoN nodes via std>0.5."""

    def __init__(self, index: int, width: int, height: int, fps: int):
        import cv2
        import threading
        self._cv2 = cv2
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"cv2.VideoCapture({index}) failed to open")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except Exception:
            pass
        warm = None
        for _ in range(60):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0 and frame.std() > 0.5:
                warm = frame
                break
            time.sleep(0.05)
        if warm is None:
            cap.release()
            raise RuntimeError(f"/dev/video{index}: no usable frame after 3 s warm-up")
        self._cap = cap
        self._latest_bgr = warm
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._reader = threading.Thread(
            target=self._loop, name=f"async-cam-{index}", daemon=True)
        self._reader.start()
        self.index = index
        self.shape = warm.shape

    def _loop(self):
        cap = self._cap
        while not self._stop.is_set():
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                with self._lock:
                    self._latest_bgr = frame
            else:
                time.sleep(0.001)

    def read_rgb(self) -> np.ndarray:
        with self._lock:
            bgr = self._latest_bgr.copy()
        return self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)

    def disconnect(self):
        self._stop.set()
        try:
            self._reader.join(timeout=2.0)
        except Exception:
            pass
        try:
            self._cap.release()
        except Exception:
            pass


def find_async_camera(want_idx: int, w: int, h: int, fps: int) -> AsyncCamera:
    last_err = None
    for idx in [want_idx] + [i for i in range(8) if i != want_idx]:
        try:
            cam = AsyncCamera(idx, w, h, fps)
            if idx != want_idx:
                print(f"[cam] /dev/video{want_idx} unusable; using /dev/video{idx}")
            print(f"[cam] /dev/video{idx} (async): {cam.shape[1]}x{cam.shape[0]}")
            return cam
        except Exception as e:
            last_err = e
            print(f"[cam] /dev/video{idx} skipped: {e}")
    raise SystemExit(f"no usable camera; last error: {last_err}")


# =============================================================================
# observation / action plumbing
# =============================================================================

STATE_KEY = "observation.state"
IMG_TOP = "observation.images.camera_top"
IMG_WL = "observation.images.camera_wrist_left"
IMG_WR = "observation.images.camera_wrist_right"


def read_arm_state(arm, gmap: GripperMap, gripper_rad: float):
    """Read xArm EEF pose + gripper -> (state_10, xyz_m, rpy_rad).

    xArm get_position (is_radian=True) returns [x,y,z (mm), roll,pitch,yaw (rad)].
    Model wants xyz in METERS + rot6d + gripper channel.
    """
    code, pose = arm.get_position(is_radian=True)
    if code != 0 or pose is None:
        raise RuntimeError(f"arm.get_position code={code}")
    xyz_mm = np.asarray(pose[:3], dtype=np.float64)
    rpy = np.asarray(pose[3:6], dtype=np.float64)
    xyz_m = xyz_mm / 1000.0
    rot6d = rpy_to_rot6d(rpy)
    g_model = gmap.rad_to_model(gripper_rad)
    state = np.concatenate([xyz_m, rot6d, [g_model]]).astype(np.float32)  # (10,)
    return state, xyz_m, rpy


def load_state_bounds(norm_path: str):
    """Load [min, max] for the 9-dim arm state + 1-dim gripper from umi.json, so we
    can warn if a live pose is outside the training distribution (a frame/units
    mismatch shows up here immediately). Returns (arm_min, arm_max, grip_min, grip_max)
    or None if the file is unavailable."""
    import json
    try:
        ns = json.load(open(norm_path)).get("norm_stats", {})
        arm = ns["observation.state.arm.position"]
        eff = ns["observation.state.effector.position"]
        return (np.asarray(arm["min"]), np.asarray(arm["max"]),
                float(eff["min"][0]), float(eff["max"][0]))
    except Exception as e:
        print(f"[rollout] (skip workspace check: {e})")
        return None


def check_in_workspace(state: np.ndarray, bounds, margin: float = 0.15):
    """Print a loud warning if the live state falls well outside training min/max."""
    if bounds is None:
        return
    amin, amax, gmin, gmax = bounds
    span = np.maximum(amax - amin, 1e-6)
    lo, hi = amin - margin * span, amax + margin * span
    arm = state[:9]
    bad = np.where((arm < lo) | (arm > hi))[0]
    if len(bad):
        print("[rollout] !!! WARNING: live arm state is OUTSIDE the training range on "
              f"dims {bad.tolist()} (idx 0-2=xyz[m], 3-8=rot6d).")
        print(f"           live: {np.round(arm, 3).tolist()}")
        print(f"           train min: {np.round(amin, 3).tolist()}")
        print(f"           train max: {np.round(amax, 3).tolist()}")
        print("           => likely a FRAME or UNITS mismatch. Verify before executing.")
    else:
        print("[rollout] live arm pose is within the training workspace. good.")


def decode_action_chunk(action: np.ndarray):
    """(T,10) [xyz(m), rot6d, gripper] -> (xyz_mm (T,3), rpy (T,3), grip (T,))."""
    action = np.asarray(action, dtype=np.float64)
    xyz_mm = action[:, :3] * 1000.0
    rpy = np.stack([rot6d_to_rpy(action[i, 3:9]) for i in range(action.shape[0])])
    grip = action[:, 9]
    return xyz_mm, rpy, grip


# =============================================================================
# main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="LingBot-VLA xArm7 + UMI rollout")
    # policy server
    ap.add_argument("--host", default="127.0.0.1", help="policy server host")
    ap.add_argument("--port", type=int, default=8000, help="policy server port")
    ap.add_argument("--robo-name", default="umi",
                    help="robot config name the server loads (configs/robot_configs/<name>.yaml)")
    ap.add_argument("--task",
                    default="Pick up the small circular item and put it in the cup on the left.",
                    help="language instruction (must match training)")
    # arm
    ap.add_argument("--arm-ip", default="192.168.1.226", help="xArm 7 controller IP")
    ap.add_argument("--chunk-hz", type=float, default=15.0,
                    help="servo streaming rate for a chunk (Hz)")
    ap.add_argument("--horizon", type=int, default=25,
                    help="steps of each returned chunk to execute (<= server --use_length)")
    ap.add_argument("--servo-speed", type=float, default=100.0,
                    help="set_servo_cartesian speed (mm/s)")
    ap.add_argument("--servo-acc", type=float, default=2000.0,
                    help="set_servo_cartesian acceleration (mm/s^2)")
    ap.add_argument("--max-step-mm", type=float, default=40.0,
                    help="SAFETY: abort a chunk if any single servo step jumps "
                         "more than this many mm from the previous target")
    # camera
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--no-show", action="store_true", help="don't open the cv2 preview window")
    # gripper
    ap.add_argument("--gripper-can", default="can_follower_r", help="Damiao gripper CAN channel")
    ap.add_argument("--umi-path", default="/home/pierre/Desktop/umi-fastrobo",
                    help="dir containing umi.py (get_umi)")
    ap.add_argument("--no-gripper", action="store_true", help="skip gripper hardware entirely")
    ap.add_argument("--gripper-close-rad", type=float, default=0.0,
                    help="motor rad when jaws grasp/closed (calibrate with umi.py)")
    ap.add_argument("--gripper-open-rad", type=float, default=1.0,
                    help="motor rad when jaws fully open (calibrate with umi.py)")
    ap.add_argument("--model-gripper-close", type=float, default=0.27,
                    help="model gripper channel value at closed (dataset min ~0.27)")
    ap.add_argument("--model-gripper-open", type=float, default=1.0,
                    help="model gripper channel value at open (dataset max ~1.0)")
    # sanity / loop control
    ap.add_argument("--norm-path", default="assets/norm_stats/umi.json",
                    help="training norm stats; used only to sanity-check that the live "
                         "arm pose lands inside the training workspace (frame/units guard)")
    ap.add_argument("--auto", action="store_true", help="skip prompts; auto-execute every chunk")
    ap.add_argument("--max-chunks", type=int, default=50,
                    help="stop after this many EXECUTED chunks (0 = unlimited)")
    ap.add_argument("--settle-secs", type=float, default=0.3,
                    help="sleep after each executed chunk before next observation")
    args = ap.parse_args()

    gmap = GripperMap(args.model_gripper_close, args.model_gripper_open,
                      args.gripper_close_rad, args.gripper_open_rad)

    # ---- connect to policy server first (fail fast on a bad server) ----
    print(f"[rollout] connecting to policy server ws://{args.host}:{args.port} ...")
    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"[rollout] server metadata: {policy.get_server_metadata()}")
    policy.reset(args.robo_name)     # loads configs/robot_configs/<robo_name>.yaml server-side
    print(f"[rollout] server reset with robo_name={args.robo_name!r}")
    print(f"[rollout] task: {args.task!r}")

    import cv2

    arm = None
    gripper = None
    cam = None
    executed = 0
    lat_log: list[dict] = []
    try:
        # ---- hardware ----
        print(f"[rollout] connecting xArm7 @ {args.arm_ip} ...")
        from xarm.wrapper import XArmAPI
        arm = XArmAPI(args.arm_ip, is_radian=True)
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(0.3)
        code, pose0 = arm.get_position(is_radian=True)
        print(f"[rollout] arm ready. current pose (mm,rad): "
              f"{[round(v, 3) for v in pose0]}  (code {code})")

        # workspace/frame sanity check against training stats (arm dims only;
        # gripper not connected yet, so pass a dummy rad — it's ignored by the check)
        bounds = load_state_bounds(args.norm_path)
        state0, _, _ = read_arm_state(arm, gmap, args.gripper_open_rad)
        check_in_workspace(state0, bounds)

        if not args.no_gripper:
            print(f"[rollout] connecting UMI gripper (get_umi) from {args.umi_path} ...")
            sys.path.insert(0, args.umi_path)
            from umi import get_umi
            gripper = get_umi(args.gripper_can)
            g0 = float(gripper.get_joint_pos()[0])
            print(f"[rollout] gripper motor pos: {g0:+.4f} rad "
                  f"(-> model {gmap.rad_to_model(g0):.3f})")
        else:
            print("[rollout] --no-gripper: gripper channel of state fixed at 'open'")

        cam = find_async_camera(args.camera, args.width, args.height, 30)

        print()
        print("[rollout] === live rollout ===")
        if args.auto:
            print(f"[rollout] !!! --auto: every chunk executes; "
                  f"stop after {args.max_chunks or '∞'} chunks / Ctrl-C.")
        else:
            print("[rollout] you will be prompted [E]xecute / [S]kip / [Q]uit each chunk.")

        q_idx = 0
        while True:
            stage: dict[str, float] = {}
            t = time.time()

            # 1. observation
            grip_rad = float(gripper.get_joint_pos()[0]) if gripper is not None \
                else args.gripper_open_rad
            state, xyz_m, rpy = read_arm_state(arm, gmap, grip_rad)
            rgb = cam.read_rgb()
            stage["obs_ms"] = (time.time() - t) * 1000.0

            if not args.no_show:
                disp = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(disp, f"obs q{q_idx:02d} (fed to policy x3 cams)",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
                cv2.putText(disp, f"xyz(m) {xyz_m.round(3).tolist()}  grip {state[9]:.2f}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("lingbot obs (live)", disp)
                cv2.waitKey(1)

            # 2. build obs dict — one camera replicated into all three slots
            obs = {
                IMG_TOP: rgb,
                IMG_WL: rgb.copy(),
                IMG_WR: rgb.copy(),
                STATE_KEY: state,
                "task": args.task,
            }

            # 3. query policy
            t = time.time()
            out = policy.infer(obs)
            stage["infer_ms"] = (time.time() - t) * 1000.0
            if "action" not in out:
                raise RuntimeError(f"server returned no 'action' key; got {list(out)}")
            action = np.asarray(out["action"], dtype=np.float64)   # (T,10)
            T = min(args.horizon, action.shape[0])
            xyz_mm, rpy_chunk, grip = decode_action_chunk(action[:T])

            # step deltas vs current pose (safety + preview)
            cur_mm = xyz_m * 1000.0
            first_jump = float(np.linalg.norm(xyz_mm[0] - cur_mm))
            step_jumps = np.linalg.norm(np.diff(np.vstack([cur_mm, xyz_mm]), axis=0), axis=1)
            print(f"[rollout] q{q_idx:02d}: infer={stage['infer_ms']:.0f}ms  T={T}  "
                  f"end xyz(mm) {xyz_mm[-1].round(1).tolist()}  "
                  f"grip {grip[0]:.2f}->{grip[-1]:.2f}  "
                  f"first_jump={first_jump:.1f}mm  max_step={step_jumps.max():.1f}mm")

            # 4. prompt
            if args.auto:
                ans = "e"
            else:
                try:
                    ans = input(f"[rollout] q{q_idx}: [E]xecute / [S]kip / [Q]uit? ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    ans = "q"
            if ans == "q":
                print("[rollout] quit requested.")
                break
            if ans in ("s", ""):
                print("[rollout] skipped — re-querying with a fresh observation.")
                stage["executed"] = 0.0
                lat_log.append(stage)
                q_idx += 1
                continue

            # 5. SAFETY: reject wild chunks before switching to servo mode
            if step_jumps.max() > args.max_step_mm:
                print(f"[rollout] ABORT chunk: max step {step_jumps.max():.1f}mm > "
                      f"--max-step-mm {args.max_step_mm}. Skipping (re-query).")
                stage["executed"] = 0.0
                lat_log.append(stage)
                q_idx += 1
                continue

            # 6. stream chunk (servo cartesian, mode 1)
            print(f"[rollout] streaming {T} steps @ {args.chunk_hz:.0f} Hz (mode 1)...")
            arm.set_mode(1)
            arm.set_state(0)
            time.sleep(0.05)
            period = 1.0 / args.chunk_hz
            next_t = time.time()
            t = time.time()
            for i in range(T):
                pose_cmd = [float(xyz_mm[i, 0]), float(xyz_mm[i, 1]), float(xyz_mm[i, 2]),
                            float(rpy_chunk[i, 0]), float(rpy_chunk[i, 1]), float(rpy_chunk[i, 2])]
                code = arm.set_servo_cartesian(
                    pose_cmd, speed=args.servo_speed, mvacc=args.servo_acc, is_radian=True)
                if code != 0:
                    print(f"[rollout] set_servo_cartesian code {code} at step {i}; stopping chunk.")
                    break
                if gripper is not None:
                    gripper.command_joint_pos(np.array([gmap.model_to_rad(grip[i])]))
                next_t += period
                sl = next_t - time.time()
                if sl > 0:
                    time.sleep(sl)
                else:
                    next_t = time.time()
            stage["stream_ms"] = (time.time() - t) * 1000.0

            arm.set_mode(0)
            arm.set_state(0)
            time.sleep(0.05)
            if args.settle_secs > 0:
                time.sleep(args.settle_secs)

            stage["executed"] = 1.0
            lat_log.append(stage)
            executed += 1
            print(f"[rollout] q{q_idx} executed  ({executed} total).")
            q_idx += 1
            if args.max_chunks and executed >= args.max_chunks:
                print(f"[rollout] reached --max-chunks={args.max_chunks}, stopping.")
                break

    except KeyboardInterrupt:
        print("\n[rollout] KeyboardInterrupt.")
    finally:
        # latency summary
        if lat_log:
            ex = [r for r in lat_log if r.get("executed", 0) > 0]
            sk = [r for r in lat_log if r.get("executed", 0) == 0]
            print()
            print(f"=== latency summary (executed {len(ex)}, skipped {len(sk)}) ===")
            for k in ("obs_ms", "infer_ms", "stream_ms"):
                xs = [r[k] for r in lat_log if k in r]
                if xs:
                    a = np.asarray(xs)
                    print(f"  {k:<12} mean={a.mean():7.1f}  min={a.min():7.1f}  max={a.max():7.1f}")
        if arm is not None:
            try:
                arm.set_mode(0)
                arm.set_state(0)
                arm.disconnect()
            except Exception:
                pass
        if gripper is not None:
            try:
                gripper.close()
            except Exception:
                pass
        if cam is not None:
            cam.disconnect()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[rollout] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
