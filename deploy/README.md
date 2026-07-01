# Running the LingBot-VLA Policy Server

The deployment is split into **two processes** that talk over a websocket:

```
┌─────────────────────────────┐         ws (msgpack)         ┌──────────────────────────────┐
│  POLICY SERVER               │  <───────────────────────>  │  ROLLOUT CLIENT               │
│  deploy.lingbot_vla_policy   │   obs  ─────────────────►    │  deploy.xarm_umi_rollout      │
│  • big GPU                   │   ◄───────────── action      │  • xArm 7 + UMI gripper + cam │
│  • lingbotvla env            │      chunk (de-normalized)   │  • umi-real env, no GPU       │
└─────────────────────────────┘                              └──────────────────────────────┘
```

The server owns the model, normalization, feature transform and de-normalization.
The client owns the hardware. **Run the client on the machine wired to the robot;
run the server wherever there's enough GPU** — they don't have to be the same box.

See [`../INFERENCE.md`](../INFERENCE.md) for the model/observation contract and
[`xarm_umi_rollout.py`](xarm_umi_rollout.py) for the client.

---

## 1. Hardware requirements (read this first)

The `umi_real_depth` checkpoint is **4.22 B params stored in float32 → 16.9 GB on
disk**. How it loads determines whether it fits:

| | fp32 (as-is loader) | bf16 |
|---|---|---|
| GPU VRAM | 16.9 GB + overhead | **~8.4 GB** |
| Peak host RAM during load | ~34 GB | ~9 GB (with the low-mem loader) |

- **> 24 GB GPU** (A5000/3090/4090/A100/…): the stock loader works as-is. Recommended.
- **16 GB GPU** (e.g. RTX 5080) with 32 GB RAM: the stock loader **OOMs** — it reads
  all shards in fp32 (16.9 GB RAM), builds a second fp32 copy (~34 GB peak RAM), then
  moves fp32 to a 16 GB GPU. You must load in bf16 (see [Troubleshooting](#5-troubleshooting)).

If the robot box has a small GPU, the intended answer is: **run the server on a
bigger box** and point the client at it (Section 4).

---

## 2. Environment

The server needs the full VLA stack (torch 2.8, transformers 4.51.3, lerobot). Keep
it in its own env so it never disturbs the robot/`umi-real` env.

```bash
conda create -y -n lingbotvla python=3.10
conda activate lingbotvla
pip install -e .          # from repo root; pulls torch 2.8, transformers 4.51.3, etc.
pip install 'lerobot==0.4.4'
```

> **lerobot must be ≥ 0.4** — `pip install -e .` alone backtracks to `lerobot 0.3.2`,
> whose `PI0Config` schema is too old and rejects the checkpoint's `config.json`
> (`num_inference_steps`, `rtc_config`, `image_resolution`, …). 0.4.4 has the matching
> superset schema. (0.4.4 pulls `numpy 2.x`/`datasets 4.x`, overriding the repo pins —
> harmless for the server; the `pi0` config registers when the model module imports.)

### flash-attn (optional)

The model is trained with `flash_attention_2`. The code now **auto-detects** the
`flash_attn` package and falls back to **eager** attention if it's missing
(`modeling_lingbot_vla.py` and `qwenvl_in_vla.py`). So the server runs without it.

- Eager: works everywhere, slower per inference — fine for testing / open-loop eval.
- flash-attn: needed for real-time robot control latency. Install the prebuilt wheel
  matching your torch/CUDA/python/ABI, e.g. for torch 2.8 / cu12 / cp310 / cxx11abiTRUE:

  ```bash
  pip install 'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl'
  ```

  Check your tags with:
  `python -c "import torch,sys;print(torch.__version__,torch.version.cuda,f'cp{sys.version_info.major}{sys.version_info.minor}',torch._C._GLIBCXX_USE_CXX11_ABI)"`

---

## 3. Model assets

The server reads two repo files by **relative path**, so launch from the repo root:

- `configs/robot_configs/umi.yaml` — maps the 10-dim UMI space `[xyz(3), rot6d(6), gripper(1)]`
  onto `arm.position` (9) + `effector.position` (1) and the 3 camera slots.
- `assets/norm_stats/umi.json` — `meanstd` stats for state/action normalization.

Plus two large downloads:

```bash
# Qwen2.5-VL-3B base (tokenizer + vision config the checkpoint references)
hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir /home/pierre/models/Qwen2.5-VL-3B-Instruct

# a checkpoint (hf_ckpt) from S3 — ~16 GB each
aws s3 cp --recursive \
  s3://safesentinel-inc/lingbot-vla/umi_real_depth/global_step_8000/hf_ckpt/ \
  /home/pierre/models/umi_real_depth/global_step_8000/hf_ckpt/
```

The checkpoint's `lingbotvla_cli.yaml` records `data_name: umi` and
`norm_stats_file: assets/norm_stats/umi.json`, which is why the two repo files above
must exist and match.

---

## 4. Start the server

`QWEN25_PATH` must point at the Qwen base. Launch from the repo root.

```bash
conda activate lingbotvla
export QWEN25_PATH=/home/pierre/models/Qwen2.5-VL-3B-Instruct
export PYTHONNOUSERSITE=1        # avoid ~/.local site-packages leaking in

python -m deploy.lingbot_vla_policy \
    --model_path /home/pierre/models/umi_real_depth/global_step_8000/hf_ckpt \
    --norm_path assets/norm_stats/umi.json \
    --use_length 25 \
    --num_denoising_step 10 \
    --port 8000
# --use_compile   # optional: torch.compile for speed after warm-up
```

Or via the Makefile (defaults are overridable):

```bash
make serve CKPT=/home/pierre/models/umi_real_depth/global_step_8000/hf_ckpt PORT=8000
```

Flags:
- `--use_length` — how many steps of each 50-step chunk the client executes before
  re-planning (`-1` = action ensembling).
- `--num_denoising_step` — flow-matching steps (lower = faster, slightly lower quality).

The server binds `0.0.0.0`, so it's reachable over the network once port 8000 is open.

### Connecting the client

- **Same machine:** `make rollout HOST=127.0.0.1 PORT=8000`
- **Server on another box (LAN):** `make rollout HOST=<server-ip> PORT=8000`
  (open port 8000 on the server firewall)
- **No open ports (SSH tunnel):** on the robot box
  `ssh -L 8000:localhost:8000 user@server-box`, then `make rollout HOST=127.0.0.1 PORT=8000`

### Verify it's up (no robot needed)

Hit the running server with a synthetic observation to exercise the full pipeline:

```bash
python /tmp/smoke_test_umi_server.py     # connects, reset('umi'), one infer, checks (T,10) action
```

---

## 5. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `DecodingError: fields ... not valid for PI0Config` | lerobot too old (0.3.2). `pip install 'lerobot==0.4.4'`. |
| `Couldn't find a choice class for 'pi0'` | `pi0` registers on model-module import; the server import order handles it. Reproduce by importing `lingbotvla.models.vla.pi0.modeling_lingbot_vla` before `PreTrainedConfig.from_pretrained`. |
| `flash_attn seems to be not installed` | Fixed by the eager auto-fallback in the model code. For speed, install the flash-attn wheel (Section 2). |
| Process dies during load with **no traceback**, machine freezes | **OOM.** On a 16 GB GPU / 32 GB RAM box the fp32 load path peaks ~34 GB RAM and 16.9 GB VRAM. Either run the server on a bigger GPU, or load in bf16: cast each shard to `bfloat16` on read, build the policy on the `meta` device and `load_state_dict(..., assign=True)`, then `.cuda()` (peak ~8.4 GB VRAM / ~9 GB RAM). |
| Client `Still waiting for server...` | Wrong host/port, firewall, or the server is still loading weights (can take a minute). |

---

## Quick reference

```bash
# SERVER (big-GPU box, lingbotvla env, from repo root)
make serve CKPT=/path/to/global_step_8000/hf_ckpt PORT=8000

# CLIENT (robot box, umi-real env)
make rollout HOST=<server-ip> PORT=8000 ARM_IP=192.168.1.226 CAMERA=0
```
