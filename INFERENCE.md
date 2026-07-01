# Running Inference with LingBot-VLA

This guide covers how to run inference/evaluation with a post-trained LingBot-VLA
checkpoint. It includes the general flow plus the setup-specific notes for a
**single-camera UMI** model trained with a `[xyz, rot6d, gripper]` action space
and depth injection.

## Prerequisites

Run from the repo root with the environment active:

```bash
source .venv/bin/activate            # or: conda activate lingbotvla
export QWEN25_PATH=/path/to/Qwen2.5-VL-3B-Instruct
```

`--model_path` must point at an **`hf_ckpt`** directory — the one produced at each
checkpoint under `train.output_dir/checkpoints/global_step_*/hf_ckpt`. It must contain:

- `*.safetensors` weight shards + `model.safetensors.index.json`
- `config.json`
- `lingbotvla_cli.yaml`   (records the training data/robot config, norm stats, cameras, joints)
- tokenizer / preprocessor files

Example (this project):

```bash
CKPT=output/umi_real_depth/checkpoints/global_step_2000/hf_ckpt
```

---

## 1. Open-Loop Evaluation (recommended sanity check)

Compares the policy's predicted action chunks against ground-truth trajectories
from a dataset and saves comparison plots. No robot required.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/open_loop_eval.py \
    --model_path $CKPT \
    --data_path /path/to/lerobot_v3_dataset \
    --norm_path assets/norm_stats/<name>.json \
    --traj_ids 0 1 2 \
    --use_length 50 \
    --num_denoising_step 10 \
    --save_plot_path ./open_loop_test/
```

- If `--data_path` is omitted, it defaults to the `data.train_path` recorded in the
  checkpoint's `lingbotvla_cli.yaml`.
- `--use_length` = number of action steps used from each predicted chunk (chunk size is 50).
- `--num_denoising_step` = flow-matching denoising steps (lower = faster, slightly lower quality).
- Add `--use_compile` to enable `torch.compile`.

Concrete example for this project:

```bash
export QWEN25_PATH=/home/pierre/models/Qwen2.5-VL-3B-Instruct
CUDA_VISIBLE_DEVICES=0 python scripts/open_loop_eval.py \
    --model_path output/umi_real_depth/checkpoints/global_step_2000/hf_ckpt \
    --data_path /home/pierre/data/combined-umi-filtered \
    --norm_path assets/norm_stats/umi.json \
    --traj_ids 0 1 2 --use_length 50 --num_denoising_step 10 \
    --save_plot_path ./open_loop_test/
```

---

## 2. Real-Robot Deployment (policy server)

Starts a WebSocket policy server; your robot client streams observations and
receives action chunks.

```bash
export QWEN25_PATH=/path/to/Qwen2.5-VL-3B-Instruct
python -m deploy.lingbot_vla_policy \
    --model_path $CKPT \
    --use_compile \
    --use_length 25 \
    --port 8000
# --num_denoising_step 5   # optional: faster inference
```

- `--use_length` controls how many steps of each predicted chunk are executed
  before re-planning (`-1` enables action ensembling).

---

## 3. Simulation Deployment (RoboTwin)

```bash
export QWEN25_PATH=/path/to/Qwen2.5-VL-3B-Instruct
python -m deploy.lingbot_vla_policy \
    --model_path $CKPT \
    --use_compile \
    --use_length 50 \
    --port <port>
```

---

## ⚠️ Setup-specific notes (single-camera UMI + rot6d model)

This model was post-trained with a customized data representation. Inference input
and output must match it:

1. **Cameras — 3 slots expected.** The model (and depth alignment) are built for 3
   cameras. With a single UMI camera, feed the real image as **`camera_top`**; the
   two wrist slots are zero-filled and masked out automatically (as in training).
   The robot config `configs/robot_configs/umi.yaml` handles this mapping for
   open-loop eval. For the live server, send the image under the `camera_top` key.

2. **State input** must be **`[x, y, z, rot6d(0..5), gripper]`** (10-dim), normalized
   with the training norm stats (`assets/norm_stats/umi.json`, `meanstd`). Convert
   your live end-effector **quaternion → 6D rotation** exactly as
   `scripts/convert_umi_quat_to_rot6d.py` does before sending.

3. **Action output** is returned in that same normalized 10-dim space. To command a
   robot you must **invert** it:
   - de-normalize with the norm stats,
   - convert the **6D rotation back to a quaternion / rotation matrix**
     (Gram-Schmidt on the two 3-vectors),
   - map the **gripper** channel to your gripper command (continuous; ≈ **1.0 = open**,
     ≈ **0.5 = grasping** the object; dataset range ≈ 0.27–1.0).

4. **Depth is not needed at inference.** MoGe / LingBot-Depth are training-time
   auxiliaries (depth-feature distillation). The policy predicts actions from
   RGB + state + language only.

5. **Task instruction.** The language prompt is read from the dataset's
   `meta/tasks.parquet` for open-loop eval. For live deployment, pass the same
   instruction the model was trained on:
   *"Pick up the small circular item and put it in the cup on the left."*

## Choosing a checkpoint

With a small single-task dataset the VLA loss overfits quickly, so the best policy
is often an **early** checkpoint rather than the final one. Open-loop-eval several
(`global_step_2000`, `4000`, `6000`, …) and pick the best before committing to a
real-robot run.
