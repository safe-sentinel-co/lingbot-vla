.PHONY: build commit quality style test serve rollout rollout-auto

check_dirs := tasks tests lingbot docs setup.py

# ---- inference / real-robot deployment (UMI single-cam + rot6d model) --------
# The policy server needs the VLA env (torch 2.8 / transformers); the rollout
# client needs the robot env (xarm + i2rt). Defaults point at the two conda envs
# set up for this box; override any var on the command line, e.g.:
#   make serve CKPT=/path/to/global_step_4000/hf_ckpt PORT=8001
#   make rollout ARM_IP=192.168.1.226 CAMERA=2
SERVER_PY   ?= $(HOME)/miniconda3/envs/lingbotvla/bin/python
CLIENT_PY   ?= $(HOME)/miniconda3/envs/umi-real/bin/python
QWEN25_PATH ?= $(HOME)/models/Qwen2.5-VL-3B-Instruct
CKPT        ?= $(HOME)/models/umi_real_depth/global_step_8000/hf_ckpt
NORM        ?= assets/norm_stats/umi.json
HOST        ?= 127.0.0.1
PORT        ?= 8000
USE_LENGTH  ?= 25
DENOISE     ?= 10
ARM_IP      ?= 192.168.1.226
CAMERA      ?= 0
TASK        ?= Pick up the small circular item and put it in the cup on the left.

# Start the LingBot-VLA websocket policy server (run in the VLA env).
serve:
	QWEN25_PATH=$(QWEN25_PATH) PYTHONNOUSERSITE=1 $(SERVER_PY) -m deploy.lingbot_vla_policy \
		--model_path $(CKPT) --norm_path $(NORM) \
		--use_length $(USE_LENGTH) --num_denoising_step $(DENOISE) --port $(PORT)

# Interactive xArm7 + UMI rollout client (run in the robot env). Prompts E/S/Q per chunk.
rollout:
	$(CLIENT_PY) -m deploy.xarm_umi_rollout \
		--host $(HOST) --port $(PORT) --arm-ip $(ARM_IP) --camera $(CAMERA) \
		--task "$(TASK)"

# Same, unattended: auto-execute every chunk (use only after validating in interactive mode).
rollout-auto:
	$(CLIENT_PY) -m deploy.xarm_umi_rollout \
		--host $(HOST) --port $(PORT) --arm-ip $(ARM_IP) --camera $(CAMERA) \
		--task "$(TASK)" --auto --max-chunks 50


build:
	python3 setup.py sdist bdist_wheel

commit:
	pre-commit install
	pre-commit run --all-files

quality:
	ruff check $(check_dirs)
	ruff format --check $(check_dirs)

style:
	ruff check $(check_dirs) --fix
	ruff format $(check_dirs)

test:
	pytest tests/
