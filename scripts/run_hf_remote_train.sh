#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   HF_TOKEN=hf_xxx HF_REPO_ID=username/ptz-qwen-grpo \
#   bash scripts/run_hf_remote_train.sh
#
# Optional overrides:
#   TRAIN_STEPS=1000 GROUP_SIZE=4 MAX_NEW_TOKENS=12 bash scripts/run_hf_remote_train.sh

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required."
  exit 1
fi

if [[ -z "${HF_REPO_ID:-}" ]]; then
  echo "HF_REPO_ID is required. Example: username/ptz-qwen-grpo"
  exit 1
fi

export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-$HF_TOKEN}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

TRAIN_STEPS="${TRAIN_STEPS:-1000}"
GROUP_SIZE="${GROUP_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-12}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.9}"
LOG_EVERY="${LOG_EVERY:-1}"
SAVE_EVERY="${SAVE_EVERY:-100}"

echo "Starting remote-friendly training run..."
echo "Repo: $HF_REPO_ID | steps=$TRAIN_STEPS group=$GROUP_SIZE"

uv run python scripts/train_llm.py \
  --train-steps "$TRAIN_STEPS" \
  --group-size "$GROUP_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top-p "$TOP_P" \
  --log-every "$LOG_EVERY" \
  --save-every "$SAVE_EVERY" \
  --push-checkpoints-to-hub \
  --hub-repo-id "$HF_REPO_ID" \
  --hub-private
