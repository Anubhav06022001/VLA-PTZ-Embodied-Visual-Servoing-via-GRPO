# Hugging Face Remote Training (MuJoCo + Qwen GRPO)

This guide runs your current `scripts/train_llm.py` remotely on Hugging Face hardware while keeping the same codebase.

## 1) Decide your topology

You have two valid options:

- **Option A (recommended first):** run MuJoCo inside remote container itself.  
  Your training and environment both run on HF compute.
- **Option B:** keep MuJoCo local and stream transitions to remote trainer via an API queue.  
  This is more complex and usually unnecessary unless local hardware/sensors must stay local.

The steps below are for **Option A**.

## 2) Create a Docker Space on Hugging Face

1. Go to [https://huggingface.co/new-space](https://huggingface.co/new-space)
2. Set:
   - **Owner:** your account
   - **Space name:** e.g. `ptz-grpo-train`
   - **SDK:** Docker
   - **Visibility:** Private recommended
3. Open Space **Settings -> Hardware** and select a GPU tier that fits your budget.

## 3) Push your repo to the Space

From your local project root:

```bash
git remote add hf https://huggingface.co/spaces/<username>/<space_name>
git push hf HEAD:main
```

If `hf` remote already exists, update:

```bash
git remote set-url hf https://huggingface.co/spaces/<username>/<space_name>
git push hf HEAD:main
```

## 4) Add Space secrets/env vars

In Space **Settings -> Variables and secrets** add:

- `HF_TOKEN` = your Hugging Face token
- `HUGGINGFACE_HUB_TOKEN` = same token (optional but recommended)
- `MUJOCO_GL` = `egl`
- `PYOPENGL_PLATFORM` = `egl`

## 5) Start training remotely

Use the helper script added in this repo:

```bash
HF_TOKEN=hf_xxx \
HF_REPO_ID=<username>/ptz-qwen-grpo \
bash scripts/run_hf_remote_train.sh
```

This will:

- run `scripts/train_llm.py`
- save checkpoints in `models/`
- upload checkpoints to your model repo via `--push-checkpoints-to-hub`

## 6) Monitor that training is healthy

In logs, watch for:

- `valid_action_rate` trending above `0.0`
- reward variation (not constant every step)
- non-zero loss at least on some steps

If `valid_action_rate` stays 0 for long periods, reduce `max_new_tokens` and `temperature`.

## 7) Resume from a saved checkpoint

Current script saves checkpoints with:

`models/ptz_qwen_grpo_step_<N>.pt`

You can download from your HF model repo under `checkpoints/` and later extend training (resume hook can be added next if you want).

## 8) Budget-safe run strategy

Start with short bursts:

```bash
TRAIN_STEPS=100 GROUP_SIZE=4 SAVE_EVERY=50 bash scripts/run_hf_remote_train.sh
```

Then scale only if metrics improve.

## 9) If your data must stay from local live MuJoCo

Use Option B:

- Keep MuJoCo loop on local machine.
- Send `(similarity_vector, pan, tilt, reward)` batches to remote trainer API.
- Remote side updates model and periodically sends back latest policy/projector weights.

If you want this architecture, add a small FastAPI producer/consumer pair next.
