---
title: Camera PTZ Alignment Environment
emoji: 📹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
app_file: server/app.py
base_path: /web
tags:
  - openenv
  - pytorch-hackathon
  - reinforcement-learning
  - continuous-control
---

# Camera PTZ Alignment Environment 🏥

A high-precision reinforcement learning environment developed for the **Meta PyTorch OpenEnv Hackathon 2026**. This project simulates the real-world challenge of aligning drifting Hospital Security PTZ (Pan-Tilt-Zoom) cameras back to their calibrated presets.

---

## 📖 Problem Statement

In large-scale facilities like hospitals, PTZ cameras often lose their alignment due to mechanical wear, physical contact, or vibrations. Maintaining precise coverage is critical for patient safety and security. This environment allows an AI agent to act as a "Virtual Operator," correcting camera drift through iterative, high-precision adjustments.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/JanaksinhVen/camera-ptz-alignment.git
cd camera-ptz-alignment

# Sync dependencies and create a virtual environment (.venv) based on uv.loc and pyproject.toml file (Prerequisite: install uv in your system)
uv sync
```

---

## ▶️ Basic Usage

The following snippet demonstrates how to interact with the environment using the provided `CameraPresetEnv` client.

```python
import asyncio
from client import CameraPresetEnv, CameraAction

async def main():
    # 1. Initialize environment from Docker image
    async with await CameraPresetEnv.from_docker_image("camera-ptz-alignment:latest") as env:
        
        # 2. Reset with a specific difficulty task
        result = await env.reset(task_id="human_touch_medium")
        print(f"Initial Drift Distance: {result.observation.distance_to_target:.4f}")

        # 3. Apply an alignment nudge (Pan Right, Tilt Down)
        action = CameraAction(delta_pan=0.15, delta_tilt=-0.05, delta_zoom=0.0)
        result = await env.step(action)
        
        print(f"New Distance: {result.observation.distance_to_target:.4f}")
        print(f"Reward Received: {result.reward:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🛠️ Reinforcement Learning MDP Details

### 1. Task Difficulties (Episodes)

The environment supports three levels of initial drift, defined in `openenv.yaml`:

- **glitch_easy**: 10% Initial Drift  
  _Simulates minor software or jitter errors._

- **human_touch_medium**: 30% Initial Drift  
  _Simulates physical contact with the camera housing._

- **hardware_drift_hard**: 70% Initial Drift  
  _Simulates severe mechanical misalignment or motor wear._

---

### 2. Action Space (Continuous Control)

The agent provides a `CameraAction` with three continuous values in the range `[-1.0, 1.0]`:

- `delta_pan`: Horizontal movement  
- `delta_tilt`: Vertical movement  
- `delta_zoom`: Focal adjustment  

---

### 3. Observation Space

The `CameraObservation` provides the agent with the necessary state to compute the next move:

- `current_ptz`: `[pan, tilt, zoom]` current coordinates  
- `target_ptz`: `[pan, tilt, zoom]` goal coordinates (Preset)  
- `distance_to_target`: Euclidean distance error between current and target  

---

## 🏆 Reward Policy: Progress-Based Shaping

The environment uses **Dense Reward Shaping** to provide immediate feedback after every action.

---

### 1. The Progress Formula (Dense)

**Math:**

```
Reward = (Previous_Error - Current_Error) * 10.0
```

**Logic:**

- Moving **closer** → ✅ Positive reward  
- Moving **away** → ❌ Negative reward  

**Scaling:**

- Multiplier `10.0` amplifies small improvements  

---

### 2. The Success Jackpot (Sparse)

**Condition:**

```
Current_Error < ALIGNMENT_THRESHOLD (0.03)
```

**Bonus:**

- `+10.0`

**Result:**

- Episode ends (`done = True`)

**Purpose:**

- Encourages precise alignment quickly  

---

### 3. Physical Constraints

- **Max Velocity Scaling:** `0.2` (20%)

**Purpose:**

- Prevents unrealistic movement  
- Encourages the agent to learn smooth, realistic motor control trajectories.

---

## 📦 Deployment & Validation

---

### Docker Build

```bash
docker build -t camera-ptz-alignment:latest .
```

### Docker Run (Local)
```bash
docker run -it -p 8000:8000 camera-ptz-alignment:latest
```
---

### Local Validation

```bash
uv run openenv validate .

[OK] : Ready for multi-mode deployment
```
---

### Baseline Inference Proof

```bash
uv run python inference.py

[START] task=glitch_easy env=camera_ptz_alignment_v1 model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=move(-0.13,0.09,0.19) reward=0.49 done=false error=null
[STEP] step=2 action=move(-0.05,0.05,0.05) reward=0.17 done=false error=null
[STEP] step=3 action=move(-0.09,0.07,0.14) reward=0.36 done=false error=null
[STEP] step=4 action=move(-0.05,0.05,0.05) reward=0.16 done=false error=null
[STEP] step=5 action=move(-0.03,0.02,0.05) reward=0.12 done=false error=null
[STEP] step=6 action=move(-0.06,0.04,0.09) reward=0.23 done=false error=null
[STEP] step=7 action=move(-0.04,0.03,0.07) reward=0.18 done=false error=null
[STEP] step=8 action=move(-0.04,0.02,0.06) reward=0.15 done=false error=null
[STEP] step=9 action=move(-0.03,0.02,0.05) reward=0.12 done=false error=null
[STEP] step=10 action=move(-0.02,0.02,0.04) reward=0.09 done=false error=null
[STEP] step=11 action=move(-0.02,0.01,0.03) reward=0.08 done=false error=null
[STEP] step=12 action=move(-0.01,0.01,0.02) reward=10.06 done=true error=null
[END] success=true steps=12 score=0.902 rewards=0.49,0.17,0.36,0.16,0.12,0.23,0.18,0.15,0.12,0.09,0.08,10.06
[START] task=human_touch_medium env=camera_ptz_alignment_v1 model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=move(0.40,0.46,-0.22) reward=1.30 done=false error=null
[STEP] step=2 action=move(0.32,0.37,-0.17) reward=1.04 done=false error=null
[STEP] step=3 action=move(0.26,0.30,-0.14) reward=0.83 done=false error=null
[STEP] step=4 action=move(0.21,0.24,-0.11) reward=0.67 done=false error=null
[STEP] step=5 action=move(0.10,0.10,-0.05) reward=0.30 done=false error=null
[STEP] step=6 action=move(0.15,0.17,-0.08) reward=0.47 done=false error=null
[STEP] step=7 action=move(0.04,0.04,-0.02) reward=0.12 done=false error=null
[STEP] step=8 action=move(0.05,0.05,-0.05) reward=0.17 done=false error=null
[STEP] step=9 action=move(0.05,0.05,-0.05) reward=0.16 done=false error=null
[STEP] step=10 action=move(0.09,0.11,-0.04) reward=0.29 done=false error=null
[DEBUG] Failed task human_touch_medium: no close frame received or sent
[START] task=hardware_drift_hard env=camera_ptz_alignment_v1 model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=move(-0.10,0.20,0.02) reward=0.44 done=false error=null
[STEP] step=2 action=move(-0.10,0.10,0.05) reward=0.28 done=false error=null
[STEP] step=3 action=move(-0.10,0.10,0.01) reward=0.28 done=false error=null
[STEP] step=4 action=move(-0.10,0.10,0.01) reward=0.28 done=false error=null
[STEP] step=5 action=move(-0.11,0.17,0.01) reward=0.41 done=false error=null
[STEP] step=6 action=move(-0.05,0.05,0.01) reward=0.14 done=false error=null
[STEP] step=7 action=move(-0.05,0.05,0.00) reward=0.14 done=false error=null
[STEP] step=8 action=move(-0.05,0.05,0.01) reward=0.14 done=false error=null
[STEP] step=9 action=move(-0.05,0.05,-0.01) reward=0.14 done=false error=null
[STEP] step=10 action=move(-0.05,0.10,0.02) reward=0.22 done=false error=null
[STEP] step=11 action=move(-0.04,0.07,-0.00) reward=0.16 done=false error=null
[STEP] step=12 action=move(-0.03,0.06,-0.00) reward=0.14 done=false error=null
[STEP] step=13 action=move(-0.03,0.05,-0.00) reward=0.11 done=false error=null
[STEP] step=14 action=move(-0.02,0.02,-0.01) reward=0.05 done=false error=null
[STEP] step=15 action=move(-0.02,0.04,0.01) reward=0.08 done=false error=null
[STEP] step=16 action=move(-0.01,0.03,-0.00) reward=10.06 done=true error=null
[END] success=true steps=16 score=0.922 rewards=0.44,0.28,0.28,0.28,0.41,0.14,0.14,0.14,0.14,0.22,0.16,0.14,0.11,0.05,0.08,10.06
```
---

## 📂 Project Structure

- `server/preset_env.py` → Core logic for drift generation and reward calculation  
- `server/app.py` → FastAPI & WebSocket entry points  
- `models.py` → Pydantic schemas  
- `client.py` → OpenEnv client  
- `openenv.yaml` → Task definitions  
- `inference.py` → Baseline agent demo  

---

## ✍️ Authors (Team: fp16)

**Janaksinh Ven, Anubhav Tripathi, Shivam Singh**

**Submission:** Meta PyTorch OpenEnv Hackathon 2026
