import mujoco as mj
import numpy as np
import pickle
import pandas as pd
from pathlib import Path

# ------------------ Paths ------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
xml_path = PROJECT_ROOT / "assets" / "icu" / "hackathon_icu.xml"
# ------------------ Load model ------------------
model = mj.MjModel.from_xml_path(str(xml_path))
data = mj.MjData(model)
renderer = mj.Renderer(model, height=64, width=64)

# IDs for raycasting
cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "robot_camera")
bed_geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "bed")

# ------------------ Data buffer ------------------
dataset = []

# ------------------ Parameters ------------------
max_steps = 1000  # Simulation steps per action
num_episodes = 5000

# Action limits (Pan: -1.57 to 1.57, Tilt: -0.78 to 0.78)
u_min = np.array([-1.57, -0.78])
u_max = np.array([1.57, 0.78])

print("Starting data collection...")

for episode in range(num_episodes):
    # Sample random joint actions

    if np.random.rand() < 0.5:
        # Force camera to look near the bed coordinates
        u_action = np.random.uniform([-0.8, -0.6], [-0.2, -0.1])
    else:
        # Look anywhere
        u_action = np.random.uniform(u_min, u_max)
        
    data.ctrl[0] = u_action[0]
    data.ctrl[1] = u_action[1]
    
    # Step simulation to apply physical movement
    for t in range(max_steps): 
        mj.mj_step(model, data)

    # 1. Get Image
    renderer.update_scene(data, camera="robot_camera")
    img = renderer.render()

    # 2. Get Joint Angles
    q = np.array([data.qpos[0], data.qpos[1]], dtype=np.float32)
    qd = np.array([data.qvel[0], data.qvel[1]], dtype=np.float32)

    # 3. Raycast to find distance to the bed
    cam_pos = data.cam_xpos[cam_id]
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
    cam_vec = -cam_mat[:, 2] # Camera looks down its negative Z-axis

    geom_id_arr = np.array([-1], dtype=np.intc)
    dist = mj.mj_ray(model, data, cam_pos, cam_vec, None, 1, -1, geom_id_arr)

    # If the ray doesn't hit the bed, assign a safe maximum distance
    if geom_id_arr[0] != bed_geom_id:
        dist = 8.0  

    # 4. Append to Dataset
    dataset.append({
        "image": img,
        "q": q,
        "qd": qd,
        "u": u_action,
        "distance": np.float32(dist)
    })

    if (episode + 1) % 500 == 0:
        print(f"Progress: {episode + 1}/{num_episodes}")

print("Total samples:", len(dataset))

# ------------------ Save ------------------
PROJECT_ROOT.joinpath("data").mkdir(parents=True, exist_ok=True)

# Save as pickle
with open(PROJECT_ROOT / "data" / "offline_dataset.pkl", "wb") as f:
    pickle.dump(dataset, f)

print("Dataset saved.")