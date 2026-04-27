import mujoco as mj
import numpy as np
import torch
import sys
from pathlib import Path
from openenv.core.env_server import Environment, Action, Observation, State

# ------------------ Paths & Imports ------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Assuming your network class is inside the training script based on your previous test
from scripts.train_vocbf import VOCBFNet 
from src.qp import cbf_qp_osqp

# ------------------ OpenEnv Spaces ------------------
class ICUAction(Action):
    pan_target: float
    tilt_target: float

class ICUObservation(Observation):
    current_pan: float
    current_tilt: float
    monitor_distance: float
    reward: float
    done: bool

class ICUState(State):
    steps: int = 0

# ------------------ The Environment ------------------
class HackathonICUEnv(Environment):
    def __init__(self):
        super().__init__()
        
        # 1. Load MuJoCo
        xml_path = PROJECT_ROOT / "assets" / "franka_emika_panda" / "hackathon_icu.xml"
        # xml_path = PROJECT_ROOT / "assets" / "hackathon_icu.xml"
        print("XML PATH:", xml_path)
        print("EXISTS:", xml_path.exists())
        self.model = mj.MjModel.from_xml_path(str(xml_path))
        self.data = mj.MjData(self.model)
        self.renderer = mj.Renderer(self.model, height=64, width=64)
        
        # 2. Setup Geometry IDs
        self.cam_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, "robot_camera")
        self.monitor_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "vitals_monitor")
        
        # 3. Load V-OCBF Safety Filter
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.b_net = VOCBFNet().to(self.device)
        model_path = PROJECT_ROOT / "models" / "vocbf_weights.pth"
        self.b_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.b_net.eval()
        
        self.steps = 0
        self.max_steps = 200

    def reset(self) -> ICUObservation:
        # Reset simulation and randomize starting position slightly
        mj.mj_resetData(self.model, self.data)
        self.data.ctrl[0] = np.random.uniform(-0.5, 0.5)
        self.data.ctrl[1] = np.random.uniform(-0.2, 0.2)
        mj.mj_forward(self.model, self.data)
        self.steps = 0
        
        return self._get_obs(reward=0.0, done=False)

    def step(self, action: ICUAction) -> ICUObservation:
        u_ref = np.array([action.pan_target, action.tilt_target])
        
        # 1. Get Current Visual State for Safety Filter
        q = np.array([self.data.qpos[0], self.data.qpos[1]], dtype=np.float32)
        self.renderer.update_scene(self.data, camera="robot_camera")
        img_array = self.renderer.render()
        
        img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)
        q_tensor = torch.tensor(q, dtype=torch.float32).unsqueeze(0).to(self.device)
        q_tensor.requires_grad = True

        # 2. V-OCBF Forward Pass & Lie Derivative
        B_val_tensor = self.b_net(img_tensor, q_tensor)
        grad_B_full = torch.autograd.grad(
            outputs=B_val_tensor,
            inputs=q_tensor,
            grad_outputs=torch.ones_like(B_val_tensor)
        )[0].cpu().detach().numpy()[0]

        B_val = B_val_tensor.item()
        
        # 3. OSQP Safety Filter (Threshold: 1.5 units from bed)
        B_shifted = B_val - 1.5 
        u_safe = cbf_qp_osqp(u_ref, grad_B_full, B_shifted)
        
        # 4. Apply Safe Action to MuJoCo
        self.data.ctrl[0] = u_safe[0]
        self.data.ctrl[1] = u_safe[1]
        
        for _ in range(10): # Step simulation 10 times per action to simulate real time dt
            mj.mj_step(self.model, self.data)
            
        self.steps += 1
        
        # 5. Calculate RL Reward (Tracking the Monitor)
        cam_pos = self.data.cam_xpos[self.cam_id]
        cam_mat = self.data.cam_xmat[self.cam_id].reshape(3, 3)
        cam_vec = -cam_mat[:, 2] 

        geom_id_arr = np.array([-1], dtype=np.intc)
        # dist_to_monitor = mj.mj_ray(self.model, self.data, cam_pos, cam_vec, None, 1, -1, geom_id_arr)
        # Get camera and monitor 3D coordinates
        cam_pos = self.data.cam_xpos[self.cam_id]
        monitor_pos = self.data.geom_xpos[self.monitor_geom_id]
        
        # Calculate Euclidean distance
        euclidean_dist = np.linalg.norm(monitor_pos - cam_pos)
        
        # Dense reward
        reward = 10.0 - euclidean_dist
        dist_to_monitor = euclidean_dist

        if geom_id_arr[0] == self.monitor_geom_id:
            # High reward for looking at the monitor, scaled by precision
            reward = 10.0 - dist_to_monitor 
        else:
            # Negative reward for drifting off target
            reward = -1.0
            dist_to_monitor = 10.0 # Max penalty distance

        done = self.steps >= self.max_steps

        return self._get_obs(reward=float(reward), done=done, dist_to_monitor=float(dist_to_monitor))

    def _get_obs(self, reward: float, done: bool, dist_to_monitor: float = 10.0) -> ICUObservation:
        return ICUObservation(
            current_pan=float(self.data.qpos[0]),
            current_tilt=float(self.data.qpos[1]),
            monitor_distance=dist_to_monitor,
            reward=reward,
            done=done
        )
        
    def state(self) -> ICUState:
        return ICUState(step_count=self.steps)