import mujoco as mj
import numpy as np
import torch
import glfw
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from envs.icu_env import HackathonICUEnv, ICUAction
from scripts.train_rl import ActorCritic

def test_rl():
    env = HackathonICUEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Trained Agent
    agent = ActorCritic().to(device)
    agent.load_state_dict(torch.load(PROJECT_ROOT / "models" / "native_rl_policy.pth", map_location=device))
    agent.eval()

    # Setup GLFW for visualization
    if not glfw.init():
        raise Exception("Could not initialize GLFW")
    window = glfw.create_window(1200, 900, "RL + V-OCBF Test", None, None)
    glfw.make_context_current(window)

    cam = mj.MjvCamera()
    opt = mj.MjvOption()
    scene = mj.MjvScene(env.model, maxgeom=10000)
    context = mj.MjrContext(env.model, int(mj.mjtFontScale.mjFONTSCALE_150.value))
    mj.mjv_defaultCamera(cam)
    mj.mjv_defaultOption(opt)

    cam.azimuth, cam.elevation, cam.distance = 120, -20, 5.0
    cam.lookat = np.array([0.0, 0.0, 1.0])

    obs = env.reset()
    
    print("Testing RL Agent. Press ESC to stop.")
    while not glfw.window_should_close(window):
        # 1. RL Agent Predicts Action
        state_tensor = torch.tensor(
            [obs.current_pan, obs.current_tilt, obs.monitor_distance], 
            dtype=torch.float32
        ).unsqueeze(0).to(device)
        
        with torch.no_grad():
            mean, _, _ = agent(state_tensor)
            action_val = mean[0].cpu().numpy()
            
        action = ICUAction(pan_target=float(action_val[0]), tilt_target=float(action_val[1]))
        
        # 2. Step Environment (Includes V-OCBF Safety Filter)
        obs = env.step(action)
        
        # 3. Render
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
        mj.mjv_updateScene(env.model, env.data, opt, None, cam, int(mj.mjtCatBit.mjCAT_ALL.value), scene)
        mj.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()
        
        if obs.done:
            obs = env.reset()

    glfw.terminate()

if __name__ == "__main__":
    test_rl()