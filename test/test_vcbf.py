import os
import mujoco as mj
import numpy as np
import torch
import glfw
import sys
from pathlib import Path

# Add the project root to the path for clean imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

print("path: ", PROJECT_ROOT)
print("")
# from models.vocbf_net import VOCBFNet
from scripts.train_vocbf import VOCBFNet
from src.qp import cbf_qp_osqp

# ================= SIM CONFIG =================
xml_path = PROJECT_ROOT / "assets" / "icu" / "hackathon_icu.xml"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= LOAD BARRIER MODEL =================
B_net = VOCBFNet().to(DEVICE)
model_path = PROJECT_ROOT / "models" / "vocbf_weights.pth"
B_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
B_net.eval()

# ================= CONTROLLER =================
def controller(model, data):
    # Make the camera pan back and forth over time using a sine wave
    sweep_pan = 0.8 * np.sin(data.time)
    u_ref = np.array([sweep_pan, 0.0]) 
    
    q = np.array([data.qpos[0], data.qpos[1]], dtype=np.float32)
    
    renderer.update_scene(data, camera="robot_camera")
    img_array = renderer.render()
    img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(DEVICE)
    
    q_tensor = torch.tensor(q, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    q_tensor.requires_grad = True

    B_val_tensor = B_net(img_tensor, q_tensor)
    
    grad_B_full = torch.autograd.grad(
        outputs=B_val_tensor,
        inputs=q_tensor,
        grad_outputs=torch.ones_like(B_val_tensor)
    )[0].cpu().detach().numpy()[0]

    B_val = B_val_tensor.item()
    
    # Safety boundary
    B_shifted = B_val - 0.5
    
    u_safe = cbf_qp_osqp(u_ref, grad_B_full, B_shifted)
    
    print(f"Dist: {B_val:.2f} | u_ref: {u_ref} | u_safe: [{u_safe[0]:.2f}, {u_safe[1]:.2f}]")

    data.ctrl[0] = u_safe[0]
    data.ctrl[1] = u_safe[1]

# ================= MUJOCO & GLFW INIT =================
model = mj.MjModel.from_xml_path(str(xml_path))
data = mj.MjData(model)
renderer = mj.Renderer(model, height=64, width=64)

if not glfw.init():
    raise Exception("Could not initialize GLFW")
window = glfw.create_window(1200, 900, "V-OCBF Test", None, None)
glfw.make_context_current(window)

cam = mj.MjvCamera()
opt = mj.MjvOption()
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, int(mj.mjtFontScale.mjFONTSCALE_150.value))

mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)

cam.azimuth = 120
cam.elevation = -20
cam.distance = 5.0
cam.lookat = np.array([0.0, 0.0, 1.0])

mj.set_mjcb_control(controller)

# ================= SIM LOOP =================
while not glfw.window_should_close(window):
    time_prev = data.time
    while data.time - time_prev < 1.0/60.0:
        mj.mj_step(model, data)

    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    mj.mjv_updateScene(model, data, opt, None, cam, int(mj.mjtCatBit.mjCAT_ALL.value), scene)
    mj.mjr_render(viewport, scene, context)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()