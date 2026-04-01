"""
demonstrate_lift.py
Runs a live simulation of a model (SVD) performing a PickPlace task.
Saves a GIF of the result to results/videos/lift_demo.gif.
"""
import os, sys, torch
import numpy as np
import robosuite as suite
from robosuite.utils.transform_utils import get_orientation_error, mat2quat, quat_multiply
from robosuite.controllers import load_part_controller_config
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
import PIL.Image as Image
from pathlib import Path
from tqdm import tqdm

# Setup paths
sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from model_loader import load_from_checkpoint, predict_R

def main():
    print("Initializing Robosuite Demonstration...")
    os.makedirs("results/videos", exist_ok=True)
    
    # 1. Selection and Loading
    output_dir = Path("outputs")
    svd_ckpt = sorted(output_dir.rglob("*SVD*/best.pt"))[-1]
    print(f"Loading model: {svd_ckpt.name}")
    model, _ = load_from_checkpoint(svd_ckpt)
    
    # 2. Setup Env
    arm_cfg = load_part_controller_config(default_controller="OSC_POSE")
    ctrl_cfg = refactor_composite_controller_config(arm_cfg, "Panda", ["right"])
    
    env = suite.make(
        "PickPlace", 
        robots="Panda", 
        has_renderer=False, 
        has_offscreen_renderer=True,
        use_camera_obs=True, 
        controller_configs=ctrl_cfg, 
        control_freq=20, 
        horizon=400,
        object_type="milk", 
        single_object_mode=2
    )
    
    obs = env.reset()
    frames = []

    def capture_frame():
        img = obs["agentview_image"]
        img = np.flipud(img)
        frames.append(Image.fromarray(img))

    # 3. Predict Rotation
    # Generate random ground truth for the demo
    np_rng = np.random.RandomState()
    M = np_rng.randn(3, 3)
    U, _, Vt = np.linalg.svd(M)
    R_gt = U @ Vt
    if np.linalg.det(R_gt) < 0: U[:, -1] *= -1; R_gt = U @ Vt

    # P_seed=0 matching training
    rng_p = np.random.RandomState(0)
    P = rng_p.normal(size=(32, 3))
    P /= np.linalg.norm(P, axis=-1, keepdims=True)
    
    X_obs = P @ R_gt.T + np_rng.randn(32, 3) * 0.01
    R_pred = predict_R(model, X_obs.flatten().astype(np.float32))
    
    R_err = R_pred @ R_gt.T
    err_quat = mat2quat(R_err)
    
    neutral_quat = obs["robot0_eef_quat"].copy()
    target_quat = quat_multiply(err_quat, neutral_quat)
    
    print("Running simulation phases...")

    # Phase 1: Approach
    for _ in range(30):
        action = np.zeros(7)
        action[:3] = (obs["Milk_pos"] + np.array([0, 0, 0.15]) - obs["robot0_eef_pos"]) * 5.0
        action[3:6] = get_orientation_error(neutral_quat, obs["robot0_eef_quat"]) * 10.0
        action[6] = -1
        obs, _, _, _ = env.step(action)
        capture_frame()

    # Phase 2: Lower and Grasp
    for _ in range(40):
        action = np.zeros(7)
        action[:3] = (obs["Milk_pos"] - obs["robot0_eef_pos"]) * 5.0
        action[3:6] = get_orientation_error(target_quat, obs["robot0_eef_quat"]) * 10.0
        action[6] = -1
        obs, _, _, _ = env.step(action)
        capture_frame()

    # Phase 3: Close Gripper
    for _ in range(20):
        action = np.zeros(7); action[6] = 1
        obs, _, _, _ = env.step(action)
        capture_frame()

    # Phase 4: Lift
    for _ in range(60):
        action = np.zeros(7); action[2] = 0.5; action[6] = 1
        action[3:6] = get_orientation_error(target_quat, obs["robot0_eef_quat"]) * 10.0
        obs, _, _, _ = env.step(action)
        capture_frame()

    # Save GIF
    gif_path = "results/videos/lift_demo.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=50, loop=0)
    print(f"Simulation complete! Animation saved to {gif_path}")
    env.close()

if __name__ == "__main__":
    main()
