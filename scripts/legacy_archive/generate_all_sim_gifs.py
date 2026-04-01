"""
generate_all_sim_gifs.py
Runs a live robosuite simulation for EVERY model and saves a GIF for each.
Demonstrates the qualitative difference between low-tail and high-tail methods.
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

# Define models to run
OUTPUTS = Path("outputs")
CKPTS = Path("checkpoints")

MODELS = {
    "svd":      sorted(OUTPUTS.rglob("*SVD*/best.pt"))[-1],
    "sixd":     CKPTS / "sixd_checkpoint.pt",
    "atlas":    sorted(OUTPUTS.rglob("*atlas*/best.pt"))[-1],
    "quat":     sorted(OUTPUTS.rglob("*quat*/best.pt"))[-1],
    "lie_raw":  sorted(OUTPUTS.rglob("*lie_raw*/best.pt"))[-1],
    "fullfix":  CKPTS / "lie_fullfix_checkpoint.pt",
}

def run_sim_and_save_gif(name, path, env):
    print(f"\nProcessing {name}...")
    model, _ = load_from_checkpoint(path)
    if model is None:
        print(f"Skipping {name}, checkpoint not found.")
        return

    obs = env.reset()
    frames = []

    def capture_frame():
        img = obs["agentview_image"]
        img = np.flipud(img)
        frames.append(Image.fromarray(img))

    # Generate a random ground truth rotation for THIS trial
    # We use a fixed seed for the trial to make models comparable if possible,
    # but since they use their own predictions, the motion will differ.
    np_rng = np.random.RandomState(42) # Fixed seed per model run for comparable starting pose
    M = np_rng.randn(3, 3)
    U, _, Vt = np.linalg.svd(M)
    R_gt = U @ Vt
    if np.linalg.det(R_gt) < 0: U[:, -1] *= -1; R_gt = U @ Vt

    # Canonical points
    rng_p = np.random.RandomState(0)
    P = rng_p.normal(size=(32, 3))
    P /= np.linalg.norm(P, axis=-1, keepdims=True)
    
    # Predict
    X_obs = P @ R_gt.T + np_rng.randn(32, 3) * 0.01
    R_pred = predict_R(model, X_obs.flatten().astype(np.float32))
    
    R_err = R_pred @ R_gt.T
    err_quat = mat2quat(R_err)
    
    neutral_quat = obs["robot0_eef_quat"].copy()
    target_quat = quat_multiply(err_quat, neutral_quat)

    # Simulation phases
    # 1. Approach
    for _ in range(40):
        action = np.zeros(7)
        action[:3] = (obs["Milk_pos"] + np.array([0, 0, 0.15]) - obs["robot0_eef_pos"]) * 5.0
        action[3:6] = get_orientation_error(neutral_quat, obs["robot0_eef_quat"]) * 10.0
        action[6] = -1
        obs, _, _, _ = env.step(action)
        capture_frame()

    # 2. Lower and orient
    for _ in range(50):
        action = np.zeros(7)
        action[:3] = (obs["Milk_pos"] - obs["robot0_eef_pos"]) * 5.0
        action[3:6] = get_orientation_error(target_quat, obs["robot0_eef_quat"]) * 10.0
        action[6] = -1
        obs, _, _, _ = env.step(action)
        capture_frame()

    # 3. Grasp
    for _ in range(25):
        action = np.zeros(7); action[6] = 1
        obs, _, _, _ = env.step(action)
        capture_frame()

    # 4. Lift
    for _ in range(60):
        action = np.zeros(7); action[2] = 0.5; action[6] = 1
        action[3:6] = get_orientation_error(target_quat, obs["robot0_eef_quat"]) * 10.0
        obs, _, _, _ = env.step(action)
        capture_frame()

    # Save
    out_path = f"results/videos/sim_{name}.gif"
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=50, loop=0)
    print(f"Saved {out_path}")

def main():
    print("Generating All Simulation GIFs...")
    os.makedirs("results/videos", exist_ok=True)
    
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
        horizon=500,
        object_type="milk", 
        single_object_mode=2
    )

    for name, path in MODELS.items():
        try:
            run_sim_and_save_gif(name, path, env)
        except Exception as e:
            print(f"Error running {name}: {e}")

    env.close()
    print("\nAll simulations complete.")

if __name__ == "__main__":
    main()
