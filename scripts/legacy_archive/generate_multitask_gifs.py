"""
generate_multitask_gifs.py
Generates simulation GIFs for all 6 models across 3 tasks: PickPlace, Stack, and NutAssembly.
"""
import os, sys, torch, json
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

# Define models
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

TASKS = {
    "PickPlace": {"params": {"object_type": "milk", "single_object_mode": 2}, "obj": "Milk"},
    "Stack":     {"params": {}, "obj": "cubeA"},
    "NutAssembly": {"params": {"single_object_mode": 2, "nut_type": "round"}, "obj": "RoundNut"}
}

def run_task_model_sim(task_name, model_name, model_path, env, obj_key):
    print(f"  Model: {model_name}...")
    model, _ = load_from_checkpoint(model_path)
    if model is None: return

    obs = env.reset()
    frames = []

    def capture_frame():
        img = obs["agentview_image"]
        img = np.flipud(img)
        frames.append(Image.fromarray(img))

    # Fixed seed for comparable trials within a task
    np_rng = np.random.RandomState(42)
    M = np_rng.randn(3, 3)
    U, _, Vt = np.linalg.svd(M)
    R_gt = U @ Vt
    if np.linalg.det(R_gt) < 0: U[:, -1] *= -1; R_gt = U @ Vt

    rng_p = np.random.RandomState(0)
    P = rng_p.normal(size=(32, 3))
    P /= np.linalg.norm(P, axis=-1, keepdims=True)
    
    X_obs = P @ R_gt.T + np_rng.randn(32, 3) * 0.01
    R_pred = predict_R(model, X_obs.flatten().astype(np.float32))
    
    R_err = R_pred @ R_gt.T
    err_quat = mat2quat(R_err)
    
    neutral_quat = obs["robot0_eef_quat"].copy()
    target_quat = quat_multiply(err_quat, neutral_quat)

    # Simulation Phases
    # 1. Approach
    for _ in range(40):
        action = np.zeros(7)
        action[:3] = (obs[f"{obj_key}_pos"] + np.array([0, 0, 0.15]) - obs["robot0_eef_pos"]) * 5.0
        action[3:6] = get_orientation_error(neutral_quat, obs["robot0_eef_quat"]) * 10.0
        action[6] = -1; obs, _, _, _ = env.step(action); capture_frame()

    # 2. Lower
    for _ in range(50):
        action = np.zeros(7)
        action[:3] = (obs[f"{obj_key}_pos"] - obs["robot0_eef_pos"]) * 5.0
        action[3:6] = get_orientation_error(target_quat, obs["robot0_eef_quat"]) * 10.0
        action[6] = -1; obs, _, _, _ = env.step(action); capture_frame()

    # 3. Grasp
    for _ in range(25):
        action = np.zeros(7); action[6] = 1; obs, _, _, _ = env.step(action); capture_frame()

    # 4. Lift
    for _ in range(60):
        action = np.zeros(7); action[2] = 0.5; action[6] = 1
        action[3:6] = get_orientation_error(target_quat, obs["robot0_eef_quat"]) * 10.0
        obs, _, _, _ = env.step(action); capture_frame()

    # Save GIF
    out_dir = Path("results/videos/multitask") / task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}.gif"
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=50, loop=0)

def main():
    print("Generating Multi-Task Comparative GIFs...")
    arm_cfg = load_part_controller_config(default_controller="OSC_POSE")
    ctrl_cfg = refactor_composite_controller_config(arm_cfg, "Panda", ["right"])
    
    for task_name, info in TASKS.items():
        print(f"\nTask: {task_name}")
        env = suite.make(
            task_name, robots="Panda", has_renderer=False, has_offscreen_renderer=True,
            use_camera_obs=True, controller_configs=ctrl_cfg, control_freq=20, horizon=500,
            **info["params"]
        )
        
        for model_name, model_path in MODELS.items():
            try:
                run_task_model_sim(task_name, model_name, model_path, env, info["obj"])
            except Exception as e:
                print(f"    Error in {model_name}: {e}")
        
        env.close()
    
    print("\nAll multitask GIFs generated in results/videos/multitask/")

if __name__ == "__main__":
    main()
