import os
import sys
import json
import time
import torch
import numpy as np
import robosuite as suite
import imageio
from robosuite.utils.transform_utils import get_orientation_error, mat2quat, quat_multiply
from robosuite.controllers import load_part_controller_config
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from tqdm import tqdm
from pathlib import Path

# Setup paths to ensure we can load src and scripts
sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("scripts"))
from model_loader import load_from_checkpoint, predict_R
from rotation_benchmark.geometry.sampling import sample_stratified_rotations
from rotation_benchmark.geometry.metrics import geodesic_distance

RESULTS_PATH = Path("results/downstream_comprehensive_eval.json")
OUTPUTS = Path("outputs")
CKPTS = Path("checkpoints")

MODELS_TO_RUN = {
    "Oracle_GT": None, # Special baseline
    "SVD": sorted([str(p) for p in OUTPUTS.rglob("*svd*/best.pt")])[-1] if list(OUTPUTS.rglob("*svd*/best.pt")) else None,
    "6D": "checkpoints/sixd_checkpoint.pt",
    "Atlas": "checkpoints/atlas_checkpoint.pt",
    "Lie_FullFix": "checkpoints/lie_fullfix_checkpoint.pt",
    "Quat": sorted([str(p) for p in OUTPUTS.rglob("*quat*/best.pt")])[-1] if list(OUTPUTS.rglob("*quat*/best.pt")) else None,
    "Euler": sorted([str(p) for p in OUTPUTS.rglob("*euler*/best.pt")])[-1] if list(OUTPUTS.rglob("*euler*/best.pt")) else None,
}

# Filter out None paths
MODELS_TO_RUN = {k: v for k, v in MODELS_TO_RUN.items() if v is not None or k == "Oracle_GT"}

NUM_SEEDS = 20
TASKS = ["PickPlace", "Stack", "NutAssembly"]

def generate_target_R(seed):
    """
    Stratified setup: 30% Hard (Antipodal/Gimbal), 70% Uniform/Safe.
    We return R_gt and a string label for the type.
    """
    if seed % 10 < 2: 
        # 20% Antipodal
        R_gt = sample_stratified_rotations(1, region_type="antipodal", seed=seed + 1000).numpy()[0]
        return R_gt, "antipodal"
    elif seed % 10 == 2:
        # 10% Gimbal
        R_gt = sample_stratified_rotations(1, region_type="gimbal", seed=seed + 1000).numpy()[0]
        return R_gt, "gimbal"
    elif seed % 10 < 6:
        # 30% Safe
        R_gt = sample_stratified_rotations(1, region_type="safe", seed=seed + 1000).numpy()[0]
        return R_gt, "safe"
    else:
        # 40% Uniform
        R_gt = sample_stratified_rotations(1, region_type="uniform", seed=seed + 1000).numpy()[0]
        return R_gt, "uniform"

def render_pointcloud_obs(R_gt, num_points=32, noise_std=0.01):
    rng_p = np.random.RandomState(0)
    P = rng_p.normal(size=(num_points, 3))
    P /= np.linalg.norm(P, axis=-1, keepdims=True)
    X_clean = P @ R_gt.T
    X_obs = X_clean + np.random.randn(num_points, 3) * noise_std
    return X_obs.flatten().astype(np.float32)

def run_task_trial(env, task_name, model, R_gt, rep_name, seed):
    obs = env.reset()
    frames = []

    # Get representation error
    if rep_name == "Oracle_GT":
        R_pred = R_gt
        geo_err_deg = 0.0
    else:
        X_obs = render_pointcloud_obs(R_gt)
        R_pred = predict_R(model, X_obs)
        if R_pred is None: R_pred = np.eye(3)
        geo_err_deg = geodesic_distance(
            torch.tensor(R_pred).unsqueeze(0).float(),
            torch.tensor(R_gt).unsqueeze(0).float()
        ).item() * (180.0 / np.pi)

    R_err = R_pred @ R_gt.T
    err_quat = mat2quat(R_err)

    # Context specific parsing
    if task_name == "PickPlace":
        obj_name = "Milk"
        initial_z = obs[f"{obj_name}_pos"][2]
        neutral_quat = obs["robot0_eef_quat"].copy()
        target_quat = quat_multiply(err_quat, neutral_quat)

        # Approach
        for _ in range(50):
            action = np.zeros(7)
            action[:3] = (obs[f"{obj_name}_pos"] + np.array([0, 0, 0.12]) - obs["robot0_eef_pos"]) * 5.0
            action[3:6] = get_orientation_error(neutral_quat, obs["robot0_eef_quat"]) * 10.0
            action[6] = -1; obs, _, _, _ = env.step(action)
            if seed == 0: frames.append(np.flipud(obs["agentview_image"]))

        # Grasp
        for _ in range(80):
            action = np.zeros(7)
            action[:3] = (obs[f"{obj_name}_pos"] - obs["robot0_eef_pos"]) * 5.0
            action[3:6] = get_orientation_error(target_quat, obs["robot0_eef_quat"]) * 10.0
            action[6] = -1; obs, _, _, _ = env.step(action)
            if seed == 0: frames.append(np.flipud(obs["agentview_image"]))

        # Close
        for _ in range(30):
            action = np.zeros(7); action[6] = 1; obs, _, _, _ = env.step(action)
            if seed == 0: frames.append(np.flipud(obs["agentview_image"]))

        # Lift
        for _ in range(120):
            action = np.zeros(7); action[2] = 0.5; action[6] = 1
            action[3:6] = get_orientation_error(target_quat, obs["robot0_eef_quat"]) * 10.0
            obs, _, _, _ = env.step(action)
            if seed == 0: frames.append(np.flipud(obs["agentview_image"]))

        success = (obs[f"{obj_name}_pos"][2] - initial_z) > 0.04

    elif task_name == "Stack":
        obj_name = "cubeA"
        initial_z = obs[f"{obj_name}_pos"][2]
        neutral_quat = obs["robot0_eef_quat"].copy()
        target_quat = quat_multiply(err_quat, neutral_quat)

        for _ in range(40):
            action = np.zeros(7); action[:3] = (obs[f"{obj_name}_pos"] + np.array([0, 0, 0.1]) - obs["robot0_eef_pos"]) * 5.0
            action[3:6] = get_orientation_error(neutral_quat, obs["robot0_eef_quat"]) * 10.0; action[6] = -1; obs, _, _, _ = env.step(action)
        for _ in range(60):
            action = np.zeros(7); action[:3] = (obs[f"{obj_name}_pos"] - obs["robot0_eef_pos"]) * 5.0
            action[3:6] = get_orientation_error(target_quat, obs["robot0_eef_quat"]) * 10.0; action[6] = -1; obs, _, _, _ = env.step(action)
            if seed == 0: frames.append(np.flipud(obs["agentview_image"]))
        for _ in range(20):
            action = np.zeros(7); action[6] = 1; obs, _, _, _ = env.step(action)
            if seed == 0: frames.append(np.flipud(obs["agentview_image"]))
        for _ in range(60):
            action = np.zeros(7); action[2] = 0.5; action[6] = 1
            action[3:6] = get_orientation_error(target_quat, obs["robot0_eef_quat"]) * 10.0; obs, _, _, _ = env.step(action)
            if seed == 0: frames.append(np.flipud(obs["agentview_image"]))
            
        success = (obs[f"{obj_name}_pos"][2] - initial_z) > 0.04

    elif task_name == "NutAssembly":
        obj_name = "RoundNut"
        initial_z = obs[f"{obj_name}_pos"][2]
        neutral_quat = obs["robot0_eef_quat"].copy()
        target_quat = quat_multiply(err_quat, neutral_quat)

        # Approach
        for _ in range(50):
            action = np.zeros(7)
            action[:3] = (obs[f"{obj_name}_pos"] + np.array([0, 0, 0.1]) - obs["robot0_eef_pos"]) * 5.0
            action[3:6] = get_orientation_error(neutral_quat, obs["robot0_eef_quat"]) * 10.0
            action[6] = -1; obs, _, _, _ = env.step(action)
            if seed == 0: frames.append(np.flipud(obs["agentview_image"]))

        # Grasp
        for _ in range(80):
            action = np.zeros(7)
            action[:3] = (obs[f"{obj_name}_pos"] - obs["robot0_eef_pos"]) * 5.0
            action[3:6] = get_orientation_error(target_quat, obs["robot0_eef_quat"]) * 10.0
            action[6] = -1; obs, _, _, _ = env.step(action)
            if seed == 0: frames.append(np.flipud(obs["agentview_image"]))

        # Close
        for _ in range(30):
            action = np.zeros(7); action[6] = 1; obs, _, _, _ = env.step(action)
            if seed == 0: frames.append(np.flipud(obs["agentview_image"]))

        # Lift
        for _ in range(120):
            action = np.zeros(7); action[2] = 0.5; action[6] = 1
            action[3:6] = get_orientation_error(target_quat, obs["robot0_eef_quat"]) * 10.0
            obs, _, _, _ = env.step(action)
            if seed == 0: frames.append(np.flipud(obs["agentview_image"]))

        success = (obs[f"{obj_name}_pos"][2] - initial_z) > 0.04
    
    # Save video for seed 0
    if seed == 0 and len(frames) > 0:
        os.makedirs(f"results/videos/{task_name}", exist_ok=True)
        imageio.mimsave(f"results/videos/{task_name}/{rep_name}.gif", frames, fps=20)
        # Save end summary frame
        imageio.imwrite(f"results/videos/{task_name}/{rep_name}_end.png", frames[-1])

    return bool(success), geo_err_deg

def main():
    print(f"Starting Comprehensive Downstream Evaluation")
    
    arm_cfg = load_part_controller_config(default_controller="OSC_POSE")
    ctrl_cfg = refactor_composite_controller_config(arm_cfg, "Panda", ["right"])
    
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            results = json.load(f)
    else:
        results = {}

    for task in TASKS:
        print(f"\nEvaluating Task: {task}")
        env_params = {}
        if task == "PickPlace": env_params = {"object_type": "milk", "single_object_mode": 2}
        elif task == "NutAssembly": env_params = {"single_object_mode": 2, "nut_type": "round"}

        env_render = suite.make(
            task, robots="Panda", has_renderer=False, has_offscreen_renderer=True,
            use_camera_obs=True, camera_names=["agentview"], camera_heights=256, camera_widths=256,
            controller_configs=ctrl_cfg, control_freq=20, horizon=600, **env_params
        )
        env_fast = suite.make(
            task, robots="Panda", has_renderer=False, has_offscreen_renderer=False,
            use_camera_obs=False, camera_names=None,
            controller_configs=ctrl_cfg, control_freq=20, horizon=600, **env_params
        )

        if task not in results:
            results[task] = {}

        for name, path in MODELS_TO_RUN.items():
            # Skip if already fully evaluated (e.g. from previous aborted run)
            if name in results[task] and len(results[task][name].get("trials", [])) == NUM_SEEDS:
                print(f"  Skipping {name} (already evaluated)")
                continue

            print(f"  Running {name}...")
            
            if name == "Oracle_GT":
                model = None
            else:
                model, _ = load_from_checkpoint(path)
                if model is None: continue
            
            trials = []
            
            pbar = tqdm(range(NUM_SEEDS))
            for i in pbar:
                R_gt, region_type = generate_target_R(i)
                env = env_render if i == 0 else env_fast
                try:
                    success, geo_err = run_task_trial(env, task, model, R_gt, name, i)
                    trials.append({
                        "seed": i,
                        "region_type": region_type,
                        "success": success,
                        "geo_error": geo_err
                    })
                except Exception as e:
                    print(f"    ERROR in trial {i}: {e}")
                    # Minimal backoff recovery
                    env.close(); time.sleep(1)
                    if i == 0:
                        env_render = suite.make(task, robots="Panda", has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True, camera_names=["agentview"], camera_heights=256, camera_widths=256, controller_configs=ctrl_cfg, control_freq=20, horizon=600, **env_params)
                    else:
                        env_fast = suite.make(task, robots="Panda", has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False, camera_names=None, controller_configs=ctrl_cfg, control_freq=20, horizon=600, **env_params)
            
            success_rate = np.mean([t["success"] for t in trials])
            mean_err = np.mean([t["geo_error"] for t in trials])
            p99_err = np.percentile([t["geo_error"] for t in trials], 99) if trials else 0.0

            results[task][name] = {
                "success_rate": float(success_rate),
                "mean_geo_err": float(mean_err),
                "p99_geo_err": float(p99_err),
                "trials": trials
            }
            print(f"    Success: {success_rate:.1%} | Mean Err: {mean_err:.2f}° | p99: {p99_err:.2f}°")
            
            with open(RESULTS_PATH, "w") as f:
                json.dump(results, f, indent=2)

        env_render.close()
        env_fast.close()

if __name__ == "__main__":
    main()
