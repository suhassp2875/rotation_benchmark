"""
Controlled SO(3) Error Injection Sweep
=======================================
For each (task, error_level) condition, inject exactly theta degrees of
SO(3)-correct rotational error into the robot controller and measure task
success. Produces the cliff curves for the Hero Figure.
"""

import os
import sys
import json
import numpy as np
import robosuite as suite
from robosuite.controllers import load_part_controller_config
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from robosuite.utils.transform_utils import get_orientation_error, mat2quat, quat_multiply
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("scripts"))

# ── Sweep parameters ──────────────────────────────────────────────────────────
ERROR_LEVELS_DEG = [0, 2, 5, 7, 10, 15, 20, 30, 45, 60, 90]
DEFAULT_TRIALS = 30
CLIFF_TRIALS = 100  # Higher N for the transition regions
CLIFF_BINS = {
    "PickPlace": [7, 10, 15, 20, 30],
    "Stack": [20, 30, 45]
}
TASKS = ["PickPlace", "Stack"]
RESULTS_PATH = Path("results/error_cliff_sweep.json")

# ── SO(3)-correct perturbation (Rodrigues) ────────────────────────────────────
def perturb_rotation(R_gt: np.ndarray, theta_deg: float, rng: np.random.RandomState) -> np.ndarray:
    """Apply a rotation of exactly theta_deg around a uniformly random axis."""
    if theta_deg == 0.0:
        return R_gt.copy()
    axis = rng.randn(3)
    axis /= np.linalg.norm(axis)
    theta = np.radians(theta_deg)
    K = np.array([
        [0,        -axis[2],  axis[1]],
        [ axis[2],  0,       -axis[0]],
        [-axis[1],  axis[0],  0      ]
    ])
    R_perturb = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R_perturb @ R_gt

# ── Task-specific grasping protocol ──────────────────────────────────────────
def run_trial(env, task_name: str, R_perturbed: np.ndarray, seed: int) -> bool:
    obs = env.reset()

    # Compute quaternion error between perturbed and neutral end-effector pose
    R_err = R_perturbed @ np.eye(3).T  # error relative to identity
    err_quat = mat2quat(R_err)
    neutral_quat = obs["robot0_eef_quat"].copy()
    target_quat = quat_multiply(err_quat, neutral_quat)

    if task_name == "PickPlace":
        obj_name = "Milk"
        initial_z = obs[f"{obj_name}_pos"][2]

        for _ in range(50):
            action = np.zeros(7)
            action[:3] = (obs[f"{obj_name}_pos"] + np.array([0, 0, 0.12]) - obs["robot0_eef_pos"]) * 5.0
            action[3:6] = get_orientation_error(neutral_quat, obs["robot0_eef_quat"]) * 10.0
            action[6] = -1; obs, _, _, _ = env.step(action)

        for _ in range(80):
            action = np.zeros(7)
            action[:3] = (obs[f"{obj_name}_pos"] - obs["robot0_eef_pos"]) * 5.0
            action[3:6] = get_orientation_error(target_quat, obs["robot0_eef_quat"]) * 10.0
            action[6] = -1; obs, _, _, _ = env.step(action)

        for _ in range(30):
            action = np.zeros(7); action[6] = 1; obs, _, _, _ = env.step(action)

        for _ in range(120):
            action = np.zeros(7); action[2] = 0.5; action[6] = 1
            action[3:6] = get_orientation_error(target_quat, obs["robot0_eef_quat"]) * 10.0
            obs, _, _, _ = env.step(action)

        return bool((obs[f"{obj_name}_pos"][2] - initial_z) > 0.04)

    elif task_name == "Stack":
        obj_name = "cubeA"
        initial_z = obs[f"{obj_name}_pos"][2]

        for _ in range(40):
            action = np.zeros(7)
            action[:3] = (obs[f"{obj_name}_pos"] + np.array([0, 0, 0.1]) - obs["robot0_eef_pos"]) * 5.0
            action[3:6] = get_orientation_error(neutral_quat, obs["robot0_eef_quat"]) * 10.0
            action[6] = -1; obs, _, _, _ = env.step(action)

        for _ in range(60):
            action = np.zeros(7)
            action[:3] = (obs[f"{obj_name}_pos"] - obs["robot0_eef_pos"]) * 5.0
            action[3:6] = get_orientation_error(target_quat, obs["robot0_eef_quat"]) * 10.0
            action[6] = -1; obs, _, _, _ = env.step(action)

        for _ in range(20):
            action = np.zeros(7); action[6] = 1; obs, _, _, _ = env.step(action)

        for _ in range(60):
            action = np.zeros(7); action[2] = 0.5; action[6] = 1
            action[3:6] = get_orientation_error(target_quat, obs["robot0_eef_quat"]) * 10.0
            obs, _, _, _ = env.step(action)

        return bool((obs[f"{obj_name}_pos"][2] - initial_z) > 0.04)

    return False

# ── Main sweep ────────────────────────────────────────────────────────────────
def main():
    # Resume from existing results if available
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        print(f"Resuming from existing results ({RESULTS_PATH})")
    else:
        results = {}

    arm_cfg = load_part_controller_config(default_controller="OSC_POSE")
    ctrl_cfg = refactor_composite_controller_config(arm_cfg, "Panda", ["right"])

    for task in TASKS:
        if task not in results:
            results[task] = {}

        env_params = {}
        if task == "PickPlace":
            env_params = {"object_type": "milk", "single_object_mode": 2}

        env = suite.make(
            task, robots="Panda",
            has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
            controller_configs=ctrl_cfg, control_freq=20, horizon=600,
            **env_params
        )

        print(f"\n=== Task: {task} ===")
        for theta in ERROR_LEVELS_DEG:
            key = str(theta)
            target_n = CLIFF_TRIALS if theta in CLIFF_BINS.get(task, []) else DEFAULT_TRIALS
            
            if key in results[task] and len(results[task][key]) >= target_n:
                sr = np.mean(results[task][key][:target_n]) * 100
                print(f"  {theta:3.0f}°  [cached]  success={sr:.1f}% (N={len(results[task][key])})")
                continue

            current_successes = results[task].get(key, [])
            start_idx = len(current_successes)
            needed = target_n - start_idx
            
            print(f"  {theta:3.0f}°  [running {needed} more to hit N={target_n}]")
            
            for i in tqdm(range(start_idx, target_n), desc=f"  {theta:3.0f}°", leave=False):
                # Fresh random axis per trial
                trial_rng = np.random.RandomState(seed=i + int(theta * 100))
                R_gt = np.eye(3)   # Identity GT — we're testing error tolerance only
                R_perturbed = perturb_rotation(R_gt, theta, trial_rng)
                try:
                    success = run_trial(env, task, R_perturbed, seed=i)
                except Exception as e:
                    print(f"    ERROR trial {i}: {e}")
                    success = False
                current_successes.append(int(success))
                results[task][key] = current_successes

                # Save incrementally every 5 trials
                if i % 5 == 0:
                    with open(RESULTS_PATH, "w") as f:
                        json.dump(results, f, indent=2)

            sr = np.mean(results[task][key]) * 100
            print(f"  {theta:3.0f}°  success={sr:.1f}%  ({sum(results[task][key])}/{len(results[task][key])})")

            with open(RESULTS_PATH, "w") as f:
                json.dump(results, f, indent=2)

        env.close()

    print(f"\nSaved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()
