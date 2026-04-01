"""
robosuite_live_eval.py  —  Run the ACTUAL model in the robotics loop.
Instead of injecting fixed errors, we run model.predict(X) at each grasp step.
"""
import os, sys, json, time, torch
import numpy as np
import robosuite as suite
from robosuite.utils.transform_utils import get_orientation_error, mat2quat, quat_multiply

from robosuite.controllers import load_part_controller_config
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
from tqdm import tqdm
from pathlib import Path

# Setup paths
sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from model_loader import load_from_checkpoint, predict_R

RESULTS_PATH = Path("results/live_eval_results.json")
OUTPUTS      = Path("outputs")
CKPTS        = Path("checkpoints")

MODELS_TO_RUN = {
    "SVD (ours)":  sorted(OUTPUTS.rglob("*SVD*/best.pt"))[-1],
    "6D":          CKPTS / "sixd_checkpoint.pt",
    "Lie_Raw":     sorted(OUTPUTS.rglob("*lie_raw*/best.pt"))[-1],
    "FullFix":     CKPTS / "lie_fullfix_checkpoint.pt",
}

NUM_SEEDS = 10
TASK_NAME = "PickPlace" # Focus on the most representative task

def run_live_trial(env, model, num_points=32, noise_std=0.01):
    obs = env.reset()
    # 1. Generate a ground-truth rotation R_gt and a model prediction
    # We use a random R_gt to see how the model performs on a random pose
    np_rng = np.random.RandomState()
    # Manual random rotation to avoid dependencies
    M = np_rng.randn(3, 3)
    U, _, Vt = np.linalg.svd(M)
    R_gt = U @ Vt
    if np.linalg.det(R_gt) < 0: U[:, -1] *= -1; R_gt = U @ Vt

    # 2. Get canonical points (P_seed=0 matching training)
    rng_p = np.random.RandomState(0)
    P = rng_p.normal(size=(num_points, 3))
    P /= np.linalg.norm(P, axis=-1, keepdims=True)
    
    # 3. Predict R_pred from model
    X_clean = P @ R_gt.T
    X_obs = X_clean + np_rng.randn(num_points, 3) * noise_std
    R_pred = predict_R(model, X_obs.flatten().astype(np.float32))
    if R_pred is None: R_pred = np.eye(3)
    
    # 4. Calculate the ACTUAL orientation target for the robot
    # In 'error injection' terms, we want the robot to try to grasp at R_pred
    # but the 'correct' grasp is at neutral_quat. 
    # So if R_pred == R_gt, error is 0.
    # The relative error quat is R_pred @ R_gt.T
    R_err = R_pred @ R_gt.T
    err_quat = mat2quat(R_err)
    
    # 5. Robot control loop
    neutral_quat = obs["robot0_eef_quat"].copy()
    target_quat = quat_multiply(err_quat, neutral_quat)
    initial_z = obs["Milk_pos"][2]
    
    # Approach
    for _ in range(50):
        action = np.zeros(7)
        action[:3] = (obs["Milk_pos"] + np.array([0, 0, 0.12]) - obs["robot0_eef_pos"]) * 5.0
        action[3:6] = get_orientation_error(neutral_quat, obs["robot0_eef_quat"]) * 10.0
        action[6] = -1; obs, _, _, _ = env.step(action)
    
    # Grasp
    for _ in range(80):
        action = np.zeros(7)
        action[:3] = (obs["Milk_pos"] - obs["robot0_eef_pos"]) * 5.0
        action[3:6] = get_orientation_error(target_quat, obs["robot0_eef_quat"]) * 10.0
        action[6] = -1; obs, _, _, _ = env.step(action)
    
    # Close
    for _ in range(30):
        action = np.zeros(7); action[6] = 1; obs, _, _, _ = env.step(action)
    
    # Lift
    for _ in range(120):
        action = np.zeros(7); action[2] = 0.5; action[6] = 1
        action[3:6] = get_orientation_error(target_quat, obs["robot0_eef_quat"]) * 10.0
        obs, _, _, _ = env.step(action)
        
    return (obs["Milk_pos"][2] - initial_z) > 0.04


def main():
    print(f"Starting LIVE Model-In-The-Loop Evaluation ({TASK_NAME}, N={NUM_SEEDS})")
    
    arm_cfg = load_part_controller_config(default_controller="OSC_POSE")
    ctrl_cfg = refactor_composite_controller_config(arm_cfg, "Panda", ["right"])
    
    env = suite.make(
        TASK_NAME, robots="Panda", has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, controller_configs=ctrl_cfg, control_freq=20, horizon=500,
        object_type="milk", single_object_mode=2
    )
    
    results = {}
    for name, path in MODELS_TO_RUN.items():
        print(f"\nRunning {name}...")
        model, _ = load_from_checkpoint(path)
        if model is None: continue
        
        successes = []
        for i in tqdm(range(NUM_SEEDS)):
            success = run_live_trial(env, model)
            successes.append(float(success))
            
        results[name] = {
            "success_rate": np.mean(successes),
            "raw": successes
        }
        print(f"  Result: {results[name]['success_rate']:.1%}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved live results to {RESULTS_PATH}")
    env.close()

if __name__ == "__main__":
    main()
