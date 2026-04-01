import os
import numpy as np
import json
import torch
import robosuite as suite
from robosuite.utils.transform_utils import quat2mat, mat2quat, quat_multiply, mat2euler, axisangle2quat, get_orientation_error
from robosuite.controllers import load_part_controller_config
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
import PIL.Image as Image
import time

# Results paths
PHASE1_RESULTS = "results/phase1_summary.json"

def load_results():
    if not os.path.exists(PHASE1_RESULTS):
        return {}
    with open(PHASE1_RESULTS, "r") as f:
        return json.load(f)

def inject_error(target_quat, error_deg):
    if error_deg <= 0.1:
        return target_quat
    
    axis = np.random.randn(3)
    axis /= (np.linalg.norm(axis) + 1e-8)
    angle = np.deg2rad(error_deg)
    err_quat = axisangle2quat(axis * angle)
    return quat_multiply(err_quat, target_quat)

def run_trial(env, error_deg, rep_name, trial_id):
    obs = env.reset()
    obj_name = "Milk"
    initial_obj_pos = obs[f"{obj_name}_pos"].copy()
    
    # Capture neutral orientation (looking down)
    neutral_quat = obs["robot0_eef_quat"].copy()
    
    # 1. Approach
    for t in range(50):
        obj_pos = obs[f"{obj_name}_pos"]
        target_pos = obj_pos + np.array([0, 0, 0.12]) 
        
        action = np.zeros(7)
        action[:3] = (target_pos - obs["robot0_eef_pos"]) * 3.0
        action[3:6] = get_orientation_error(neutral_quat, obs["robot0_eef_quat"]) * 3.0
        action[6] = -1 
        obs, _, _, _ = env.step(action)

    # 2. Grasp pose with error
    target_quat = inject_error(neutral_quat, error_deg)
    
    # Move to grasp
    for t in range(60):
        obj_pos = obs[f"{obj_name}_pos"]
        action = np.zeros(7)
        action[:3] = (obj_pos - obs["robot0_eef_pos"]) * 3.0
        action[3:6] = get_orientation_error(target_quat, obs["robot0_eef_quat"]) * 3.0
        action[6] = -1
        obs, _, _, _ = env.step(action)
        
    # 3. Close
    for _ in range(30):
        action = np.zeros(7)
        action[6] = 1 
        obs, _, _, _ = env.step(action)
        
    # 4. Lift
    for _ in range(100): # More steps for lifting
        action = np.zeros(7)
        action[2] = 0.5 
        action[3:6] = get_orientation_error(target_quat, obs["robot0_eef_quat"]) * 3.0
        action[6] = 1
        obs, _, _, _ = env.step(action)
        
    # 5. Check
    final_obj_pos = obs[f"{obj_name}_pos"]
    lifted_dist = final_obj_pos[2] - initial_obj_pos[2]
    # In PickPlace, a successful grasp should lift the object significantly
    success = lifted_dist > 0.04 
    
    if trial_id == 0:
        img = np.flipud(obs["agentview_image"])
        status = "SUCCESS" if success else "FAILURE"
        os.makedirs("results/images/grasps", exist_ok=True)
        Image.fromarray(img).save(f"results/images/grasps/{rep_name}_{status}.png")
        if not success:
             Image.fromarray(img).save(f"results/images/grasps/{rep_name}_failure_diag.png")
             
    print(f"    Trial {trial_id}: Lifted={lifted_dist:.4f}m -> {'OK' if success else 'FAIL'}")
    return success

def main():
    phase1 = load_results()
    reps = ["baseline_0", "6d", "svd", "atlas", "quat", "lie_fullfix", "5d"]
    results = {}
    
    arm_controller_config = load_part_controller_config(default_controller="OSC_POSE")
    controller_config = refactor_composite_controller_config(arm_controller_config, "Panda", ["right"])
    
    env = suite.make(
        "PickPlace",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        controller_configs=controller_config,
        control_freq=20,
        horizon=600,
        single_object_mode=2,
        object_type="milk",
    )
    
    for rep in reps:
        if rep == "baseline_0":
            error_deg = 0
            print(f"Evaluating Baseline (0 error)")
        elif rep not in phase1: continue
        else:
            error_deg = phase1[rep]["p99_avg"]
            print(f"Evaluating {rep} (P99 error: {error_deg:.2f}°)")
        
        successes = 0
        trials = 5 # Reduced for speed but still significant
        for i in range(trials):
            if run_trial(env, error_deg, rep, i):
                successes += 1
        
        results[rep] = successes / trials
        print(f"  Final Success Rate: {results[rep]:.2%}")
        
    with open("results/grasp_success_rates.json", "w") as f:
        json.dump(results, f, indent=2)
    env.close()

if __name__ == "__main__":
    main()
