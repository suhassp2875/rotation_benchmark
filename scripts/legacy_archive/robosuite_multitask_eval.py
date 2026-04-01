import os
import numpy as np
import json
import torch
import robosuite as suite
from robosuite.utils.transform_utils import quat2mat, mat2quat, quat_multiply, mat2euler, axisangle2quat, get_orientation_error, convert_quat
from robosuite.controllers import load_part_controller_config
from robosuite.controllers.composite.composite_controller_factory import refactor_composite_controller_config
import PIL.Image as Image
import time
from tqdm import tqdm

# Results paths
PHASE1_RESULTS = "results/phase1_summary.json"
MULTITASK_RESULTS = "results/multitask_grasp_results.json"

def load_results(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        try:
            return json.load(f)
        except:
            return {}

def inject_error(target_quat, error_deg):
    if error_deg <= 0.1:
        return target_quat
    
    axis = np.random.randn(3)
    axis /= (np.linalg.norm(axis) + 1e-8)
    angle = np.deg2rad(error_deg)
    err_quat = axisangle2quat(axis * angle)
    return quat_multiply(err_quat, target_quat)

def run_trial(env, task_name, error_deg, rep_name, trial_id):
    obs = env.reset()
    
    # Identify target object and success metric based on task
    if task_name == "PickPlace":
        obj_name = "Milk"
        initial_pos = obs[f"{obj_name}_pos"].copy()
    elif task_name == "Stack":
        obj_name = "cubeA"
        initial_pos = obs[f"{obj_name}_pos"].copy()
    elif task_name == "NutAssembly":
        obj_name = "RoundNut"
        initial_pos = obs[f"{obj_name}_pos"].copy()
    else:
        raise ValueError(f"Unknown task: {task_name}")

    neutral_quat = obs["robot0_eef_quat"].copy()
    
    # 1. Approach
    for t in range(50):
        target_pos = obs[f"{obj_name}_pos"] + np.array([0, 0, 0.12]) 
        action = np.zeros(7)
        action[:3] = (target_pos - obs["robot0_eef_pos"]) * 5.0
        action[3:6] = get_orientation_error(neutral_quat, obs["robot0_eef_quat"]) * 5.0
        action[6] = -1 
        obs, _, _, _ = env.step(action)

    # 2. Grasp pose with error
    target_quat = inject_error(neutral_quat, error_deg)
    
    # Move to grasp
    for t in range(80):
        obj_pos = obs[f"{obj_name}_pos"]
        action = np.zeros(7)
        action[:3] = (obj_pos - obs["robot0_eef_pos"]) * 5.0
        action[3:6] = get_orientation_error(target_quat, obs["robot0_eef_quat"]) * 5.0
        action[6] = -1
        obs, _, _, _ = env.step(action)
        
    # 3. Close
    for _ in range(30):
        action = np.zeros(7)
        action[6] = 1 
        obs, _, _, _ = env.step(action)
        
    # 4. Lift
    for _ in range(120):
        action = np.zeros(7)
        action[2] = 0.6 
        action[3:6] = get_orientation_error(target_quat, obs["robot0_eef_quat"]) * 5.0
        action[6] = 1
        obs, _, _, _ = env.step(action)
        
    # 5. Check Success
    current_pos = obs[f"{obj_name}_pos"]
    lifted_dist = current_pos[2] - initial_pos[2]
    success = lifted_dist > 0.04 
    
    # Capture multiple views for first trial
    if trial_id == 0:
        os.makedirs(f"results/images/multitask/{task_name}", exist_ok=True)
        status = "SUCCESS" if success else "FAILURE"
        
        # View 1: Agentview
        img1 = np.flipud(obs["agentview_image"])
        Image.fromarray(img1).save(f"results/images/multitask/{task_name}/{rep_name}_{status}_agent.png")
        
        # View 2: Sideview (Frontview in robosuite usually)
        if "frontview_image" in obs:
            img2 = np.flipud(obs["frontview_image"])
            Image.fromarray(img2).save(f"results/images/multitask/{task_name}/{rep_name}_{status}_front.png")
             
    return success

def main():
    phase1 = load_results(PHASE1_RESULTS)
    final_results = load_results(MULTITASK_RESULTS)
    
    reps = ["baseline_0", "6d", "svd", "atlas", "quat", "lie_fullfix"]
    tasks = ["PickPlace", "Stack", "NutAssembly"]
    num_seeds = 10
    
    arm_controller_config = load_part_controller_config(default_controller="OSC_POSE")
    controller_config = refactor_composite_controller_config(arm_controller_config, "Panda", ["right"])
    
    for task in tasks:
        if task not in final_results:
            final_results[task] = {}
            
        print(f"\nEvaluating Task: {task}")
        
        # Initialize Task-specific env
        if task == "PickPlace":
            env_params = {"object_type": "milk", "single_object_mode": 2}
        elif task == "Stack":
            env_params = {}
        elif task == "NutAssembly":
            env_params = {"single_object_mode": 2, "nut_type": "round"}

        # Check if all reps are done for this task
        all_done = True
        for rep in reps:
            if rep not in final_results[task]:
                all_done = False
                break
        if all_done:
            print(f"Skipping task {task} (all representations complete)")
            continue

        env = suite.make(
            task,
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=["agentview", "frontview"],
            camera_heights=512,
            camera_widths=512,
            controller_configs=controller_config,
            control_freq=20,
            horizon=600,
            **env_params
        )

        for rep in reps:
            if rep in final_results[task]:
                print(f"  Rep: {rep} (Skipping - already exists)")
                continue

            if rep == "baseline_0":
                error_deg = 0.0
            elif rep not in phase1:
                continue
            else:
                error_deg = phase1[rep]["p99_avg"]
            
            print(f"  Rep: {rep} (P99 Err: {error_deg:.2f}°)")
            successes = []
            for i in tqdm(range(num_seeds), desc=f"    {rep}"):
                try:
                    res = run_trial(env, task, error_deg, rep, i)
                    successes.append(float(res))
                except Exception as e:
                    print(f"    ERROR in trial {i}: {e}")
                    # Try to recreate env if memory error
                    if "memory" in str(e).lower():
                        env.close()
                        time.sleep(5)
                        env = suite.make(task, robots="Panda", has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True, camera_names=["agentview", "frontview"], camera_heights=512, camera_widths=512, controller_configs=controller_config, control_freq=20, horizon=600, **env_params)
                        res = run_trial(env, task, error_deg, rep, i)
                        successes.append(float(res))
            
            final_results[task][rep] = {
                "mean": np.mean(successes),
                "std": np.std(successes),
                "raw": successes
            }
            print(f"    Success Rate: {final_results[task][rep]['mean']:.1%}")
            
            # Save intermediate
            with open(MULTITASK_RESULTS, "w") as f:
                json.dump(final_results, f, indent=2)
            
        env.close()

    print("\nFinal multitasking results saved to results/multitask_grasp_results.json")

if __name__ == "__main__":
    main()
