import robosuite as suite
import json

tasks = ["PickPlace", "Stack", "NutAssembly"]
for task in tasks:
    print(f"\nTask: {task}")
    if task == "PickPlace":
        env_params = {"object_type": "milk", "single_object_mode": 2}
    elif task == "Stack":
        env_params = {}
    elif task == "NutAssembly":
        env_params = {"single_object_mode": 2, "nut_type": "round"}
    
    env = suite.make(task, robots="Panda", **env_params)
    obs = env.reset()
    print(f"Keys: {sorted(obs.keys())}")
    env.close()
