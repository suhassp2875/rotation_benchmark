import robosuite as suite
import json

task = "PickPlace"
env_params = {"object_type": "milk", "single_object_mode": 2}

env = suite.make(task, robots="Panda", **env_params)
obs = env.reset()
print(f"Task: {task}")
print(f"Keys: {sorted(obs.keys())}")
env.close()
