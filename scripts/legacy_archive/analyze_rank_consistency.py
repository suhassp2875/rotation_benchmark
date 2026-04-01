import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict

REPS = ["6d", "lie_fullfix", "atlas", "svd", "quat", "5d"]

def find_all_checkpoints(rep):
    outputs_dir = "outputs/2026-03-12"
    if not os.path.exists(outputs_dir):
        return []
    
    paths = []
    for d in os.listdir(outputs_dir):
        if d.endswith(f"_{rep}") or d.endswith(f"_{rep.upper()}"):
            path = os.path.join(outputs_dir, d, "best.pt")
            if os.path.exists(path):
                paths.append(path)
    return paths

def get_metrics(model_path, rep):
    # This would normally run the evaluate_model function
    # For now, let's assume we can run it and parse or it cache it.
    # To keep this script fast, we'll try to find evaluation results if they exist.
    eval_path = os.path.join(os.path.dirname(model_path), "eval_benchmark.json")
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            return json.load(f)
    return None

if __name__ == "__main__":
    # This script will aggregate results once training is done.
    # We will need to run the benchmark on all 30 checkpoints first.
    pass
