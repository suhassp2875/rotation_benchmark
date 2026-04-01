import subprocess
import os
import numpy as np
import json

REPS = ["6d", "lie_fullfix", "atlas", "svd", "quat", "5d"]
SWEEP = [0.0, 0.1, 0.3, 0.5, 0.7]
SAMPLES = 10000

import subprocess
import os
import numpy as np
import json
import re

REPS = ["6d", "lie_fullfix", "atlas", "svd", "quat", "5d"]
SWEEP = [0.0, 0.1, 0.3, 0.5, 0.7]

def find_best_checkpoint(rep):
    """Find the latest best.pt for a given representation."""
    outputs_dir = "outputs/2026-03-12"
    if not os.path.exists(outputs_dir):
        return None
    
    candidates = []
    for d in os.listdir(outputs_dir):
        if d.lower().endswith(f"_{rep.lower()}"):
            path = os.path.join(outputs_dir, d, "best.pt")
            if os.path.exists(path):
                # Use the one with the latest modification time or most recent seed
                candidates.append((os.path.getmtime(path), path))
    
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]

def run_occlusion(rep, occlusion, model_path):
    print(f"--- Running Occlusion {occlusion} for {rep} ---")
    cmd = [
        "python", "-m", "rotation_benchmark.downstream.occluded_pose_regression",
        "--model_path", model_path,
        "--rep", rep,
        "--occlusion_rate", str(occlusion)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse output
    # Median Error: 1.23°
    # p99 Error:    4.56°
    # Fail > 30°:   0.00%
    metrics = {}
    for line in result.stdout.split('\n'):
        if "Median Error:" in line:
            metrics['median'] = float(re.search(r"(\d+\.\d+)", line).group(1))
        if "p99 Error:" in line:
            metrics['p99'] = float(re.search(r"(\d+\.\d+)", line).group(1))
        if "Fail > 30°:" in line:
            metrics['fail'] = float(re.search(r"(\d+\.\d+)", line).group(1))
    return metrics

if __name__ == "__main__":
    results = {}
    for rep in REPS:
        model_path = find_best_checkpoint(rep)
        if not model_path:
            print(f"Skipping {rep}: No checkpoint found.")
            continue
            
        print(f"Using checkpoint: {model_path}")
        results[rep] = {}
        for occ in SWEEP:
            res = run_occlusion(rep, occ, model_path)
            results[rep][occ] = res
            
    with open("results/occlusion_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Done! Results saved to results/occlusion_sweep.json")
