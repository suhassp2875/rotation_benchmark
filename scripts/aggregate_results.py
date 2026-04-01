import os
import subprocess
import json
import numpy as np
import re
from collections import defaultdict

REPS = ["6d", "lie_fullfix", "atlas", "svd", "quat", "5d"]
OUTPUTS_DIR = "outputs/2026-03-12"

def find_checkpoints():
    checkpoints = defaultdict(list)
    for d in os.listdir(OUTPUTS_DIR):
        # Match pattern like HH-MM-SS_REP (case insensitive)
        for rep in REPS:
            if d.lower().endswith(f"_{rep.lower()}"):
                path = os.path.join(OUTPUTS_DIR, d, "best.pt")
                if os.path.exists(path):
                    checkpoints[rep].append(path)
    return checkpoints

def run_benchmark(model_path, rep):
    print(f"Benchmarking {rep} @ {model_path}...")
    cmd = [
        "python", "-m", "rotation_benchmark.eval.benchmark",
        "--model_path", model_path,
        "--rep", rep,
        "--samples", "10000" # Use slightly fewer samples for faster aggregation
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    # Parse output
    metrics = {}
    for line in result.stdout.split('\n'):
        # Median Error:   0.024° [0.024, 0.025]
        m = re.search(r"Median Error:\s+(\d+\.\d+)°", line)
        if m: metrics['median'] = float(m.group(1))
        m = re.search(r"p99 Error:\s+(\d+\.\d+)°", line)
        if m: metrics['p99'] = float(m.group(1))
        m = re.search(r"Fail > 30°:\s+(\d+\.\d+)%", line)
        if m: metrics['fail'] = float(m.group(1))
    return metrics

if __name__ == "__main__":
    ckpts = find_checkpoints()
    summary = {}
    
    for rep, paths in ckpts.items():
        print(f"--- Summarizing {rep} ({len(paths)} seeds) ---")
        rep_metrics = []
        for p in paths[:5]: # Cap at 5 seeds if duplicates
            m = run_benchmark(p, rep)
            if m:
                rep_metrics.append(m)
        
        if rep_metrics:
            # Seed-averaged metrics
            summary[rep] = {
                "median_avg": np.mean([m['median'] for m in rep_metrics]),
                "median_std": np.std([m['median'] for m in rep_metrics]),
                "p99_avg": np.mean([m['p99'] for m in rep_metrics]),
                "fail_avg": np.mean([m['fail'] for m in rep_metrics]),
            }
            
    with open("results/phase1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\nPhase 1 Results Table:")
    print(f"{'Rep':<15} | {'Median':<10} | {'p99':<10} | {'Fail%':<10}")
    print("-" * 50)
    for rep, s in summary.items():
        print(f"{rep:<15} | {s['median_avg']:>6.3f}°    | {s['p99_avg']:>6.2f}°    | {s['fail_avg']:>6.2f}%")
