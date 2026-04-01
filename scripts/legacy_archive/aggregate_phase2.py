import os
import subprocess
import json
import numpy as np
import re
from collections import defaultdict

ABLATION_REPS = ["lie_raw", "lie_hardened", "Lie_FullFix"] 

def run_benchmark(model_path, rep):
    cmd = [
        "python", "-m", "rotation_benchmark.eval.benchmark",
        "--model_path", model_path,
        "--rep", rep,
        "--samples", "10000",
        "--no_bootstrap"
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    metrics = {}
    for line in result.stdout.split('\n'):
        m = re.search(r"Median Error:\s+(\d+\.\d+)°", line)
        if m: metrics['median'] = float(m.group(1))
        m = re.search(r"p99 Error:\s+(\d+\.\d+)°", line)
        if m: metrics['p99'] = float(m.group(1))
        m = re.search(r"Fail > 30°:\s+(\d+\.\d+)%", line)
        if m: metrics['fail'] = float(m.group(1))
    return metrics

def find_latest_ckpt(rep, outputs_dir="outputs/2026-03-12"):
    candidates = []
    for d in os.listdir(outputs_dir):
        if d.lower().endswith(f"_{rep.lower()}"):
            path = os.path.join(outputs_dir, d, "best.pt")
            if os.path.exists(path):
                candidates.append((os.path.getmtime(path), path))
    if not candidates: return None
    candidates.sort(reverse=True)
    return candidates[0][1]

if __name__ == "__main__":
    # 1. Summarize Ablation
    ablation_results = {}
    # Variants from scripts/run_lie_ablation.py
    # A1: lie_raw (Geodesic)
    # A2: lie_hardened (Geodesic)
    # A3: lie_raw (Antipodal)
    # A4: lie_fullfix (Antipodal)
    
    # We need to find the specific folders for these. 
    # For now, let's just find the last few lie runs.
    for rep in ["lie_raw", "lie_hardened", "Lie_FullFix"]:
        ckpt = find_latest_ckpt(rep)
        if ckpt:
            ablation_results[rep] = run_benchmark(ckpt, rep)
            
    with open("results/ablation_summary.json", "w") as f:
        json.dump(ablation_results, f, indent=2)
        
    print("Aggregate Phase 2 Done.")
