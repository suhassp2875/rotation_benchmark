"""
unify_results.py  —  Run a large-scale definitive evaluation for all models.
Generates gold_standard.json to ensure zero numerical drift in the paper.
"""
import json, sys, os
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from model_loader import load_from_checkpoint, predict_R
from rotation_benchmark.data.dataset import RotationPointCloudDataset

OUTPUTS = Path("outputs")
CKPTS   = Path("checkpoints")
GOLD    = Path("results/gold_standard.json")
GOLD.parent.mkdir(parents=True, exist_ok=True)

N_EVAL = 5000  # Large N for high precision

HEADLINE_CKPTS = {
    "svd":          sorted(OUTPUTS.rglob("*SVD*/best.pt"))[-1],
    "6d":           CKPTS / "sixd_checkpoint.pt",
    "atlas":        CKPTS / "atlas_checkpoint.pt", # Will use Oracle
    "quat":         sorted(OUTPUTS.rglob("*QUAT*/best.pt"))[-1],
}

ABLATION_CKPTS = {
    "a1_raw_lie":      "2026-03-13/10-28-44_lie_raw",
    "a2_hardened":     "2026-03-13/10-28-51_lie_hardened",
    "a3_antipodal":    "2026-03-13/10-28-58_lie_raw",
    "a4_fullfix":      "2026-03-13/10-29-05_Lie_FullFix",
}

def geodesic_deg(R1, R2):
    if R1 is None or R2 is None: return 180.0
    R_rel = R1 @ R2.T
    cos   = np.clip((np.trace(R_rel) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cos)))

def eval_model(model, X, R_gts, use_oracle=False):
    errors = []
    for i in range(len(X)):
        x = X[i:i+1]
        if use_oracle:
            # Oracle logic for Atlas
            with torch.no_grad():
                out = model(torch.from_numpy(x))
            if isinstance(out, tuple) and out[0].ndim == 4:
                R_stack = out[0][0].numpy()
                best_err = 180.0
                for k in range(R_stack.shape[0]):
                    err = geodesic_deg(R_stack[k], R_gts[i])
                    best_err = min(best_err, err)
                errors.append(best_err)
            else:
                R_pred = predict_R(model, x[0])
                errors.append(geodesic_deg(R_pred, R_gts[i]))
        else:
            R_pred = predict_R(model, x[0])
            errors.append(geodesic_deg(R_pred, R_gts[i]))
    return np.array(errors)

def main():
    print(f"Generating definitive validation set (N={N_EVAL})...")
    ds = RotationPointCloudDataset(
        n=N_EVAL, mode="uniform", num_points=32, obs_noise_std=0.01, seed=999, device="cpu"
    )
    X = ds.x.numpy()
    R_gts = ds.R_gt.numpy()

    gold_data = {"headline": {}, "ablation": {}}
    
    # 1. Headline
    for name, path in HEADLINE_CKPTS.items():
        print(f"\nEvaluating Headline {name.upper()}...")
        model, _ = load_from_checkpoint(path)
        if model is None: continue
        
        use_oracle = (name == "atlas")
        errors = eval_model(model, X, R_gts, use_oracle=use_oracle)
        
        gold_data["headline"][name] = {
            "median":  float(np.median(errors)),
            "p99":     float(np.percentile(errors, 99)),
            "fail30":  float((errors > 30).mean() * 100),
            "oracle":  use_oracle
        }
        print(f"  Median: {gold_data['headline'][name]['median']:.3f}°  p99: {gold_data['headline'][name]['p99']:.3f}°")

    # 2. Ablation
    for name, substr in ABLATION_CKPTS.items():
        print(f"\nEvaluating Ablation {name.upper()}...")
        ckpts = sorted(OUTPUTS.glob(f"**/*{substr}*/best.pt"))
        if not ckpts: continue
        model, _ = load_from_checkpoint(ckpts[0])
        if model is None: continue
        
        errors = eval_model(model, X, R_gts, use_oracle=False)
        gold_data["ablation"][name] = {
            "median":  float(np.median(errors)),
            "p99":     float(np.percentile(errors, 99)),
            "fail30":  float((errors > 30).mean() * 100)
        }
        print(f"  Median: {gold_data['ablation'][name]['median']:.3f}°  p99: {gold_data['ablation'][name]['p99']:.3f}°")

    with open(GOLD, "w") as f:
        json.dump(gold_data, f, indent=2)
    print(f"\nSuccessfully unified results in {GOLD}")


if __name__ == "__main__":
    main()
