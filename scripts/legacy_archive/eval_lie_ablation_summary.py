"""
eval_lie_ablation_summary.py  —  Aggregate existing Lie ablation checkpoints and plot.
"""
import json, sys, os
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")
from model_loader import load_from_checkpoint, predict_R
from rotation_benchmark.data.dataset import RotationPointCloudDataset

GOLD_PATH = Path("results/gold_standard.json")
RESULTS   = Path("results/lie_ablation_results.json")
FIG1_OUT  = Path("results/paper_figures/fig9_lie_ablation.png")
FIG2_OUT  = Path("results/figures/lie_ablation.png")
FIG1_OUT.parent.mkdir(parents=True, exist_ok=True)


def geodesic_deg(R_pred, R_gt):
    if R_pred is None: return 180.0
    R_rel = R_pred @ R_gt.T
    cos   = np.clip((np.trace(R_rel) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cos)))

def main():
    print(f"Loading unified results from {GOLD_PATH}...")
    if not GOLD_PATH.exists():
        print(f"  [ERROR] {GOLD_PATH} not found. Run unify_results.py first.")
        return

    with open(GOLD_PATH, "r") as f:
        gold = json.load(f)
    
    ab_data = gold.get("ablation", {})
    results = {}
    
    # Map internal keys to plotting labels
    key_map = {
        "a1_raw_lie":   "A1: Raw Lie\n(Baseline)",
        "a2_hardened":  "A2: Hardened Arch\n(No Loss)",
        "a3_antipodal": "A3: Antipodal Loss\n(No Arch)",
        "a4_fullfix":   "A4: Full FullFix\n(Arch+Loss)",
    }
    
    for int_key, label in key_map.items():
        if int_key in ab_data:
            results[label] = ab_data[int_key]


    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    plot(results)

def plot(results):
    labels  = list(results.keys())
    medians = [results[l]["median"] for l in labels]
    p99s    = [results[l]["p99"]    for l in labels]
    fails   = [results[l]["fail30"] for l in labels]
    x = np.arange(len(labels))
    colors = ["#95a5a6", "#e67e22", "#f1c40f", "#3498db"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"wspace": 0.4})
    for ax, vals, title, ylabel in [
        (axes[0], medians, "Median Error", "Geodesic Error (°)"),
        (axes[1], p99s,    "p99 Error",    "Geodesic Error (°)"),
        (axes[2], fails,   "Fail Rate (>30°)", "Failure (%)"),
    ]:
        bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{v:.1f}", ha="center", fontsize=8)

    fig.suptitle("Lie_FullFix Ablation: Component-Wise Contribution\n(A1 Baseline → A4 Full Fix)", fontsize=12, fontweight="bold", y=1.08)
    # No tight_layout here as it often breaks rotated labels, use subplots_adjust
    fig.subplots_adjust(bottom=0.22, top=0.85)

    for p in [FIG1_OUT, FIG2_OUT]:
        plt.savefig(p, dpi=160, bbox_inches="tight")
    plt.close()


    print("Saved Lie ablation summary table and plots.")

if __name__ == "__main__":
    main()
