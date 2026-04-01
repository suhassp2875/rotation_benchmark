import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "axes.facecolor": "white",
    "grid.color": "#E5E5E5",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5
})

REPS = ["SVD", "6D", "Atlas", "Lie_FullFix", "Quat", "Euler"]
LABELS = {
    "SVD": "SVD", "6D": "6D", "Atlas": "Atlas",
    "Lie_FullFix": "Lie (fixed)", "Quat": "Quaternion", "Euler": "Euler"
}
COLORS = {
    "SVD": "#1f77b4", "6D": "#2ca02c", "Atlas": "#ff7f0e",
    "Lie_FullFix": "#9467bd", "Quat": "#d62728", "Euler": "#4d4d4d"
}

def main():
    try:
        with open("results/stratified_summary.json") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: results/stratified_summary.json not found.")
        return

    # Extract metrics for safe-region median and antipodal-region tails
    medians = [data[r]["safe"]["median"] for r in REPS]
    p99s = [data[r]["antipodal"]["p99"] for r in REPS]
    fails = [data[r]["antipodal"]["fail_rate"] for r in REPS]
    
    display_labels = [LABELS[r] for r in REPS]
    colors = [COLORS[r] for r in REPS]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), gridspec_kw={"wspace": 0.35})
    
    # Panel 1: Median Error
    ax = axes[0]
    bars = ax.bar(display_labels, medians, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_title("Safe-Region Parity\n(Median Geodesic Error)", fontweight="bold")
    ax.set_ylabel("Error (degrees)")
    ax.set_ylim(0, 15)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2, f"{height:.1f}°", ha="center", va="bottom", fontsize=8)

    # Panel 2: P99 Error
    ax = axes[1]
    bars = ax.bar(display_labels, p99s, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_title("Worst-Case Divergence\n(99th Percentile Error)", fontweight="bold")
    ax.set_ylabel("Error (degrees)")
    ax.set_ylim(0, 185)
    ax.axhline(180, color="black", linestyle=":", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2, f"{height:.0f}°", ha="center", va="bottom", fontsize=8)

    # Panel 3: Failure Rate
    ax = axes[2]
    bars = ax.bar(display_labels, fails, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_title("Operational Reliability\n(Failure Rate > 30°)", fontweight="bold")
    ax.set_ylabel("Frequency (%)")
    ax.set_ylim(0, max(fails) * 1.25)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2, f"{height:.1f}%", ha="center", va="bottom", fontsize=8)

    for ax in axes:
        ax.set_xticklabels(display_labels, rotation=35, ha="right")
        ax.tick_params(axis='x', length=0)

    plt.tight_layout()
    out_path = "results/final_manuscript_figures/fig1_benchmark_summary.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Generated: {out_path}")

if __name__ == "__main__":
    main()
