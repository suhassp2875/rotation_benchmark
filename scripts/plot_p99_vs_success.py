"""
plot_p99_vs_success.py  —  p99 rotation error vs. downstream manipulation success
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Load data ────────────────────────────────────────────────────────────────
with open("results/multitask_grasp_results.json") as f:
    multitask = json.load(f)

with open("results/phase1_summary.json") as f:
    phase1 = json.load(f)

# ── Config ───────────────────────────────────────────────────────────────────
TASKS = ["PickPlace", "Stack"]
REPS = ["svd", "6d", "atlas", "quat", "lie_fullfix"]
LABELS = {
    "svd":         "SVD",
    "6d":          "6D",
    "atlas":       "Atlas",
    "quat":        "Quaternion",
    "lie_fullfix": "Lie (fixed)",
}
# Each representation gets a distinct color
REP_COLORS = {
    "svd":         "#1f77b4",   # blue
    "6d":          "#2ca02c",   # green
    "atlas":       "#ff7f0e",   # orange
    "quat":        "#d62728",   # red
    "lie_fullfix": "#9467bd",   # purple
}
TASK_MARKERS = {
    "PickPlace": "o",
    "Stack":     "s",
}
# Small per-rep x-nudge to prevent overlap among reps with similar p99
REP_X_NUDGE = {
    "svd":         -1.0,
    "6d":          +1.0,
    "atlas":        0.0,
    "quat":        -1.0,
    "lie_fullfix": +1.0,
}
# Label offsets (x, y) per rep to avoid crossing
LABEL_OFFSET = {
    "svd":         (-8, 8),
    "6d":          (6, -12),
    "atlas":       (6, 5),
    "quat":        (-8, 8),
    "lie_fullfix": (14, -12),
}

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "axes.facecolor": "white",
    "grid.color": "#E5E5E5",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "font.size": 10
})
fig, ax = plt.subplots(figsize=(9, 6.2))

# Task y-jitter keeps PickPlace/Stack points from sitting on top of each other
TASK_Y_JITTER = {"PickPlace": +1.5, "Stack": -1.5}

for task in TASKS:
    for rep in REPS:
        p99  = phase1[rep]["p99_avg"] if rep in phase1 else 0.0
        mean = multitask[task][rep]["mean"] * 100
        std  = multitask[task][rep]["std"]  * 100
        x    = p99 + REP_X_NUDGE[rep]
        y    = mean + TASK_Y_JITTER[task]

        ax.errorbar(
            x, y, yerr=std,
            fmt=TASK_MARKERS[task],
            color=REP_COLORS[rep],
            markersize=9,
            ecolor=REP_COLORS[rep],
            elinewidth=1.5,
            capsize=4,
            zorder=3,
            alpha=0.90,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )

        # Label once (PickPlace only) with offset per rep
        if task == "PickPlace":
            ox, oy = LABEL_OFFSET[rep]
            ax.annotate(
                LABELS[rep],
                (x, y),
                textcoords="offset points",
                xytext=(ox, oy),
                fontsize=8.5,
                color=REP_COLORS[rep],
                fontweight="semibold",
                ha="right" if ox < 0 else "left",
            )

# Trend line
xs, ys = [], []
for task in TASKS:
    for rep in REPS:
        xs.append(phase1[rep]["p99_avg"] if rep in phase1 else 0.0)
        ys.append(multitask[task][rep]["mean"] * 100)

rho, _ = spearmanr(xs, ys)
z = np.polyfit(xs, ys, 1)
p = np.poly1d(z)
x_line = np.linspace(-3, max(xs) + 5, 300)
ax.plot(x_line, p(x_line), "--", color="#aaaaaa", linewidth=1.3, alpha=0.8, zorder=1)

task_legend = [
    Line2D([0], [0], marker="o", color="gray", linestyle="None", markersize=8, label="PickPlace"),
    Line2D([0], [0], marker="s", color="gray", linestyle="None", markersize=8, label="Stack"),
    Line2D([0], [0], linestyle="--", color="#aaaaaa", linewidth=1.3, label="Linear trend"),
]
ax.legend(handles=task_legend, loc="upper right", fontsize=9, framealpha=0.88)

ax.set_xlabel("Benchmark Tail-Risk ($p_{99}$ Geodesic Error in Degrees)", fontsize=11)
ax.set_ylabel("Downstream Manipulation Success Rate (%)", fontsize=11)
ax.set_title("Operational Predictability: Tail-Risk vs. Task Success", fontsize=12, fontweight='bold', pad=20)

ax.set_xlim(-5, max(xs) + 10)
ax.set_ylim(-10, 118)
ax.axhline(0,   color="#dddddd", linewidth=0.8)
ax.axhline(100, color="#dddddd", linewidth=0.8, linestyle=":")

# Spearman Label (Nudged)
ax.text(0.95, 0.85, fr"Spearman $\rho = {rho:.3f}$", 
        transform=ax.transAxes, fontsize=12, fontweight='bold',
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=1.0, edgecolor='black', linewidth=1.0))

plt.tight_layout()
out = "results/final_manuscript_figures/fig4_predictive_correlation.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Generated: {out}")

if __name__ == "__main__":
    pass 
