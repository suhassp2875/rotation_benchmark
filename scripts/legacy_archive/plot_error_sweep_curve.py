"""
plot_error_sweep_curve.py  —  Plot error sweep results from whatever tasks are complete.
Generates separate panels per task (not stacked), so partial data still looks clean.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

with open("results/error_sweep_results.json") as f:
    data = json.load(f)

REPS = ["svd", "6d", "atlas", "quat", "lie_fullfix"]
LABELS = {
    "svd": "SVD (ours)", "6d": "6D", "atlas": "Atlas",
    "quat": "Quaternion", "lie_fullfix": "Lie_FullFix",
}
REP_COLORS = {
    "svd": "#1B7FE8", "6d": "#27AE60", "atlas": "#E8901B",
    "quat": "#C0392B", "lie_fullfix": "#8E44AD",
}
PCT_NAMES = ["p50", "p75", "p90", "p99"]
PCT_LABELS = ["p50\n(median)", "p75", "p90", "p99\n(tail)"]

available_tasks = [t for t in ["PickPlace", "Stack"] if t in data]
n_tasks = len(available_tasks)

fig, axes = plt.subplots(1, n_tasks, figsize=(6.5 * n_tasks, 5.2), sharey=True,
                          gridspec_kw={"wspace": 0.06})
if n_tasks == 1:
    axes = [axes]

for ax, task in zip(axes, available_tasks):
    task_data = data[task]
    plotted = []
    for rep in REPS:
        if rep not in task_data:
            continue
        ys   = []
        stds = []
        xs   = []
        for i, pct in enumerate(PCT_NAMES):
            if pct in task_data[rep]:
                ys.append(task_data[rep][pct]["mean"] * 100)
                stds.append(task_data[rep][pct]["std"] * 100)
                xs.append(i)

        if not ys:
            continue

        ax.plot(xs, ys, "-o", color=REP_COLORS[rep], linewidth=2.2,
                markersize=7, label=LABELS[rep], zorder=3)
        ax.fill_between(
            xs,
            [y - s for y, s in zip(ys, stds)],
            [y + s for y, s in zip(ys, stds)],
            color=REP_COLORS[rep], alpha=0.13, zorder=2,
        )
        plotted.append(rep)

    ax.set_title(task, fontsize=13, fontweight="bold", pad=8)
    ax.set_xticks(range(len(PCT_NAMES)))
    ax.set_xticklabels(PCT_LABELS, fontsize=9)
    ax.set_xlabel("Error Injection Percentile", fontsize=10)
    ax.set_ylim(-5, 112)
    ax.axhline(0,   color="#e0e0e0", linewidth=0.8)
    ax.axhline(100, color="#e0e0e0", linewidth=0.8, linestyle=":")
    ax.grid(axis="y", alpha=0.3)

    # Annotate the cliff zone
    ax.axvspan(1.5, 2.5, alpha=0.07, color="#e74c3c", zorder=1)
    ax.text(2.0, 108, "Cliff\nzone", ha="center", fontsize=7.5,
            color="#c0392b", style="italic")

axes[0].set_ylabel("Task Success Rate (%)", fontsize=10)
axes[-1].legend(loc="upper right", fontsize=9.5, framealpha=0.9)

fig.suptitle(
    "Task Success vs. Error Injection Percentile\n(shaded = ±1 std, 10 seeds)",
    fontsize=10.5, fontweight="bold", y=1.01,
)


plt.tight_layout()
out = OUT / "error_sweep_curve.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out}  (tasks: {available_tasks})")
