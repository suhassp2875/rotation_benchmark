"""
Error Cliff Plot
================
Plots task success vs. injected error magnitude (cliff curves), with
vertical lines showing each representation's benchmark p99 and risk-margin
shading to make the safe/unsafe zone immediately visible.
"""

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

CLIFF_PATH  = Path("results/error_cliff_sweep.json")
BENCH_PATH  = Path("results/stratified_summary.json")
OUT_PATH    = Path("results/hero_cliff.png")

# Representation display names, colors, p99 source key
REPS = [
    ("6D",          "#2ca02c", "6D"),
    ("SVD",         "#1f77b4", "SVD"),
    ("Atlas",       "#ff7f0e", "Atlas"),
    ("Lie (fixed)", "#9467bd", "Lie_FullFix"),
    ("Quaternion",  "#d62728", "Quat"),
    ("Euler",       "#4d4d4d", "Euler"),
]

TASK_STYLES = {
    "PickPlace": {"color": "#333333", "ls": "-",  "lw": 2.2},
    "Stack":     {"color": "#666666", "ls": "--", "lw": 2.2},
    "NutAssembly": {"color": "#999999", "ls": ":", "lw": 2.2},
}

def load_data():
    with open(CLIFF_PATH)  as f: cliff  = json.load(f)
    with open(BENCH_PATH)  as f: bench  = json.load(f)
    return cliff, bench

def get_p99(bench, rep_key):
    """Return overall (uniform region) p99 for a representation."""
    rec = bench.get(rep_key, {})
    return rec.get("uniform", {}).get("p99", None)

def cliff_curve(cliff_task: dict):
    """Return sorted (error_deg, success_rate) arrays."""
    xs, ys = [], []
    for k, v in cliff_task.items():
        xs.append(float(k))
        ys.append(np.mean(v) * 100.0)
    order = np.argsort(xs)
    return np.array(xs)[order], np.array(ys)[order]

def find_cliff_crossing(xs, ys, threshold=50.0):
    """Find the x value where success drops below threshold."""
    for i in range(len(ys) - 1):
        if ys[i] >= threshold >= ys[i+1]:
            # Linear interpolation
            frac = (ys[i] - threshold) / (ys[i] - ys[i+1] + 1e-9)
            return xs[i] + frac * (xs[i+1] - xs[i])
    return None

def main():
    cliff, bench = load_data()

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "axes.facecolor": "white",
        "grid.color": "#E5E5E5",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "font.size": 10
    })
    
    fig, ax = plt.subplots(figsize=(11, 7))

    tasks_in_data = list(cliff.keys())
    cliff_crossings = {}

    # ── 1. Plot cliff curves ──────────────────────────────────────────────────
    for task in tasks_in_data:
        style = TASK_STYLES.get(task, {"color": "black", "ls": "-", "lw": 2.5})
        xs, ys = cliff_curve(cliff[task])
        ys_smooth = gaussian_filter1d(ys, sigma=0.6)

        ax.plot(xs, ys_smooth,
                color=style["color"], ls=style["ls"], lw=style["lw"],
                label=task, zorder=4)
        ax.scatter(xs, ys, color=style["color"], s=40, zorder=5, alpha=0.5)

        cx = find_cliff_crossing(xs, ys_smooth)
        if cx:
            cliff_crossings[task] = cx
            ax.axvline(cx, color=style["color"], ls=":", lw=1.0, alpha=0.4)

    # ── 2. Overlay representation p99 lines and risk-margin shading ───────────
    for rep_name, color, bench_key in REPS:
        p99 = get_p99(bench, bench_key)
        if p99 is None: continue

        # Truncate for visual plotting if > 90
        plot_x = min(p99, 92)
        label_suffix = f"p99 = {p99:.0f}°"
        
        ax.axvline(plot_x, color=color, lw=2.2 if rep_name == "Euler" else 1.8, 
                   ls="--" if rep_name == "Euler" else "-", alpha=0.8, zorder=3,
                   label=f"{rep_name} {label_suffix}")

        crossings = [v for v in cliff_crossings.values()]
        if crossings and p99 > min(crossings):
            ax.axvspan(min(crossings), plot_x, alpha=0.06, color=color, zorder=1)

    # ── 3. Quantitative Risk Prediction Table (Inset) ────────────────────────
    table_data = []
    header = ["Rep", "p99", "P.Place", "Stack"]
    for rep_name, _, bench_key in REPS:
        p99 = get_p99(bench, bench_key)
        if p99 is None: continue
        row = [rep_name, f"{p99:.1f}°"]
        for task in ["PickPlace", "Stack"]:
            cx = cliff_crossings.get(task)
            row.append("RISK" if (cx and p99 > cx) else "SAFE")
        table_data.append(row)

    the_table = plt.table(cellText=table_data, colLabels=header, loc='upper right',
                          bbox=[0.65, 0.65, 0.33, 0.3], zorder=10)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(7.5)

    # ── 4. Formatting ────────────────────────────────────────────────────────
    ax.axhline(50, color="gray", lw=0.8, ls=":", alpha=0.5, zorder=2)
    ax.set_xlabel("Injected Rotational Error (degrees)", fontsize=11)
    ax.set_ylabel("Task Success Rate (%)", fontsize=11)
    ax.set_xlim(-2, 95)
    ax.set_ylim(-5, 110)
    ax.set_title("Operational Failure Boundaries (The 'Success Cliff')\n", fontsize=12, fontweight='bold')
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower left", fontsize=8.5, framealpha=0.9, ncol=2)

    plt.tight_layout()
    out_path = "results/final_manuscript_figures/fig3_hero_cliff.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Generated: {out_path}")

if __name__ == "__main__":
    main()
