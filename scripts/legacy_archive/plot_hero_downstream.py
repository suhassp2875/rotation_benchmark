import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from matplotlib.lines import Line2D

def main():
    try:
        with open("results/downstream_comprehensive_eval.json") as f:
            data = json.load(f)
        with open("results/stratified_summary.json") as f:
            bench_data = json.load(f)
    except Exception as e:
        print("Could not load results JSONs:", e)
        return

    TASKS = list(data.keys())
    
    REP_COLORS = {
        "Oracle_GT":   "#000000",   
        "SVD":         "#1B7FE8",   
        "6D":          "#27AE60",   
        "Atlas":       "#E8901B",   
        "Quat":        "#C0392B",   
        "Lie_FullFix": "#8E44AD",
        "Euler":       "#7F8C8D"
    }
    TASK_MARKERS = {"PickPlace": "o", "Stack": "s", "NutAssembly": "^"}

    def get_bench_p99(rep, reg):
        if rep == "Oracle_GT": return 0.0
        return bench_data.get(rep, {}).get(reg, {}).get("p99", None)
        
    def get_ds_success(task, rep, reg):
        trials = data.get(task, {}).get(rep, {}).get("trials", [])
        # Fallback to uniform if region_type is missing (for older eval) but we know it's there
        reg_trials = [t for t in trials if t.get("region_type") == reg]
        if not reg_trials: return None
        return np.mean([t["success"] for t in reg_trials]) * 100.0

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    
    for idx, (ax, region, title) in enumerate([
        (axes[0], "gimbal", "Panel A: Gimbal Region (Benchmark vs. Downstream)"),
        (axes[1], "antipodal", "Panel B: Antipodal Region (Benchmark vs. Downstream)")
    ]):
        x_vals, y_vals = [], []
        
        for task in TASKS:
            for rep in list(data[task].keys()):
                if rep not in REP_COLORS: continue
                x = get_bench_p99(rep, region)
                y = get_ds_success(task, rep, region)
                
                if x is None or y is None: continue
                
                x_vals.append(x)
                y_vals.append(y)
                
                jitter_y = {"PickPlace": 1.5, "Stack": -1.5, "NutAssembly": 0}.get(task, 0)
                
                ax.scatter(
                    x, y + jitter_y,
                    marker=TASK_MARKERS[task],
                    color=REP_COLORS[rep],
                    s=120, alpha=0.9, edgecolors="white", linewidths=1.0, zorder=3
                )
                
                if task == "PickPlace":
                    ax.annotate(
                        rep, (x, y + jitter_y),
                        textcoords="offset points", xytext=(8, -8),
                        fontsize=9, color=REP_COLORS[rep], fontweight="bold"
                    )

        if len(x_vals) > 1:
            s_corr, _ = spearmanr(x_vals, y_vals)
            p_corr, _ = pearsonr(x_vals, y_vals)
            ax.set_title(f"{title}\nSpearman ρ: {s_corr:.3f} | Pearson r: {p_corr:.3f}", fontsize=11)
            
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            x_line = np.linspace(0, max(x_vals) * 1.05, 100)
            ax.plot(x_line, p(x_line), "--", color="gray", linewidth=1.5, alpha=0.6, zorder=1)
        
        ax.set_xlabel(f"Benchmark P99 Error in {region.title()} Region (°)", fontsize=11)
        if idx == 0: ax.set_ylabel("Robosuite Success Rate in Region (%)", fontsize=11)
        ax.set_ylim(-10, 115)

    task_legend = [Line2D([0], [0], marker=TASK_MARKERS[t], color="gray", linestyle="None", markersize=8, label=t) for t in TASKS]
    axes[1].legend(handles=task_legend, loc="best", fontsize=10, framealpha=0.9)

    plt.suptitle("Predicting Downstream Failure via Geometrically Stratified Tail-Risk", fontsize=14, y=1.03)
    plt.tight_layout()
    out_path = "results/hero_downstream.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved hero figure to: {out_path}")
    
if __name__ == "__main__":
    main()
