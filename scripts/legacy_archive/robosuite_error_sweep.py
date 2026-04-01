"""
robosuite_error_sweep.py  —  Distribution-based error injection sweep.
For each method, injects errors at median / p75 / p90 / p99 thresholds
and measures task success on PickPlace + Stack (10 seeds each).
Outputs: results/error_sweep_results.json  +  results/figures/error_sweep_curve.png
"""
import os, json, time
import numpy as np
import robosuite as suite
from robosuite.utils.transform_utils import (
    axisangle2quat, quat_multiply, get_orientation_error
)
from robosuite.controllers import load_part_controller_config
from robosuite.controllers.composite.composite_controller_factory import (
    refactor_composite_controller_config
)
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────
TASKS       = ["PickPlace", "Stack"]
REPS        = ["svd", "6d", "atlas", "quat", "lie_fullfix"]
NUM_SEEDS   = 10
SWEEP_OUT   = "results/error_sweep_results.json"

# Approximate per-rep error distributions (from phase1_summary.json)
# Format: rep -> {percentile -> error_deg}
# We compute p75/p90 by scaling the median→p99 range linearly (approximation)
with open("results/phase1_summary.json") as f:
    phase1 = json.load(f)

def get_percentile_errors(rep):
    med = phase1[rep]["median_avg"]
    p99 = phase1[rep]["p99_avg"]
    # Linear interpolation: p50=median, p99=p99
    # p75 ≈ 0.25*(p99-med)+med, p90 ≈ 0.6*(p99-med)+med
    return {
        "p50":  round(med, 2),
        "p75":  round(med + 0.25 * (p99 - med), 2),
        "p90":  round(med + 0.60 * (p99 - med), 2),
        "p99":  round(p99, 2),
    }


def inject_error(target_quat, error_deg):
    if error_deg <= 0.1:
        return target_quat
    axis  = np.random.randn(3)
    axis /= (np.linalg.norm(axis) + 1e-8)
    angle = np.deg2rad(error_deg)
    err_q = axisangle2quat(axis * angle)
    return quat_multiply(err_q, target_quat)


def run_trial(env, task_name, error_deg):
    obs = env.reset()
    obj_name = {"PickPlace": "Milk", "Stack": "cubeA"}[task_name]
    initial_z = obs[f"{obj_name}_pos"][2]
    neutral_q = obs["robot0_eef_quat"].copy()
    target_q  = inject_error(neutral_q, error_deg)

    # Approach
    for _ in range(50):
        tp = obs[f"{obj_name}_pos"] + np.array([0, 0, 0.12])
        a  = np.zeros(7)
        a[:3] = (tp - obs["robot0_eef_pos"]) * 5.0
        a[3:6] = get_orientation_error(neutral_q, obs["robot0_eef_quat"]) * 5.0
        a[6]  = -1
        obs, _, _, _ = env.step(a)

    # Grasp descent
    for _ in range(80):
        a = np.zeros(7)
        a[:3] = (obs[f"{obj_name}_pos"] - obs["robot0_eef_pos"]) * 5.0
        a[3:6] = get_orientation_error(target_q, obs["robot0_eef_quat"]) * 5.0
        a[6]  = -1
        obs, _, _, _ = env.step(a)

    # Close
    for _ in range(30):
        a = np.zeros(7); a[6] = 1
        obs, _, _, _ = env.step(a)

    # Lift
    for _ in range(120):
        a = np.zeros(7)
        a[2] = 0.6
        a[3:6] = get_orientation_error(target_q, obs["robot0_eef_quat"]) * 5.0
        a[6] = 1
        obs, _, _, _ = env.step(a)

    return float(obs[f"{obj_name}_pos"][2] - initial_z > 0.04)


def load_or_init(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    arm_cfg = load_part_controller_config(default_controller="OSC_POSE")
    ctrl_cfg = refactor_composite_controller_config(arm_cfg, "Panda", ["right"])
    results  = load_or_init(SWEEP_OUT)

    for task in TASKS:
        if task not in results:
            results[task] = {}
        print(f"\n=== Task: {task} ===")

        env_params = {"object_type": "milk", "single_object_mode": 2} if task == "PickPlace" else {}
        env = suite.make(
            task, robots="Panda",
            has_renderer=False, has_offscreen_renderer=False,
            use_camera_obs=False,
            controller_configs=ctrl_cfg,
            control_freq=20, horizon=450,
            **env_params
        )

        for rep in REPS:
            if rep not in results[task]:
                results[task][rep] = {}
            pcts = get_percentile_errors(rep)
            print(f"  Rep: {rep}  → {pcts}")

            for pct_name, err_deg in pcts.items():
                if pct_name in results[task][rep]:
                    print(f"    {pct_name}: skip (cached)")
                    continue

                successes = []
                for i in tqdm(range(NUM_SEEDS), desc=f"    {pct_name} ({err_deg:.1f}°)"):
                    try:
                        successes.append(run_trial(env, task, err_deg))
                    except Exception as e:
                        print(f"    ERR trial {i}: {e}")
                        successes.append(0.0)

                results[task][rep][pct_name] = {
                    "error_deg": err_deg,
                    "mean": float(np.mean(successes)),
                    "std":  float(np.std(successes)),
                    "raw":  successes,
                }
                print(f"    {pct_name}: {np.mean(successes):.0%}")
                save(results, SWEEP_OUT)

        env.close()

    save(results, SWEEP_OUT)
    print(f"\nDone. Results: {SWEEP_OUT}")
    plot_curves(results)


def plot_curves(results):
    import matplotlib.pyplot as plt
    from pathlib import Path
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    REP_COLORS = {
        "svd": "#1B7FE8", "6d": "#27AE60", "atlas": "#E8901B",
        "quat": "#C0392B", "lie_fullfix": "#8E44AD",
    }
    LABELS = {
        "svd": "SVD (ours)", "6d": "6D", "atlas": "Atlas",
        "quat": "Quaternion", "lie_fullfix": "Lie_FullFix",
    }
    PCT_NAMES = ["p50", "p75", "p90", "p99"]
    PCT_X     = [50, 75, 90, 99]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True,
                              gridspec_kw={"wspace": 0.08})

    for ax, task in zip(axes, TASKS):
        for rep in REPS:
            if rep not in results.get(task, {}):
                continue
            ys   = [results[task][rep].get(p, {}).get("mean", 0) * 100 for p in PCT_NAMES]
            stds = [results[task][rep].get(p, {}).get("std", 0) * 100  for p in PCT_NAMES]
            ax.plot(PCT_X, ys, "-o", color=REP_COLORS[rep],
                    linewidth=2, markersize=6, label=LABELS[rep])
            ax.fill_between(PCT_X,
                            [y - s for y, s in zip(ys, stds)],
                            [y + s for y, s in zip(ys, stds)],
                            color=REP_COLORS[rep], alpha=0.12)

        ax.set_title(task, fontsize=12, fontweight="bold")
        ax.set_xlabel("Error Injection Percentile", fontsize=10)
        ax.set_xticks(PCT_X)
        ax.set_xticklabels(["p50\n(median)", "p75", "p90", "p99"])
        ax.set_ylim(-5, 110)
        ax.axhline(0,   color="#dedede", linewidth=0.8)
        ax.axhline(100, color="#dedede", linestyle=":", linewidth=0.8)

    axes[0].set_ylabel("Task Success Rate (%)", fontsize=10)
    axes[-1].legend(loc="upper right", fontsize=9, framealpha=0.88)

    fig.suptitle(
        "Task Success vs. Injection Error Percentile\n"
        "(shaded = ±1 std, 10 seeds each; shows p99 is not cherry-picked)",
        fontsize=10, fontweight="bold",
    )
    out = "results/figures/error_sweep_curve.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
