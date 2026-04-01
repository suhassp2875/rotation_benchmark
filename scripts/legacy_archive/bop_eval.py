"""
bop_eval.py  —  Evaluate each trained rotation head on YCB-V object point clouds.
Tests whether synthetic benchmark ranking transfers to a standard BOP pose dataset.
"""
import json, sys, os
import numpy as np
from pathlib import Path
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

from model_loader import load_from_checkpoint, predict_R

BOP_MODELS   = Path("bop/ycbv/ycbv/models")
CKPT_DIR     = Path("checkpoints")
RESULTS_PATH = Path("results/bop_eval_results.json")
FIG_PAPER    = Path("results/paper_figures/fig8_bop_comparison.png")
FIG_EXTRA    = Path("results/figures/bop_comparison.png")
FIG_PAPER.parent.mkdir(parents=True, exist_ok=True)

N_OBJ      = 5
N_SAMPLES  = 200
N_POINTS   = 32      # points per cloud
IN_DIM     = 96     # N_POINTS * 3 = model input size
NOISE_STD  = 0.01
THRESHOLDS = [5, 10, 15, 30]

CHECKPOINTS = {
    "SVD (ours)":  sorted(Path("outputs").rglob("*SVD*/best.pt"))[-1],
    "6D":          sorted(Path("outputs").rglob("*6D*/best.pt"))[-1],
    "Atlas":       CKPT_DIR / "atlas_checkpoint.pt",
    "Lie_FullFix": CKPT_DIR / "lie_fullfix_checkpoint.pt",
}

COLORS = {
    "SVD (ours)": "#1B7FE8", "6D": "#27AE60",
    "Atlas": "#E8901B", "Lie_FullFix": "#8E44AD",
}


def load_ply_points(path, n=N_POINTS):
    """Read vertex positions from a .ply file, return Nx3 array."""
    pts = []
    with open(path, 'rb') as f:
        n_verts, fmt = 0, "ascii"
        for _ in range(100):
            line = f.readline().decode("ascii", errors="ignore").strip()
            if "format binary" in line:
                fmt = "binary"
            if line.startswith("element vertex"):
                n_verts = int(line.split()[-1])
            if line == "end_header":
                break
        if n_verts == 0:
            return None
        indices = set(np.random.choice(n_verts, size=min(n, n_verts), replace=False))
        if fmt == "binary":
            raw = np.frombuffer(f.read(n_verts * 12), dtype=np.float32).reshape(n_verts, 3)
            pts = raw[list(indices)]
        else:
            for i in range(n_verts):
                row = f.readline().decode("ascii", errors="ignore").split()
                if i in indices:
                    try:
                        pts.append([float(row[0]), float(row[1]), float(row[2])])
                    except (ValueError, IndexError):
                        pass
            pts = np.array(pts, dtype=np.float32)
    return np.array(pts, dtype=np.float32)


def random_rotation():
    M = np.random.randn(3, 3)
    Q, _ = np.linalg.qr(M)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q.astype(np.float32)


def geodesic_deg(R_pred, R_gt):
    R_rel = R_pred @ R_gt.T
    cos   = np.clip((np.trace(R_rel) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cos)))


def make_input(pts, R_gt):
    """Apply rotation + noise to point cloud, return 96-dim input."""
    rotated = pts @ R_gt.T
    noisy   = rotated + np.random.randn(*rotated.shape).astype(np.float32) * NOISE_STD
    flat    = noisy.flatten()     # 32×3 = 96
    return flat




def main():
    np.random.seed(42)
    print("Loading YCB-V mesh point clouds...")
    ply_files = sorted(BOP_MODELS.glob("obj_*.ply"))[:N_OBJ]
    clouds = []
    for pf in ply_files:
        pts = load_ply_points(pf, N_POINTS)
        if pts is not None and len(pts) >= N_POINTS:
            pts = pts[:N_POINTS]
            pts -= pts.mean(0)
            pts /= np.linalg.norm(pts, axis=1).max() + 1e-8
            clouds.append(pts)
            print(f"  {pf.name}")

    results = {}
    for name, ckpt_path in CHECKPOINTS.items():
        print(f"\n--- {name} ({ckpt_path.name}) ---")
        model, rep = load_from_checkpoint(ckpt_path)
        if model is None:
            continue
        errors = []
        n_per = N_SAMPLES // max(len(clouds), 1)
        for pts in clouds:
            for _ in range(n_per):
                R_gt = random_rotation()
                x    = make_input(pts, R_gt)
                R_pred = predict_R(model, x)
                if R_pred is None:
                    errors.append(180.0)
                else:
                    errors.append(geodesic_deg(R_pred, R_gt))

        errors = np.array(errors)
        recall = {f"@{t}": float((errors <= t).mean() * 100) for t in THRESHOLDS}
        results[name] = {
            "median": float(np.median(errors)),
            "p99":    float(np.percentile(errors, 99)),
            "recall": recall,
            "n":      len(errors),
        }
        print(f"  median={results[name]['median']:.2f}°  "
              f"p99={results[name]['p99']:.2f}°  "
              f"@10°={recall['@10']:.1f}%")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {RESULTS_PATH}")
    plot(results)


def plot(results):
    reps = list(results.keys())
    x    = np.arange(len(reps))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                    gridspec_kw={"wspace": 0.35})
    bar_w = 0.18
    thresholds = [5, 10, 15, 30]
    xi = np.arange(len(thresholds))

    for i, rep in enumerate(reps):
        recalls = [results[rep]["recall"].get(f"@{t}", 0) for t in thresholds]
        ax1.bar(xi + i * bar_w, recalls, bar_w, label=rep,
                color=COLORS.get(rep, "#888"), alpha=0.88)
    ax1.set_xticks(xi + bar_w * (len(reps) - 1) / 2)
    ax1.set_xticklabels([f"@{t}°" for t in thresholds])
    ax1.set_ylabel("Recall (%)", fontsize=10)
    ax1.set_title("Recall @ Threshold", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    medians = [results[r]["median"] for r in reps]
    p99s    = [results[r]["p99"]    for r in reps]
    ax2.bar(x - 0.2, medians, 0.38, label="Median", color=[COLORS.get(r,"#888") for r in reps], alpha=0.88)
    ax2.bar(x + 0.2, p99s,    0.38, label="p99",    color=[COLORS.get(r,"#888") for r in reps], alpha=0.42,
            edgecolor=[COLORS.get(r,"#888") for r in reps], linewidth=1.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(reps, rotation=12, ha="right", fontsize=9)
    ax2.set_ylabel("Geodesic Error (°)", fontsize=10)
    ax2.set_title("Median & p99 Error", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"BOP YCB-V Evaluation — Does Synthetic Ranking Transfer?\n"
                 f"({N_OBJ} objects × {N_SAMPLES} samples, σ_noise={NOISE_STD})",
                 fontsize=10, fontweight="bold", y=1.02)
    plt.tight_layout()
    for p in [FIG_PAPER, FIG_EXTRA]:
        plt.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print("Saved BOP figure.")


if __name__ == "__main__":
    main()
