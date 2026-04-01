"""
make_sim_panel.py  —  Build clean simulation visual panels for the CoRL paper.
Produces 3 figures:
  A. PickPlace: SVD SUCCESS vs Quat FAILURE (agent+front)
  B. Stack: 6D SUCCESS vs Lie_FullFix FAILURE (agent+front)
  C. NutAssembly: all-FAILURE hard-ceiling panel
"""
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

BASE = Path("results/images/multitask")
OUT  = Path("results/figures")
OUT.mkdir(parents=True, exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────
REP_LABELS = {
    "svd": "SVD (ours)",
    "6d":  "6D",
    "atlas": "Atlas",
    "quat": "Quaternion",
    "lie_fullfix": "Lie_FullFix",
    "baseline_0": "Perfect Pose\n(no error)",
}
P99 = {
    "svd": "3.4°", "6d": "3.5°", "atlas": "29.1°",
    "quat": "64.4°", "lie_fullfix": "61.4°", "baseline_0": "0°",
}
SUCCESS_COLOR = "#27AE60"
FAILURE_COLOR = "#C0392B"
FONT = dict(fontsize=9, fontweight="bold")


def load(task, rep, status, view):
    prefix = "SUCCESS" if status else "FAILURE"
    p = BASE / task / f"{rep}_{prefix}_{view}.png"
    if not p.exists():
        # Try flipping status (e.g., atlas sometimes has both)
        prefix2 = "FAILURE" if status else "SUCCESS"
        p = BASE / task / f"{rep}_{prefix2}_{view}.png"
    if not p.exists():
        return None
    return np.array(Image.open(p))


def make_panel(task, pairs, title, out_name):
    """
    pairs: list of (rep, success:bool) tuples
    Each pair gets 2 columns (agent, front view).
    """
    n = len(pairs)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.8),
                             gridspec_kw={"hspace": 0.04, "wspace": 0.04})
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (rep, success) in enumerate(pairs):
        for row, view in enumerate(["agent", "front"]):
            ax = axes[row, col]
            img = load(task, rep, success, view)
            if img is not None:
                ax.imshow(img)
            else:
                ax.set_facecolor("#1a1a1a")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        color="gray", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])

            # Border color = success/failure
            color = SUCCESS_COLOR if success else FAILURE_COLOR
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)

            # Top-row header
            if row == 0:
                status_str = "✓ SUCCESS" if success else "✗ FAILURE"
                label = f"{REP_LABELS[rep]}\n{status_str}  (p99={P99[rep]})"
                ax.set_title(label, fontsize=9, fontweight="bold",
                             color=color, pad=4)

            # Left-col view label
            if col == 0:
                view_label = "Agent\nview" if view == "agent" else "Front\nview"
                ax.set_ylabel(view_label, fontsize=8, color="#555555",
                              labelpad=4)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    plt.savefig(OUT / out_name, dpi=160, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"Saved: {OUT / out_name}")


# ── Figure A: PickPlace ────────────────────────────────────────────────────
make_panel(
    "PickPlace",
    [("svd", True), ("6d", True), ("atlas", False), ("quat", False)],
    "PickPlace — p99 Error Injection Results\n(10 seeds; agent & front views)",
    "sim_panel_pickplace.png",
)

# ── Figure B: Stack ────────────────────────────────────────────────────────
make_panel(
    "Stack",
    [("svd", True), ("6d", True), ("lie_fullfix", False), ("quat", False)],
    "Stack — p99 Error Injection Results\n(10 seeds; agent & front views)",
    "sim_panel_stack.png",
)

# ── Figure C: NutAssembly (all fail — hard ceiling) ───────────────────────
make_panel(
    "NutAssembly",
    [("baseline_0", False), ("svd", False), ("6d", False), ("atlas", False)],
    "NutAssembly — Hard-Ceiling Diagnostic\n(All methods fail, including oracle-pose baseline; controller-precision-limited task)",
    "sim_panel_nutassembly.png",
)

print("All panels done.")
