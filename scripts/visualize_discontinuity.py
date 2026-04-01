import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

from model_loader import load_from_checkpoint
from rotation_benchmark.geometry.metrics import geodesic_distance

def get_prediction(model, angle_deg, axis=np.array([0.0, 0, 1])):
    """Get predicted rotation for a single ground truth angle."""
    angle_rad = np.radians(angle_deg)
    # Rodrigues for GT
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R_gt = np.eye(3) + np.sin(angle_rad)*K + (1-np.cos(angle_rad))*(K@K)
    
    # Use canonical points (EXACTLY as in training setup)
    rng = np.random.RandomState(0)
    pts = rng.normal(size=(32, 3))
    pts = pts / np.linalg.norm(pts, axis=-1, keepdims=True)
    points = pts.astype(np.float32)
    # Rotate points
    points_rot = points @ R_gt.T
    
    device = next(model.parameters()).device
    x = torch.from_numpy(points_rot).to(device).float().unsqueeze(0).flatten(1)
    
    with torch.no_grad():
        out = model(x)
        if isinstance(out, tuple):
            R_pred = out[0]
        else:
            R_pred = out
            
    # Project to SO(3) if necessary
    R_np = R_pred.squeeze().cpu().numpy()
    if R_np.shape != (3, 3):
        R_np = R_np.reshape(3, 3)
        
    U, _, Vt = np.linalg.svd(R_np)
    R_pred_so3 = U @ Vt
    if np.linalg.det(R_pred_so3) < 0:
        U[:, -1] *= -1
        R_pred_so3 = U @ Vt
        
    return R_pred_so3, R_gt

def main():
    checkpoint_map = {
        "Euler": "outputs/2026-03-30/09-16-24_euler/best.pt",
        "Quaternion": "outputs/2026-03-30/09-17-01_QUAT/best.pt",
        "6D": "checkpoints/sixd_checkpoint.pt",
        "Lie (fixed)": "checkpoints/lie_fullfix_checkpoint.pt"
    }
    # Unified Color Palette
    colors = {
        "SVD": "#1f77b4", 
        "6D": "#2ca02c", 
        "Atlas": "#ff7f0e", 
        "Lie (fixed)": "#9467bd", 
        "Quaternion": "#d62728", 
        "Euler": "#4d4d4d"
    }
    
    angles = np.linspace(170, 190, 200) # Sweep through 180
    
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "axes.facecolor": "white",
        "grid.color": "#E5E5E5",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    
    reps_panel_a = ["6D", "Quaternion", "Lie (fixed)"]
    reps_panel_b = ["Euler"]
    
    for rep, ckpt_path in checkpoint_map.items():
        if not os.path.exists(ckpt_path):
            continue
            
        try:
            model, _ = load_from_checkpoint(ckpt_path, device="cuda" if torch.cuda.is_available() else "cpu")
            if model is None: continue
            
            errors = []
            for a in angles:
                R_pred, R_gt = get_prediction(model, a)
                err = np.degrees(geodesic_distance(torch.from_numpy(R_pred).unsqueeze(0).float(), 
                                                   torch.from_numpy(R_gt).unsqueeze(0).float()).item())
                errors.append(err)
            
            ls = "--" if rep == "Euler" else "-"
            if rep in reps_panel_a:
                ax1.plot(angles, errors, label=rep, color=colors[rep], lw=2.5, ls=ls)
            else:
                ax2.plot(angles, errors, label=rep, color=colors[rep], lw=2.5, ls=ls)
                
        except Exception as e:
            print(f"Error processing {rep}: {e}")

    # Panel A: Topological Failure
    ax1.axvline(180, color="black", ls="--", alpha=0.3)
    ax1.set_ylim(-2, 185)
    ax1.set_xlim(170, 190)
    ax1.set_xlabel("Ground Truth Angle (degrees)", fontsize=11)
    ax1.set_ylabel("Geodesic Error (degrees)", fontsize=11)
    ax1.set_title("Topological Antipodal Failure (Z-axis)", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8.5)
    
    ax1.text(170.5, 10, "Quaternion failure is stochastic\nand axis-dependent.", 
             fontsize=11, color=colors["Quaternion"], fontweight='bold', 
             bbox=dict(facecolor='white', alpha=1.0, edgecolor='black', linewidth=1.0))

    # Panel B: Gimbal Singularity
    ax2.axvline(180, color="black", ls="--", alpha=0.3)
    ax2.axvline(171.8, color="#E67E22", ls=":", alpha=0.6)
    ax2.annotate("Gimbal Lock Region\n(Pitch \u2248 90\u00b0)", xy=(171.8, 160), xytext=(174, 170),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                 fontsize=10, fontweight='bold', color="#E67E22")
    
    ax2.set_ylim(-2, 185)
    ax2.set_xlim(170, 190)
    ax2.set_xlabel("Ground Truth Angle (degrees)", fontsize=11)
    ax2.set_title("Representation Failure (Gimbal Singularity)", fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8.5)

    plt.suptitle("Rotation Representation Failure Analysis (Worst-Case)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_path = "results/final_manuscript_figures/fig2_mechanistic_discontinuity.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Generated: {out_path}")

if __name__ == "__main__":
    main()
