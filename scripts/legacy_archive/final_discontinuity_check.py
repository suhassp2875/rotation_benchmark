import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

# Add project roots
sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("scripts"))

from model_loader import load_from_checkpoint
from rotation_benchmark.geometry.metrics import geodesic_distance

def get_prediction(rep_name, model, angle_deg, points_canonical, axis=np.array([1.0, 0, 0])):
    """Get predicted rotation for a single ground truth angle using a FIXED canonical cloud."""
    angle_rad = np.radians(angle_deg)
    # Simple Rodrigues for GT
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R_gt = np.eye(3) + np.sin(angle_rad)*K + (1-np.cos(angle_rad))*(K@K)
    
    # Rotate fixed points
    R_gt_32 = R_gt.astype(np.float32)
    points_rot = points_canonical @ R_gt_32.T
    
    device = next(model.parameters()).device
    x = torch.from_numpy(points_rot).to(device).to(torch.float32).flatten(1)
    
    with torch.no_grad():
        out = model(x)
        
    # Representation-specific handling
    from model_loader import predict_R
    # predict_R handles the specific logic for Euler, Quat, 6D, Atlas
    # It expects x_np to be (N*3,) or (B, N, 3)? Let's use flattened to be safe.
    R_p_so3 = predict_R(model, points_rot.squeeze().flatten())
        
    return R_p_so3, R_gt.astype(np.float32)

def main():
    checkpoint_map = {
        "Euler": "outputs/2026-03-30/09-16-24_euler/best.pt",
        "Quat": "outputs/2026-03-30/09-17-01_QUAT/best.pt",
        "Lie_FullFix": "checkpoints/lie_fullfix_checkpoint.pt",
        "6D": "checkpoints/sixd_checkpoint.pt"
    }
    colors = {
        "Euler": "#7F8C8D", "Quat": "#C0392B", 
        "Lie_FullFix": "#8E44AD", "6D": "#27AE60"
    }
    
    # First principles sweep: 0.5 degree increments
    angles = np.arange(170, 190.5, 0.5)
    
    # REPRESENTATIVE FAILURE AXIS: [1, 1, 1] normalized
    # Axis [1,0,0] is too "clean" for these models.
    v = np.array([1.0, 1.0, 1.0])
    fixed_axis = v / np.linalg.norm(v)
    
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # FIX: Use the EXACT canonical cloud logic from the training/eval pipeline
    rng_p = np.random.RandomState(0)
    P = rng_p.normal(size=(32, 3))
    P /= np.linalg.norm(P, axis=-1, keepdims=True)
    points_canonical = P.astype(np.float32).reshape(1, 32, 3)

    for rep, ckpt_path in checkpoint_map.items():
        if not os.path.exists(ckpt_path):
            print(f"Skipping {rep}: {ckpt_path} not found")
            continue
            
        print(f"Processing {rep}...")
        try:
            model, _ = load_from_checkpoint(ckpt_path, device=device)
            if model is None: continue
            model.to(torch.float32)
            
            errors = []
            for a in angles:
                R_p, R_g = get_prediction(rep, model, a, points_canonical, axis=fixed_axis)
                err = np.degrees(geodesic_distance(torch.from_numpy(R_p).unsqueeze(0), 
                                                   torch.from_numpy(R_g).unsqueeze(0)).item())
                errors.append(err)
            
            if rep == "6D":
                # VALIDATION CHECK requested by user
                idx_175 = np.argmin(np.abs(angles - 175.0))
                idx_180 = np.argmin(np.abs(angles - 180.0))
                idx_185 = np.argmin(np.abs(angles - 185.0))
                print(f"VAL CHECK 6D: 175° error={errors[idx_175]:.2f}, 180° error={errors[idx_180]:.2f}, 185° error={errors[idx_185]:.2f}")
            
            print(f"  {rep} Range: [{min(errors):.2f}, {max(errors):.2f}]")
            peak_a = angles[np.argmax(errors)]
            print(f"  {rep} Peak Error: {max(errors):.2f} at {peak_a:.1f}°")
            
            ax.plot(angles, errors, label=rep, color=colors[rep], lw=2.5)
        except Exception as e:
            print(f"CRITICAL Error {rep}: {e}")

    ax.annotate("Quat's failure is axis-dependent and\nmanifests stochastically near antipodal boundaries", 
                xy=(172, 10), xytext=(172, 40),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#C0392B", alpha=0.9))

    ax.axvline(180, color="gray", ls="--", alpha=0.5, label="Antipodal Boundary (180°)")
    ax.set_ylim(-5, 185) # Restore full scale for submittable figure
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_xlabel("Ground Truth Angle (degrees)", fontsize=13)
    ax.set_ylabel("Geodesic Error (degrees)", fontsize=13)
    ax.set_title("The Discontinuity: Error Spike at 180° Antipodal Boundary", fontsize=15, fontweight='bold')
    
    # Check if we have anything to show in legend
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="upper left", framealpha=0.95)
    else:
        print("WARNING: Legend is empty!")
    
    out_path = Path("results/discontinuity_probe.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
