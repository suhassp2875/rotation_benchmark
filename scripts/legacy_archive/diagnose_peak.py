import torch
import numpy as np
import os, sys
from pathlib import Path

# Add project roots
sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("scripts"))

from model_loader import load_from_checkpoint
from rotation_benchmark.geometry.metrics import geodesic_distance

def get_prediction_diagnose(model, angle_deg, axis=np.array([0., 0, 1])):
    angle_rad = np.radians(angle_deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R_gt = np.eye(3) + np.sin(angle_rad)*K + (1-np.cos(angle_rad))*(K@K)
    
    # Use fixed points for diagnostic consistency
    np.random.seed(42)
    points = np.random.randn(1, 32, 3).astype(np.float32)
    points_rot = points @ R_gt.T
    
    device = next(model.parameters()).device
    x = torch.from_numpy(points_rot).to(device).float().flatten(1)
    
    with torch.no_grad():
        out = model(x)
        R_pred = out[0] if isinstance(out, tuple) else out
            
    R_np = R_pred.squeeze().cpu().numpy()
    if R_np.shape != (3, 3): R_np = R_np.reshape(3, 3)
    U, _, Vt = np.linalg.svd(R_np)
    R_pred_so3 = U @ Vt
    if np.linalg.det(R_pred_so3) < 0:
        U[:, -1] *= -1
        R_pred_so3 = U @ Vt
        
    return R_pred_so3, R_gt

def main():
    checkpoint_map = {
        "Euler": "outputs/2026-03-30/09-16-24_euler/best.pt",
        "Lie_FullFix": "checkpoints/lie_fullfix_checkpoint.pt"
    }
    
    angles = np.linspace(170, 190, 41) # 0.5 degree steps
    
    for rep, ckpt_path in checkpoint_map.items():
        if not os.path.exists(ckpt_path):
            print(f"Skipping {rep}: {ckpt_path} not found")
            continue
            
        print(f"\n--- Diagnosing {rep} ---")
        model, _ = load_from_checkpoint(ckpt_path, device="cpu")
        if model is None: continue
        
        peak_angle = -1
        max_err = -1
        
        for a in angles:
            R_pred, R_gt = get_prediction_diagnose(model, a)
            tr = np.trace(R_gt)
            err = np.degrees(geodesic_distance(torch.from_numpy(R_pred).unsqueeze(0).float(), 
                                               torch.from_numpy(R_gt).unsqueeze(0).float()).item())
            
            if abs(a - 180) < 0.1 or abs(a - 177.5) < 0.1:
                print(f"Angle: {a:6.2f} | Trace(R_gt): {tr:7.4f} | Error: {err:7.2f}")
            
            if err > max_err:
                max_err = err
                peak_angle = a
        
        print(f"PEAK FOUND AT: {peak_angle:.2f} degrees (Max Error: {max_err:.2f})")

if __name__ == "__main__":
    main()
