import torch
import numpy as np
import os, sys
from pathlib import Path

# Add project roots
sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("scripts"))

from model_loader import load_from_checkpoint, predict_R
from rotation_benchmark.geometry.metrics import geodesic_distance

def get_P():
    rng = np.random.RandomState(0)
    pts = rng.normal(size=(32, 3))
    pts = pts / np.linalg.norm(pts, axis=-1, keepdims=True)
    return pts.astype(np.float32)

def diagnose_axis(model, axis_name, axis_vec, angles):
    P = get_P()
    K = np.array([[0, -axis_vec[2], axis_vec[1]], [axis_vec[2], 0, -axis_vec[0]], [-axis_vec[1], axis_vec[0], 0]])
    
    max_err = -1
    peak_a = -1
    for a in angles:
        rad = np.radians(a)
        R_gt = np.eye(3) + np.sin(rad)*K + (1-np.cos(rad))*(K@K)
        R_pred = predict_R(model, P @ R_gt.T)
        err = np.degrees(geodesic_distance(torch.from_numpy(R_pred).unsqueeze(0).float(), 
                                           torch.from_numpy(R_gt).unsqueeze(0).float()).item())
        if err > max_err:
            max_err = err
            peak_a = a
    return peak_a, max_err

def main():
    checkpoint_map = {
        "Euler": "outputs/2026-03-30/09-16-24_euler/best.pt",
        "Lie_FullFix": "checkpoints/lie_fullfix_checkpoint.pt"
    }
    
    axes = {
        "X": [1.0, 0, 0],
        "Y": [0, 1.0, 0],
        "Z": [0, 0, 1.0],
        "Diagonal": [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]
    }
    
    angles_180 = np.linspace(170, 190, 41)
    angles_90 = np.linspace(80, 100, 41)
    
    for rep, ckpt_path in checkpoint_map.items():
        if not os.path.exists(ckpt_path): continue
        print(f"\n--- Checking {rep} ---")
        model, _ = load_from_checkpoint(ckpt_path)
        
        for name, vec in axes.items():
            # Check near 180
            peak_180, err_180 = diagnose_axis(model, name, vec, angles_180)
            # Check near 90
            peak_90, err_90 = diagnose_axis(model, name, vec, angles_90)
            
            print(f"Axis {name:8} | Peak @ 180-region: {peak_180:6.2f} (Err: {err_180:7.2f}) | Peak @ 90-region: {peak_90:6.2f} (Err: {err_90:7.2f})")

if __name__ == "__main__":
    main()
