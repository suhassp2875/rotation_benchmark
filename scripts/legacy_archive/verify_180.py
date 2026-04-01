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

def main():
    checkpoint_map = {
        "Euler": "outputs/2026-03-30/09-16-24_euler/best.pt",
        "Lie_FullFix": "checkpoints/lie_fullfix_checkpoint.pt"
    }
    
    P = get_P()
    axis = np.array([1., 0, 0])
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    
    for rep, ckpt_path in checkpoint_map.items():
        if not os.path.exists(ckpt_path): continue
        print(f"\n--- Verifying {rep} ---")
        model, _ = load_from_checkpoint(ckpt_path)
        
        for a in [177.5, 179.0, 179.9, 180.0, 180.1, 181.0]:
            rad = np.radians(a)
            R_gt = np.eye(3) + np.sin(rad)*K + (1-np.cos(rad))*(K@K)
            R_pred = predict_R(model, P @ R_gt.T)
            
            err = np.degrees(geodesic_distance(torch.from_numpy(R_pred).unsqueeze(0).float(), 
                                               torch.from_numpy(R_gt).unsqueeze(0).float()).item())
            print(f"Angle: {a:6.2f} | Trace(R_gt): {np.trace(R_gt):7.4f} | Error: {err:7.2f}")

if __name__ == "__main__":
    main()
