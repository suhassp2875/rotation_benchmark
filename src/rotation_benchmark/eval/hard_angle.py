from __future__ import annotations
import torch
import numpy as np
import argparse
from rotation_benchmark.data.dataset import RotationPointCloudDataset as RotationDataset
from torch.utils.data import DataLoader
from rotation_benchmark.geometry.metrics import geodesic_distance
from rotation_benchmark.geometry.so3 import log_map
from rotation_benchmark.models.factory import build_model

def evaluate_hard_angles(model_path, rep_type, num_samples=100000, batch_size=512):
    print(f"Exposing Antipodal Failure (Hard Angles) for {rep_type}...")
    
    ckpt = torch.load(model_path, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    hidden_dim = cfg.get("model", {}).get("hidden_dim", 256)
    num_pts = cfg.get("num_points", cfg.get("data", {}).get("num_points", 32))
    
    model = build_model(rep=rep_type, in_dim=num_pts * 3, hidden_dim=hidden_dim, depth=cfg.get("model", {}).get("depth", 3), dropout=0.0)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = RotationDataset(n=num_samples, num_points=num_pts, device="cpu")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    results = [] # Store tuple of (gt_angle, error)
    with torch.no_grad():
        for batch in loader:
            x, rot_gt = batch["x"].to(device), batch["R_gt"].to(device)
            # Efficiently get gt angles
            gt_angle = torch.norm(log_map(rot_gt), dim=-1) * (180.0 / np.pi)
            
            outputs = model(x)
            rot_pred = outputs[0] if isinstance(outputs, tuple) else outputs
            if rot_pred.dim() == 4: rot_pred = rot_pred[:, 0]
            
            err = geodesic_distance(rot_pred, rot_gt) * (180.0 / np.pi)
            
            results.extend(zip(gt_angle.cpu().numpy(), err.cpu().numpy()))

    results = np.array(results)
    
    # Bin by GT angle
    bins = [0, 90, 150, 170, 180]
    print("\n--- Failure Anatomy (Binned by GT Angle) ---")
    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1]
        mask = (results[:, 0] >= low) & (results[:, 0] < high)
        if np.any(mask):
            bin_errs = results[mask, 1]
            print(f"Angle {low:3d}°-{high:3d}° | Median: {np.median(bin_errs):5.2f}° | Fail > 30°: {np.mean(bin_errs > 30.0)*100:5.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--rep', type=str, required=True)
    parser.add_argument('--samples', type=int, default=50000)
    args = parser.parse_args()
    evaluate_hard_angles(args.model_path, args.rep, num_samples=args.samples)
