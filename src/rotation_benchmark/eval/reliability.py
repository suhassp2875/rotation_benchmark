from __future__ import annotations
import torch
import numpy as np
import argparse
from rotation_benchmark.data.dataset import RotationPointCloudDataset as RotationDataset
from torch.utils.data import DataLoader
from rotation_benchmark.geometry.metrics import geodesic_distance
from rotation_benchmark.models.factory import build_model

def evaluate_reliability(model_path, rep_type, num_samples=100000, batch_size=512):
    print(f"Evaluating Reliability AUC/Tail-Risk for {rep_type}...")
    
    ckpt = torch.load(model_path, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    hidden_dim = cfg.get("model", {}).get("hidden_dim", 256)
    depth = cfg.get("model", {}).get("depth", 3)
    num_pts = cfg.get("num_points", cfg.get("data", {}).get("num_points", 32))
    
    model = build_model(rep=rep_type, in_dim=num_pts * 3, hidden_dim=hidden_dim, depth=depth, dropout=0.0)
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = RotationDataset(n=num_samples, num_points=num_pts, device="cpu")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    errors = []
    with torch.no_grad():
        for batch in loader:
            x, rot_gt = batch["x"].to(device), batch["R_gt"].to(device)
            outputs = model(x)
            rot_pred = outputs[0] if isinstance(outputs, tuple) else outputs
            if rot_pred.dim() == 4: rot_pred = rot_pred[:, 0]
            
            err = geodesic_distance(rot_pred, rot_gt) * (180.0 / np.pi)
            errors.extend(err.cpu().numpy())

    errors = np.array(errors)
    
    # Calculate Reliability Metrics
    # Reliability AUC = Area under error-sorted curve or similar
    # Here we use a simpler metric: % of samples below certain thresholds
    thresholds = [1.0, 2.0, 5.0, 10.0, 30.0]
    accuracy_at = {t: np.mean(errors <= t) * 100 for t in thresholds}
    
    print("\n--- Reliability Diagnostics ---")
    for t, acc in accuracy_at.items():
        print(f"Accuracy @ {t:>2}°: {acc:.2f}%")
    
    # Worst-case: Max error
    print(f"Max Tail Error: {np.max(errors):.2f}°")
    
    return accuracy_at

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--rep', type=str, required=True)
    parser.add_argument('--samples', type=int, default=50000)
    args = parser.parse_args()
    evaluate_reliability(args.model_path, args.rep, num_samples=args.samples)
