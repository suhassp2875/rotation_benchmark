import torch
import numpy as np
import argparse
from rotation_benchmark.data.dataset import RotationPointCloudDataset as RotationDataset
from torch.utils.data import DataLoader
from rotation_benchmark.geometry.metrics import geodesic_distance

def geodesic_error(R1, R2):
    return geodesic_distance(R1, R2) * (180.0 / np.pi)

def bootstrap_metric(data, metric_fn, num_bootstraps=1000, ci=0.95):
    boot_stats = []
    n = len(data)
    for _ in range(num_bootstraps):
        resample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(metric_fn(resample))
    boot_stats = np.sort(boot_stats)
    lower = boot_stats[int((1.0 - ci) / 2.0 * num_bootstraps)]
    upper = boot_stats[int((1.0 + ci) / 2.0 * num_bootstraps) - 1]
    return np.mean(boot_stats), lower, upper

from rotation_benchmark.models.factory import build_model

def evaluate_model(model_path, rep_type, num_samples=100000, batch_size=512, bootstrap=True):
    print(f"Loading model from {model_path} (Type: {rep_type})")
    
    ckpt = torch.load(model_path, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    
    hidden_dim = cfg.get("model", {}).get("hidden_dim", 256)
    depth = cfg.get("model", {}).get("depth", 3)
    num_pts = cfg.get("num_points", cfg.get("data", {}).get("num_points", 32))
    dropout = cfg.get("model", {}).get("dropout", 0.0)

    model = build_model(
        rep=rep_type,
        in_dim=num_pts * 3,
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
    )
    
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = RotationDataset(n=num_samples, num_points=num_pts, device="cpu")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_errors = []
    
    print(f"Evaluating on {num_samples} samples...")
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            rot_gt = batch["R_gt"].to(device)
            outputs = model(x)
            
            if isinstance(outputs, tuple):
                rot_pred = outputs[0]
            else:
                rot_pred = outputs
            
            if rot_pred.dim() == 4: # (B, K, 3, 3) - Multi-chart
                # For benchmark, if there are multiple charts, we use the one closest to GT 
                # to measure the 'best-case' capacity, similar to Chamfer-style eval.
                from rotation_benchmark.geometry.metrics import geodesic_distance
                dists = []
                for k in range(rot_pred.shape[1]):
                    dists.append(geodesic_distance(rot_pred[:, k], rot_gt))
                dists = torch.stack(dists, dim=1) # [B, K]
                best_k = torch.argmin(dists, dim=1)
                rot_pred = rot_pred[torch.arange(rot_pred.size(0)), best_k]

            err = geodesic_error(rot_pred, rot_gt)
            all_errors.extend(err.cpu().numpy())

    all_errors = np.array(all_errors)
    
    if bootstrap:
        median_m, median_l, median_u = bootstrap_metric(all_errors, np.median)
        p99_m, p99_l, p99_u = bootstrap_metric(all_errors, lambda x: np.percentile(x, 99))
        fail_m, fail_l, fail_u = bootstrap_metric(all_errors, lambda x: np.mean(x > 30.0) * 100)
        
        print("--- Benchmark Results (with 95% CI) ---")
        print(f"Representation: {rep_type}")
        print(f"Median Error:   {median_m:.3f}° [{median_l:.3f}, {median_u:.3f}]")
        print(f"p99 Error:      {p99_m:.2f}° [{p99_l:.2f}, {p99_u:.2f}]")
        print(f"Fail > 30°:     {fail_m:.3f}% [{fail_l:.3f}, {fail_u:.3f}]")
    else:
        median_err = np.median(all_errors)
        p99_err = np.percentile(all_errors, 99)
        fail_rate = np.mean(all_errors > 30.0) * 100
        print("--- Benchmark Results ---")
        print(f"Representation: {rep_type}")
        print(f"Median Error:   {median_err:.2f}°")
        print(f"p99 Error:      {p99_err:.2f}°")
        print(f"Fail > 30°:     {fail_rate:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RotationBench Evaluator')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the PyTorch model weights (.pt)')
    parser.add_argument('--rep', type=str, required=True, help='Representation type (6D, Atlas, Lie, etc.)')
    parser.add_argument('--samples', type=int, default=100000, help='Number of synthetic test samples')
    parser.add_argument('--no_bootstrap', action='store_true', help='Disable bootstrap CI calculation')
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.rep, num_samples=args.samples, bootstrap=not args.no_bootstrap)
