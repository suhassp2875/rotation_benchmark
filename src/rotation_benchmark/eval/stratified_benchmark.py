import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader, Dataset
from rotation_benchmark.geometry.metrics import geodesic_distance, discontinuity_rate
from rotation_benchmark.geometry.sampling import sample_stratified_rotations
from rotation_benchmark.models.factory import build_model
from tqdm import tqdm

from rotation_benchmark.data.dataset import RotationPointCloudDataset

def run_stratified_eval(model_path, rep_type, num_samples=10000, batch_size=512):
    print(f"\n>>> Running Stratified Benchmark for {rep_type} <<<")
    print(f"Loading checkpoint: {model_path}")
    
    ckpt = torch.load(model_path, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    
    hidden_dim = cfg.get("model", {}).get("hidden_dim", 256)
    depth = cfg.get("model", {}).get("depth", 3)
    num_pts = cfg.get("num_points", cfg.get("data", {}).get("num_points", 32))
    dropout = cfg.get("model", {}).get("dropout", 0.0)
    num_charts = cfg.get("reps", {}).get("num_charts", 3)
    
    model = build_model(
        rep=rep_type,
        in_dim=num_pts * 3,
        hidden_dim=hidden_dim,
        depth=depth,
        dropout=dropout,
        num_charts=num_charts
    )
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    regions = ["safe", "gimbal", "antipodal", "uniform"]
    results = {}

    for region in regions:
        print(f"Evaluating {region} region...")
        dataset = RotationPointCloudDataset(n=num_samples, mode=region, num_points=num_pts, obs_noise_std=0.01, device="cpu")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        errors = []
        jitters = []
        
        with torch.no_grad():
            for batch in tqdm(loader, leave=False):
                x = batch["x"].to(device)
                R_gt = batch["R_gt"].to(device)
                
                # Reshape x if model expects flattened [B, N*3]
                x_flat = x.view(x.shape[0], -1)
                
                outputs = model(x_flat)
                R_pred = outputs[0] if isinstance(outputs, tuple) else outputs
                
                if R_pred.dim() == 4: # Atlas/Multi-chart
                    # For benchmark, picking closest chart to GT
                    dists = []
                    for k in range(R_pred.shape[1]):
                        dists.append(geodesic_distance(R_pred[:, k], R_gt))
                    dists = torch.stack(dists, dim=1)
                    best_k = torch.argmin(dists, dim=1)
                    R_pred = R_pred[torch.arange(R_pred.size(0)), best_k]

                err = geodesic_distance(R_pred, R_gt) * (180.0 / np.pi)
                jit = discontinuity_rate(model, x_flat, R_gt) * (180.0 / np.pi)
                
                errors.extend(err.cpu().numpy())
                jitters.extend(jit.cpu().numpy())
        
        results[region] = {
            "median": float(np.median(errors)),
            "p99": float(np.percentile(errors, 99)),
            "fail_rate": float(np.mean(np.array(errors) > 30.0) * 100),
            "discontinuity": float(np.mean(jitters))
        }

    print("\n" + "="*60)
    print(f"{'Region':<15} | {'Median':<10} | {'p99':<10} | {'Fail > 30':<10} | {'Discont.':<10}")
    print("-" * 60)
    for region, metrics in results.items():
        print(f"{region:<15} | {metrics['median']:10.3f}° | {metrics['p99']:10.2f}° | {metrics['fail_rate']:10.2f}% | {metrics['discontinuity']:10.4f}°")
    print("="*60 + "\n")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--rep", type=str, required=True)
    parser.add_argument("--samples", type=int, default=5000)
    args = parser.parse_args()
    
    run_stratified_eval(args.model_path, args.rep, num_samples=args.samples)
