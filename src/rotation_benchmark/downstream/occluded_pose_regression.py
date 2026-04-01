import torch
import numpy as np
import os
import sys
import math
import argparse

# Add root directory to sys.path to allow imports from models, dataset, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rotation_benchmark.data.dataset import RotationPointCloudDataset
from rotation_benchmark.models.factory import build_model
from rotation_benchmark.geometry.metrics import geodesic_distance

class OccludedPoseTask:
    """
    Simulates a robotic vision downstream task: Object Pose Regression under severe sensor occlusion.
    """
    def __init__(self, model_path, rep_type=None, device="cpu"):
        self.device = torch.device(device)
        self.model_path = model_path
        
        # Load checkpoint
        ckpt = torch.load(model_path, map_location="cpu")
        cfg = ckpt.get("cfg", {})
        
        rep = rep_type if rep_type else cfg.get("rep", cfg.get("model", {}).get("rep", "6D"))
        num_pts = cfg.get("num_points", cfg.get("data", {}).get("num_points", 32))
        hidden_dim = cfg.get("model", {}).get("hidden_dim", 256)
        depth = cfg.get("model", {}).get("depth", 3)
        
        in_dim = int(num_pts) * 3
        self.num_pts = int(num_pts)

        self.model = build_model(
            rep=rep,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=0.0
        ).to(self.device)

        state_dict = ckpt.get("model", ckpt)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

    def run_occluded_inference(self, occlusion_ratio=0.5, n_samples=5000):
        # Generate clean data
        ds = RotationPointCloudDataset(
            n=n_samples, 
            mode="uniform", 
            num_points=self.num_pts, 
            seed=42, 
            device=self.device,
            obs_noise_std=0.0
        )
        x_clean = ds.x.reshape(n_samples, self.num_pts, 3)
        R_gt = ds.R_gt
        
        # Apply structured occlusion (Drop N connected points)
        num_drop = int(self.num_pts * occlusion_ratio)
        x_occluded = x_clean.clone()
        # To simulate self-occlusion we drop the first N points (since canonical points are ordered sequentially somewhat).
        # Or simply select random points. Let's do random unstructured dropout for robustness.
        for i in range(n_samples):
            drop_idx = torch.randperm(self.num_pts)[:num_drop]
            x_occluded[i, drop_idx, :] = 0.0
            
        x_occluded = x_occluded.reshape(n_samples, -1)
        
        errs = []
        batch_size = 512
        for i in range(0, n_samples, batch_size):
            x_batch = x_occluded[i:i+batch_size]
            r_gt_batch = R_gt[i:i+batch_size]
            
            with torch.no_grad():
                outputs = self.model(x_batch)
                if isinstance(outputs, tuple):
                    R_pred = outputs[0]
                else:
                    R_pred = outputs
                    
            if R_pred.dim() == 4: # Atlas
                R_pred = R_pred[:, 0]
                
            e = geodesic_distance(R_pred, r_gt_batch) * (180.0 / math.pi)
            errs.append(e)
            
        return torch.cat(errs).cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="Occluded Pose Regression Benchmark")
    parser.add_argument("--model_path", required=True, help="Path to weights.pt")
    parser.add_argument("--rep", default=None, help="Representation type (optional, auto-detected from ckpt if possible)")
    parser.add_argument("--occlusion_rate", type=float, default=0.5, help="Fraction of points to drop (0.0 to 1.0)")
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Occluded Pose Task with {args.occlusion_rate*100}% occlusion...")
    
    task = OccludedPoseTask(args.model_path, rep_type=args.rep, device=device)
    errs = task.run_occluded_inference(occlusion_ratio=args.occlusion_rate)
    
    median_err = float(np.median(errs))
    p99_err = float(np.quantile(errs, 0.99))
    fail_rate = float(np.mean(errs > 30.0) * 100)
    
    print("\n--- Downstream Results ---")
    print(f"Median Error: {median_err:.2f}°")
    print(f"p99 Error:    {p99_err:.2f}°")
    print(f"Fail > 30°:   {fail_rate:.2f}%")

if __name__ == "__main__":
    main()
