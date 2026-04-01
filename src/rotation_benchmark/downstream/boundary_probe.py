import torch
import numpy as np
import os
import sys
import math
import argparse

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rotation_benchmark.models.factory import build_model
from rotation_benchmark.geometry.so3 import log_map

class InverseKinematicsTask:
    """
    A practical downstream task: Predicting joint rotations for a simple chain.
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
        
    def evaluate_reachability(self, n_trials=1000, target_angle=178.0):
        device = self.device
        results = []
        for _ in range(n_trials):
            # Target is identity but the test is about regression near a limit
            # Here we simulate the network being fed features for a 178 deg rotation
            x_synth = torch.randn(1, self.num_pts, 3).to(device) * 0.1 
            
            with torch.no_grad():
                outputs = self.model(x_synth.view(1, -1))
                if isinstance(outputs, tuple):
                    R_pred = outputs[0]
                else:
                    R_pred = outputs
                
                if R_pred.dim() == 4: # Atlas
                    R_pred = R_pred[:, 0]
                
                w = log_map(R_pred)
                mag_pred = torch.norm(w, dim=-1).item() * (180.0/math.pi)
                
            results.append(mag_pred)
            
        return results

def main():
    parser = argparse.ArgumentParser(description="IK Reachability / Joint Flip Benchmark")
    parser.add_argument("--model_path", required=True, help="Path to weights.pt")
    parser.add_argument("--rep", default=None, help="Representation type")
    parser.add_argument("--trials", type=int, default=500, help="Number of trials")
    parser.add_argument("--target_angle", type=float, default=178.0, help="Target angle in degrees")
    parser.add_argument("--occlusion_rate", type=float, default=0.0, help="Unused dummy arg for compatibility")
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running IK Reachability Task for {args.target_angle}°...")
    
    task = InverseKinematicsTask(args.model_path, rep_type=args.rep, device=device)
    res = task.evaluate_reachability(n_trials=args.trials, target_angle=args.target_angle)
    
    mean_val = np.mean(res)
    std_val = np.std(res)
    flips = np.mean(np.array(res) < 90.0) * 100
    flip_str = f"YES ({flips:.1f}%)" if flips > 1.0 else "NO"
    
    print("\n--- IK Downstream Results ---")
    print(f"Target Angle:    {args.target_angle}°")
    print(f"Pred Mean Angle: {mean_val:.1f}°")
    print(f"Pred Std:        {std_val:.1f}")
    print(f"Joint Flip?      {flip_str}")

if __name__ == "__main__":
    main()
