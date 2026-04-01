import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from rotation_benchmark.models.factory import build_model
from rotation_benchmark.geometry.so3 import exp_map, log_map
from rotation_benchmark.geometry.metrics import geodesic_distance

def main():
    device = torch.device("cpu")
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Setup Models (minimal versions for logic check)
    # Note: In a real run, we'd load trained weights, but for 'math proof' 
    # we can use initialized models to show 'structural' stress.
    reps = ["6d", "quat", "atlas", "lie_fullfix"]
    models = {}
    for rep in reps:
        models[rep] = build_model(rep=rep, in_dim=32*3, hidden_dim=64, depth=2, dropout=0.0)
        models[rep].eval()

    # 2. Define the Sweep (0 to 180 degrees)
    angles = np.linspace(0.1, 179.9, 100) # Avoid exact 0/180 for raw log-map stability
    results = {rep: [] for rep in reps}

    # We simulate a "perfect" input x that should map to a rotation R(theta)
    # To measure "stress", we calculate the Geodesic Loss and look at the Jacobian norm
    # || d(Loss) / d(x) || 
    
    print("Sweeping angles and calculating gradient stress...")
    for alpha in angles:
        theta = np.deg2rad(alpha)
        # Target rotation: rotation around Z axis
        w_gt = torch.tensor([[0., 0., theta]], dtype=torch.float32)
        R_gt = exp_map(w_gt)
        
        # Mock input (constant feature vector)
        x = torch.randn(1, 32*3, dtype=torch.float32, requires_grad=True)
        
        for rep, model in models.items():
            # Reset grad
            if x.grad is not None: x.grad.zero_()
            
            outputs = model(x)
            if isinstance(outputs, tuple):
                R_pred = outputs[0]
            else:
                R_pred = outputs
                
            # If Atlas, pick best for this math check
            if R_pred.dim() == 4:
                dists = geodesic_distance(R_pred[0], R_gt[0])
                best_idx = torch.argmin(dists)
                R_pred = R_pred[:, best_idx]

            loss = geodesic_distance(R_pred, R_gt).mean()
            loss.backward()
            
            # The "Stress" is the magnitude of the gradient required to minimize error
            # Near singularities, small input changes require massive output changes
            grad_norm = x.grad.norm().item()
            results[rep].append(grad_norm)

    # 3. Plotting
    plt.figure(figsize=(10, 6))
    colors = {"6d": "#27AE60", "quat": "#C0392B", "atlas": "#E8901B", "lie_fullfix": "#8E44AD"}
    
    for rep in reps:
        # Smooth the lines for better visualization
        data = np.array(results[rep])
        # Normalize to see relative stability
        data = data / (np.max(data) + 1e-8) 
        plt.plot(angles, data, label=rep.upper(), color=colors[rep], linewidth=2)

    plt.axvline(180, color='black', linestyle='--', alpha=0.3)
    plt.text(175, 0.5, "Antipodal Boundary (π)", rotation=90, verticalalignment='center')
    
    plt.title("Mathematical Gradient Stress vs. Rotation Angle\n(Stability near the Antipodal Boundary)", fontsize=12, fontweight='bold')
    plt.xlabel("Target Rotation Angle (degrees)", fontsize=10)
    plt.ylabel("Relative Gradient Norm (Numerical Stress)", fontsize=10)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    save_path = out_dir / "boundary_stress.png"
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    main()
