import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from rotation_benchmark.models.factory import build_model
from rotation_benchmark.data.dataset import RotationPointCloudDataset
from torch.utils.data import DataLoader
from rotation_benchmark.geometry.so3 import log_map
from tqdm import tqdm
import math

def rotation_error(R1, R2):
    R_rel = torch.bmm(R1.transpose(-1, -2), R2)
    w = log_map(R_rel)
    return torch.linalg.norm(w, dim=-1) * (180.0 / math.pi)

def calculate_ece(errors, confidences, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    bin_accs = []
    bin_confs = []
    bin_sizes = []

    # Map errors to 'accuracy' (e.g. error < 5 deg = 1, else 0)
    # Or more generally: p(correct) = 1 - error/threshold
    threshold = 10.0 # deg
    accuracies = (errors < threshold).astype(float)

    for i in range(n_bins):
        bin_idx = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if np.any(bin_idx):
            bin_acc = np.mean(accuracies[bin_idx])
            bin_conf = np.mean(confidences[bin_idx])
            bin_size = np.sum(bin_idx)
            
            ece += np.abs(bin_acc - bin_conf) * (bin_size / len(errors))
            
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            bin_sizes.append(bin_size)
        else:
            bin_accs.append(0)
            bin_confs.append((bin_boundaries[i] + bin_boundaries[i+1])/2)
            bin_sizes.append(0)

    return ece, bin_accs, bin_confs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RotationPointCloudDataset(n=2000, num_points=32)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # We'll compare Atlas (Natural confidence) vs Lie (No confidence)
    reps = ["atlas", "lie_fullfix"]
    
    os.makedirs("results/calibration", exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    for idx, rep in enumerate(reps):
        # Load weights
        if rep == "atlas":
            checkpoint_path = "checkpoints/atlas_checkpoint.pt"
        elif rep == "lie_fullfix":
            checkpoint_path = "checkpoints/lie_fullfix_checkpoint.pt"
        else:
            checkpoint_path = f"checkpoints/{rep}_checkpoint.pth"

        if not os.path.exists(checkpoint_path):
            print(f"Skipping {rep}, checkpoint {checkpoint_path} not found.")
            continue
            
        model = build_model(rep=rep, in_dim=32*3, hidden_dim=256, depth=3, dropout=0.0)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        # Handle cases where model is wrapped in 'model' key
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        all_errors = []
        all_confidences = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Eval {rep}"):
                x, R_gt = batch["x"].to(device), batch["R_gt"].to(device)
                
                outputs = model(x)
                # Handle different output formats
                if isinstance(outputs, tuple) and len(outputs) == 3:
                    R_pred_stack, _, logits = outputs
                    # Confidence = max softmax
                    probs = torch.softmax(logits, dim=-1)
                    conf, best_idx = torch.max(probs, dim=-1)
                    B = x.shape[0]
                    R_pred = R_pred_stack[torch.arange(B), best_idx]
                elif isinstance(outputs, tuple):
                    R_pred, _ = outputs
                    conf = torch.ones(x.shape[0]) * 0.5
                else:
                    R_pred = outputs
                    conf = torch.ones(x.shape[0]) * 0.5
                
                err = rotation_error(R_pred, R_gt)
                all_errors.extend(err.cpu().numpy())
                all_confidences.extend(conf.cpu().numpy())
                
        errors = np.array(all_errors)
        confidences = np.array(all_confidences)
        
        ece, accs, confs = calculate_ece(errors, confidences)
        
        plt.subplot(1, 2, idx + 1)
        plt.bar(np.linspace(0.05, 0.95, 10), accs, width=0.1, alpha=0.5, label="Accuracy", color='blue')
        plt.plot([0, 1], [0, 1], '--', color='gray')
        plt.title(f"{rep}\nECE: {ece:.4f}")
        plt.xlabel("Confidence")
        plt.ylabel("Accuracy (<10°)")
        plt.legend()
        
    plt.tight_layout()
    plt.savefig("results/calibration_ece.png")
    print("Saved results/calibration_ece.png")

if __name__ == "__main__":
    main()
