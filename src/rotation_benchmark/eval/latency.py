from __future__ import annotations
import torch
import time
import argparse
import numpy as np
from rotation_benchmark.models.factory import build_model

def measure_latency(rep_type, num_pts=32, hidden_dim=256, depth=3, trials=1000):
    print(f"Measuring CPU Microbenchmarks for {rep_type} (depth={depth})...")
    
    device = torch.device("cpu")
    model = build_model(rep=rep_type, in_dim=num_pts * 3, hidden_dim=hidden_dim, depth=depth, dropout=0.0)
    model.to(device)
    model.eval()

    # Create dummy input
    x = torch.randn(1, num_pts * 3).to(device)
    
    # Warmup
    for _ in range(10):
        _ = model(x)
    
    times = []
    with torch.no_grad():
        for _ in range(trials):
            t0 = time.time()
            _ = model(x)
            times.append((time.time() - t0) * 1000) # ms

    mean_ms = np.mean(times)
    std_ms = np.std(times)
    p99_ms = np.percentile(times, 99)

    print(f"\n--- Latency Results ({rep_type}) ---")
    print(f"Mean Latency: {mean_ms:.3f} ms")
    print(f"Std Dev:      {std_ms:.3f} ms")
    print(f"p99 Latency:  {p99_ms:.3f} ms")

    return mean_ms

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rep', type=str, required=True)
    parser.add_argument('--pts', type=int, default=32)
    parser.add_argument('--trials', type=int, default=1000)
    args = parser.parse_args()
    measure_latency(args.rep, num_pts=args.pts, trials=args.trials)
