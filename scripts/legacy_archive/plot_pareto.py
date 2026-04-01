import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from rotation_benchmark.eval.latency import measure_latency

# Results paths
PHASE1_SUMMARY = "results/phase1_summary.json"
GRASP_SUCCESS = "results/grasp_success_rates.json"

def main():
    if not os.path.exists(PHASE1_SUMMARY):
        print(f"Error: {PHASE1_SUMMARY} not found.")
        return

    with open(PHASE1_SUMMARY, "r") as f:
        phase1 = json.load(f)
    
    success_rates = {}
    if os.path.exists(GRASP_SUCCESS):
        with open(GRASP_SUCCESS, "r") as f:
            success_rates = json.load(f)

    reps = list(phase1.keys())
    latencies = {}
    reliability = {}
    grasp_success = {}

    for rep in reps:
        # Measure latency (CPU microbenchmark)
        # Use depth=3, num_pts=32 as per standard config
        try:
            latencies[rep] = measure_latency(rep)
        except Exception as e:
            print(f"Could not measure latency for {rep}: {e}")
            latencies[rep] = 1.0 # Fallback
        
        # Reliability = 1 - (p99_error / 180) or similar? 
        # Actually, let's use Reliability = 100 - FailureRate
        reliability[rep] = 100.0 - phase1[rep]["fail_avg"]
        
        # Grasp Success Rate
        grasp_success[rep] = success_rates.get(rep, 0.0) * 100.0

    # Plot
    plt.figure(figsize=(10, 6))
    
    # Pareto 1: Reliability vs Latency
    for rep in reps:
        plt.scatter(latencies[rep], reliability[rep], label=rep, s=100)
        plt.annotate(rep, (latencies[rep], reliability[rep]), xytext=(5, 5), textcoords='offset points')

    plt.xlabel("Inference Latency (ms, CPU)")
    plt.ylabel("Reliability (%)")
    plt.title("Pareto Front: Reliability vs. Latency")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("results/pareto_reliability.png")
    print("Saved results/pareto_reliability.png")

    # Pareto 2: Grasp Success vs Latency
    plt.figure(figsize=(10, 6))
    for rep in reps:
        plt.scatter(latencies[rep], grasp_success[rep], label=rep, s=100)
        plt.annotate(rep, (latencies[rep], grasp_success[rep]), xytext=(5, 5), textcoords='offset points')

    plt.xlabel("Inference Latency (ms, CPU)")
    plt.ylabel("Grasp Success Rate (%)")
    plt.title("Downstream Pareto: Grasp Success vs. Latency")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("results/pareto_grasp_success.png")
    print("Saved results/pareto_grasp_success.png")

    # Combine into a summary JSON
    final_metrics = {}
    for rep in reps:
        final_metrics[rep] = {
            "latency_ms": latencies[rep],
            "reliability": reliability[rep],
            "grasp_success_rate": grasp_success[rep],
            "p99_error": phase1[rep]["p99_avg"]
        }
    
    with open("results/final_pareto_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

if __name__ == "__main__":
    main()
