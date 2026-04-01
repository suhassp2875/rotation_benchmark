#!/bin/bash
# Reproduce Table 1: Main SO(3) Benchmark

REPS=("6d" "lie_raw" "lie_fullfix" "atlas")

echo "Starting Main Benchmark Reproduction..."

for rep in "${REPS[@]}"; do
    echo "Evaluating $rep..."
    python -m rotation_benchmark.eval.benchmark --model_path checkpoints/${rep}_checkpoint.pt --rep $rep
done
