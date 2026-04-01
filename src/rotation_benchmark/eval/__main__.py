from __future__ import annotations
import argparse
import sys
from rotation_benchmark.eval.benchmark import evaluate_model

def main():
    parser = argparse.ArgumentParser(description='RotationBench Evaluator CLI')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the PyTorch model weights')
    parser.add_argument('--rep', type=str, required=True, choices=['6D', 'Lie', 'Lie_FullFix', 'Atlas'], help='Representation type')
    parser.add_argument('--samples', type=int, default=100000, help='Number of test samples')
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.rep, num_samples=args.samples)

if __name__ == "__main__":
    main()
