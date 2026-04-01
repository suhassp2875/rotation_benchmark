import os
import sys
import json
from pathlib import Path

# Add src and scripts to path
sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("scripts"))

from robosuite_model_eval import MODELS_TO_RUN
from rotation_benchmark.eval.stratified_benchmark import run_stratified_eval

def main():
    final_output = {}
    
    for name, path in MODELS_TO_RUN.items():
        if name == "Oracle_GT" or path is None:
            continue
            
        print(f"Generating stratified summary for {name}...")
        # Reduce samples to 5000 for speed since 5000 points still gives highly accurate bounds
        res = run_stratified_eval(path, name, num_samples=5000, batch_size=512)
        final_output[name] = res

    os.makedirs("results", exist_ok=True)
    with open("results/stratified_summary.json", "w") as f:
        json.dump(final_output, f, indent=2)
    print("Saved results/stratified_summary.json")

if __name__ == "__main__":
    main()
