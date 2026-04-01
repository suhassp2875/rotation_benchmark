import subprocess
import os

REPS = ["6d", "lie_fullfix", "atlas", "svd", "quat", "5d"]
SEEDS = [1, 2, 3, 4, 5] 

def run_train(rep, seed):
    print(f"--- Training {rep} with seed {seed} ---")
    cmd = [
        "python", "-m", "rotation_benchmark.train",
        f"reps={rep}",
        f"seed={seed}",
        "train.epochs=3" # Keep it short for now as in existing manifests
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    for rep in REPS:
        for seed in SEEDS:
            try:
                run_train(rep, seed)
            except Exception as e:
                print(f"Failed to train {rep} with seed {seed}: {e}")
