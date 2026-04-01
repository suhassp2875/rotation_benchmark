import subprocess
import os

VARIANTS = [
    ("A1_Baseline", "reps=lie_raw", "train.loss_type=geodesic"),
    ("A2_ArchOnly", "reps=lie_hardened", "train.loss_type=geodesic"),
    ("A3_LossOnly", "reps=lie_raw", "train.loss_type=antipodal"),
    ("A4_FullFix", "reps=lie_fullfix", "train.loss_type=antipodal")
]

def run_train(name, rep_cfg, loss_cfg):
    print(f"--- Ablation: {name} ---")
    cmd = [
        "python", "-m", "rotation_benchmark.train",
        rep_cfg,
        loss_cfg,
        "seed=1",
        "train.epochs=3"
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    subprocess.run(cmd, check=True, env=env)

if __name__ == "__main__":
    for name, rep, loss in VARIANTS:
        try:
            run_train(name, rep, loss)
        except Exception as e:
            print(f"Failed ablation {name}: {e}")
