from __future__ import annotations
import os
import json
import time
import argparse

# Monkeypatch for Python 3.14 compatibility with Hydra 1.3.2
# ArgumentParser checks for '%' in help string, which fails on LazyCompletionHelp
if hasattr(argparse.ArgumentParser, "add_argument"):
    _orig_add_argument = argparse.ArgumentParser.add_argument
    def _patched_add_argument(self, *args, **kwargs):
        if "help" in kwargs:
             # Force help to string to avoid TypeError in argparse._check_help
             if not isinstance(kwargs["help"], str):
                 kwargs["help"] = str(kwargs["help"])
        return _orig_add_argument(self, *args, **kwargs)
    argparse.ArgumentParser.add_argument = _patched_add_argument

import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from rotation_benchmark.data.dataset import RotationPointCloudDataset
from rotation_benchmark.models.factory import build_model
from rotation_benchmark.losses.geodesic import GeodesicLoss
from rotation_benchmark.utils.repro import seed_everything
from rotation_benchmark.utils.experiment import config_hash
from rotation_benchmark.geometry.metrics import grad_norm

def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)

@hydra.main(config_path="../../configs", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    print(f"DEBUG: Current Working Directory is {os.getcwd()}")
    seed_everything(int(cfg.seed), deterministic=True)
    device = resolve_device(str(cfg.device))
    dtype = torch.float32

    # Data
    ds_train = RotationPointCloudDataset(
        n=int(cfg.splits.n_train),
        mode="uniform",
        num_points=int(cfg.data.num_points),
        obs_noise_std=float(cfg.data.obs_noise_std),
        rot_noise_std=float(cfg.data.rot_noise_std),
        seed=int(cfg.seed) + 0,
        device="cpu",
        dtype=dtype,
    )
    ds_val = RotationPointCloudDataset(
        n=int(cfg.splits.n_val),
        mode="uniform",
        num_points=int(cfg.data.num_points),
        obs_noise_std=float(cfg.data.obs_noise_std),
        rot_noise_std=float(cfg.data.rot_noise_std),
        seed=int(cfg.seed) + 1,
        device="cpu",
        dtype=dtype,
    )

    in_dim = ds_train.x.shape[-1]
    dl_train = DataLoader(ds_train, batch_size=int(cfg.train.batch_size), shuffle=True, num_workers=0, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=int(cfg.train.batch_size), shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model = build_model(
        rep=str(cfg.reps.rep),
        in_dim=in_dim,
        hidden_dim=int(cfg.model.hidden_dim),
        depth=int(cfg.model.depth),
        dropout=float(cfg.model.dropout),
        angle_range=float(cfg.head.euler_angle_range),
        euler_use_tanh=bool(cfg.head.euler_use_tanh),
        lie_max_angle=float(cfg.head.lie_max_angle),
        lie_use_tanh=bool(cfg.head.lie_use_tanh),
    ).to(device)

    loss_fn = GeodesicLoss(reduction="mean")
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))

    # run_dir = os.getcwd()  # Hydra output dir (Broken in Hydra 1.2+ if chdir=False)
    from hydra.core.hydra_config import HydraConfig
    run_dir = HydraConfig.get().runtime.output_dir
    print(f"DEBUG: Saving outputs to {run_dir}")
    exp_id = config_hash(cfg)
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "exp_id": exp_id,
                "rep": str(cfg.reps.rep),
                "device": str(device),
                "torch_version": torch.__version__,
                "config": OmegaConf.to_container(cfg, resolve=True),
            },
            f,
            indent=2,
        )

    metrics_path = os.path.join(run_dir, "train_metrics.jsonl")
    best_val = float("inf")
    best_path = os.path.join(run_dir, "best.pt")

    def run_epoch(dl, train: bool):
        model.train(train)
        total_loss = 0.0
        total_gn = 0.0
        steps = 0

        pbar = tqdm(dl, desc=("train" if train else "val"), leave=False)
        for it, batch in enumerate(pbar):
            x = batch["x"].to(device, non_blocking=True)
            R_gt = batch["R_gt"].to(device, non_blocking=True)

            if train:
                opt.zero_grad(set_to_none=True)

            outputs = model(x)
            # Extract outputs based on model rep
            if isinstance(outputs, tuple) and len(outputs) == 3:
                # Atlas returns R_stack, w_stack, logits
                R_pred, w_pred, logits = outputs
            elif isinstance(outputs, tuple):
                R_pred, w_pred = outputs
            else:
                R_pred = outputs
                w_pred = None
            
            # Loss calculation
            loss_type = str(cfg.train.loss_type).lower()
            if loss_type == "antipodal":
                from rotation_benchmark.losses.antipodal import AntipodalLieLoss
                if not globals().get("_antipodal_loss_obj"):
                    globals()["_antipodal_loss_obj"] = AntipodalLieLoss().to(device)
                
                # Curriculum: Ramp axis loss from 0 to target over 10 epochs
                curriculum_ramp = min(1.0, epoch / 10.0) if train else 1.0
                globals()["_antipodal_loss_obj"].lambda_axis = float(cfg.train.get("lambda_axis", 0.1)) * curriculum_ramp
                
                loss = globals()["_antipodal_loss_obj"](R_pred, R_gt, w_pred)
            elif loss_type == "atlas":
                from rotation_benchmark.losses.atlas_loss import AtlasLoss
                if not globals().get("_atlas_loss_obj"):
                    globals()["_atlas_loss_obj"] = AtlasLoss().to(device)
                loss_reg, best_indices = globals()["_atlas_loss_obj"](R_pred, R_gt)
                loss_cls = torch.nn.functional.cross_entropy(logits, best_indices)
                loss = loss_reg + 5.0 * loss_cls if train else loss_reg + loss_cls
            else:
                # If R_pred comes from Atlas we need to get best_R for standard eval
                if isinstance(R_pred, torch.Tensor) and R_pred.dim() == 4:
                    # R_pred is [B, K, 3, 3], meaning we ran an atlas model using standard config
                    # Default: just take the nearest
                    from rotation_benchmark.geometry.metrics import geodesic_distance
                    with torch.no_grad():
                        dists = []
                        for k in range(R_pred.shape[1]):
                            dists.append(geodesic_distance(R_pred[:, k], R_gt))
                        dists = torch.stack(dists, dim=1) # [B, K]
                        best_k = torch.argmin(dists, dim=1)
                    best_R_pred = R_pred[torch.arange(R_pred.size(0)), best_k]
                    loss = loss_fn(best_R_pred, R_gt)
                else:
                    loss = loss_fn(R_pred, R_gt)

            if train:
                loss.backward()
                if float(cfg.train.grad_clip) and float(cfg.train.grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
                opt.step()

            gn = grad_norm(model.parameters())
            total_loss += float(loss.item())
            total_gn += float(gn)
            steps += 1

            if (it + 1) % int(cfg.train.log_every) == 0:
                pbar.set_postfix(loss=total_loss / steps, grad=total_gn / steps)

        return {"loss": total_loss / max(1, steps), "grad_norm": total_gn / max(1, steps)}

    for epoch in range(1, int(cfg.train.epochs) + 1):
        t0 = time.time()
        train_m = run_epoch(dl_train, train=True)
        val_m = run_epoch(dl_val, train=False)
        dt = time.time() - t0

        row = {
            "epoch": epoch,
            "time_sec": dt,
            "train_loss": train_m["loss"],
            "train_grad_norm": train_m["grad_norm"],
            "val_loss": val_m["loss"],
            "val_grad_norm": val_m["grad_norm"],
        }
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        print(
            f"Epoch {epoch:03d} | train {row['train_loss']:.6f} | val {row['val_loss']:.6f} "
            f"| grad {row['train_grad_norm']:.4f} | {dt:.1f}s"
        )

        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            torch.save({"model": model.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True)}, best_path)

    torch.save({"model": model.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True)}, os.path.join(run_dir, "last.pt"))

    # Auto-eval
    # from eval import evaluate_checkpoint
    # results = evaluate_checkpoint(best_path)
    # with open(os.path.join(run_dir, "eval.json"), "w", encoding="utf-8") as f:
    #     json.dump(results, f, indent=2)
    # print("Eval(best):", results)

if __name__ == "__main__":
    main()
