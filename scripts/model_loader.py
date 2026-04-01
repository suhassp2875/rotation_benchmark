"""
model_loader.py  —  Shared utility to load any rotation model from a checkpoint.
Handles both new format (config/model_state_dict) and old format (cfg/model keys).
"""
import sys, math
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_from_checkpoint(ckpt_path, device="cpu"):
    """Load a model from checkpoint. Returns (model, rep_name) or (None, None)."""
    try:
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    except Exception as e:
        print(f"  [WARN] Could not open {ckpt_path}: {e}")
        return None, None

    # Support both key formats
    cfg      = ckpt.get("config") or ckpt.get("cfg") or {}
    state    = (ckpt.get("model_state_dict")
                or ckpt.get("state_dict")
                or ckpt.get("model"))

    rep_cfg   = cfg.get("reps", {})
    model_cfg = cfg.get("model", {})
    head_cfg  = cfg.get("head", {})

    rep = (rep_cfg.get("rep")
           or cfg.get("rep")
           or ckpt.get("rep")
           or "6d")

    # Auto-detect in_dim from the first backbone weight in state dict
    in_dim_detected = None
    if isinstance(state, dict):
        for k, v in state.items():
            if "backbone" in k and "weight" in k and hasattr(v, "shape") and len(v.shape) == 2:
                in_dim_detected = v.shape[1]
                break

    in_dim     = in_dim_detected or cfg.get("data", {}).get("num_points", 32)
    hidden_dim = model_cfg.get("hidden_dim", 256)
    depth      = model_cfg.get("depth", 3)
    dropout    = model_cfg.get("dropout", 0.0)
    num_charts = rep_cfg.get("num_charts", 3)

    try:
        from rotation_benchmark.models.factory import build_model
        model = build_model(
            rep=rep,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=dropout,
            angle_range=head_cfg.get("euler_angle_range", math.pi),
            euler_use_tanh=head_cfg.get("euler_use_tanh", True),
            lie_max_angle=head_cfg.get("lie_max_angle", math.pi),
            lie_use_tanh=head_cfg.get("lie_use_tanh", True),
            num_charts=num_charts,
        )
        if isinstance(state, dict):
            model.load_state_dict(state, strict=False)
        else:
            # state might itself be an OrderedDict from old save
            model.load_state_dict(state.state_dict() if hasattr(state, "state_dict") else state,
                                  strict=False)
        model.eval()
        return model, rep
    except Exception as e:
        print(f"  [WARN] build_model failed for {Path(ckpt_path).name} (rep={rep}): {e}")
        return None, None


def predict_R(model, x_np):
    """Forward pass → cleaned 3×3 rotation matrix (numpy). Handles Atlas multi-charts."""
    x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)
    if x.ndim == 3 and x.shape[1:] == (32, 3):
        x = x.flatten(1)
    with torch.no_grad():
        out = model(x)
    
    # 1. Handle Atlas (R_stack, w_stack, logits)
    if isinstance(out, tuple) and len(out) == 3 and out[0].ndim == 4:
        R_stack, _, logits = out  # (B, K, 3, 3), (B, K, 3), (B, K)
        best_idx = torch.argmax(logits, dim=1)
        R = R_stack[torch.arange(R_stack.shape[0]), best_idx] # (B, 3, 3)
    
    # 2. Handle standard Reps (R, rep_out)
    elif isinstance(out, tuple):
        R = out[0]
    elif isinstance(out, dict):
        R = out.get("R") or out.get("pred_R") or list(out.values())[0]
    else:
        R = out

    R = R.squeeze().numpy()
    if R.shape != (3, 3):
        return None

    # Project to SO(3)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R

