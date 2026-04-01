from __future__ import annotations
import numpy as np
import torch
from rotation_benchmark.geometry.so3 import exp_map

def perturb_rotmat(R: torch.Tensor, sigma_rad: float, seed: int | None = None) -> torch.Tensor:
    """
    Right-multiplies a small random rotation: R_noisy = R @ exp(eps), eps ~ N(0, sigma_rad^2 I)
    """
    if sigma_rad <= 0:
        return R
    rng = np.random.RandomState(seed)
    eps = rng.normal(scale=sigma_rad, size=R.shape[:-2] + (3,))
    eps = torch.tensor(eps, device=R.device, dtype=R.dtype)
    dR = exp_map(eps)
    return R @ dR
