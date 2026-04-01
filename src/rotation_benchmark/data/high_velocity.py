from __future__ import annotations
import numpy as np
import torch
from rotation_benchmark.geometry.so3 import exp_map

def high_velocity_sequence(
    T: int,
    omega_std: float = 0.35,
    seed: int | None = None,
    device: str = "cpu",
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Generates a rotation sequence: R_t = R_{t-1} @ exp(omega_t), omega_t ~ N(0, omega_std^2 I)
    Returns: (T,3,3) with R_0 = I
    """
    rng = np.random.RandomState(seed)
    # T frames means T-1 transitions
    omega = rng.normal(scale=omega_std, size=(T - 1, 3))
    omega = torch.tensor(omega, device=device, dtype=dtype)

    R = torch.eye(3, device=device, dtype=dtype)
    seq = [R]
    for t in range(T - 1):
        R = R @ exp_map(omega[t])
        seq.append(R)
    return torch.stack(seq, dim=0)
