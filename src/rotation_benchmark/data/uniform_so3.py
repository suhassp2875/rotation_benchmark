from __future__ import annotations
import math
import numpy as np
import torch
from rotation_benchmark.geometry.conversions import quat_to_rotmat

def random_uniform_rotmat(n: int, seed: int | None = None, device: str = "cpu", dtype=torch.float32) -> torch.Tensor:
    """
    Uniform random rotations on SO(3) using Shoemake's method (uniform random quaternions).
    Returns: (n,3,3)
    """
    rng = np.random.RandomState(seed)
    u1 = rng.rand(n)
    u2 = rng.rand(n)
    u3 = rng.rand(n)

    qx = np.sqrt(1.0 - u1) * np.sin(2.0 * math.pi * u2)
    qy = np.sqrt(1.0 - u1) * np.cos(2.0 * math.pi * u2)
    qz = np.sqrt(u1) * np.sin(2.0 * math.pi * u3)
    qw = np.sqrt(u1) * np.cos(2.0 * math.pi * u3)

    q = torch.tensor(np.stack([qx, qy, qz, qw], axis=-1), device=device, dtype=dtype)
    return quat_to_rotmat(q)

def random_axis(n: int, device: str = "cpu", dtype=torch.float32) -> torch.Tensor:
    """
    Uniformly sample n axes on the unit sphere S^2.
    """
    v = torch.randn(n, 3, device=device, dtype=dtype)
    return v / torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=1e-8)
