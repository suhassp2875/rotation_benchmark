from __future__ import annotations
import math
import numpy as np
import torch
from .conversions import quat_to_rotmat, euler_to_rotmat, rotmat_to_euler
from .so3 import log_map, exp_map

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

def sample_stratified_rotations(n: int, region_type: str = "uniform", seed: int | None = None, device: str = "cpu", dtype=torch.float32) -> torch.Tensor:
    """
    Sample rotations specifically designed to hit failure regimes:
    - 'gimbal': Pitch near +/- 90 degrees (ZYX convention).
    - 'antipodal': Rotation angle near 180 degrees.
    - 'safe': Away from singularities (small angles, near identity).
    - 'uniform': Standard uniform sampling.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if region_type == "uniform":
        return random_uniform_rotmat(n, seed=seed, device=device, dtype=dtype)
    
    elif region_type == "gimbal":
        # Sample Roll and Yaw uniformly, but Pitch near +/- 90 deg.
        # ZYX Euler: [z (yaw), y (pitch), x (roll)]
        yaw = torch.rand(n, device=device, dtype=dtype) * 2 * math.pi - math.pi
        roll = torch.rand(n, device=device, dtype=dtype) * 2 * math.pi - math.pi
        # Pitch: Sample near 90 or -90
        side = torch.randint(0, 2, (n,), device=device).float() * 2 - 1 # +/- 1
        pitch = side * (math.pi / 2.0 - torch.rand(n, device=device, dtype=dtype) * 0.1) # within 0.1 rad
        
        euler = torch.stack([yaw, pitch, roll], dim=-1)
        return euler_to_rotmat(euler, convention="ZYX")

    elif region_type == "antipodal":
        # Sample rotations with angles near PI
        axis = random_axis(n, device=device, dtype=dtype)
        angle = math.pi - torch.rand(n, device=device, dtype=dtype) * 0.2 # within 0.2 rad of PI
        vec = axis * angle.unsqueeze(-1)
        return exp_map(vec)

    elif region_type == "safe":
        # Sample rotations with small angles (near identity)
        axis = random_axis(n, device=device, dtype=dtype)
        angle = torch.rand(n, device=device, dtype=dtype) * (math.pi / 4.0) # < 45 degrees
        vec = axis * angle.unsqueeze(-1)
        return exp_map(vec)
    
    else:
        raise ValueError(f"Unknown region_type: {region_type}")
