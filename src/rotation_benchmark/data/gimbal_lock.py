from __future__ import annotations
import numpy as np
import torch
from rotation_benchmark.geometry.conversions import euler_zyx_to_rotmat

def gimbal_lock_rotmat(
    n: int,
    pitch_deg_low: float = 85.0,
    pitch_deg_high: float = 95.0,
    seed: int | None = None,
    device: str = "cpu",
    dtype=torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Samples rotations in the gimbal-lock regime for ZYX yaw-pitch-roll:
      pitch ~ Uniform([pitch_deg_low, pitch_deg_high]) (around 90°)
      yaw, roll ~ Uniform([-pi, pi])

    Returns:
      R: (n,3,3)
      angles: (n,3) = (yaw, pitch, roll) in radians
    """
    rng = np.random.RandomState(seed)
    pitch = np.deg2rad(rng.uniform(pitch_deg_low, pitch_deg_high, size=n))
    yaw = rng.uniform(-np.pi, np.pi, size=n)
    roll = rng.uniform(-np.pi, np.pi, size=n)

    yaw_t = torch.tensor(yaw, device=device, dtype=dtype)
    pitch_t = torch.tensor(pitch, device=device, dtype=dtype)
    roll_t = torch.tensor(roll, device=device, dtype=dtype)

    R = euler_zyx_to_rotmat(yaw_t, pitch_t, roll_t)
    angles = torch.stack([yaw_t, pitch_t, roll_t], dim=-1)
    return R, angles
