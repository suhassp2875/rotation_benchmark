from __future__ import annotations
import math
import torch
import torch.nn as nn
from rotation_benchmark.geometry.conversions import euler_zyx_to_rotmat

class EulerHead(nn.Module):
    def __init__(self, in_dim: int, angle_range: float = math.pi, use_tanh: bool = True):
        super().__init__()
        self.proj = nn.Linear(in_dim, 3)
        self.angle_range = float(angle_range)
        self.use_tanh = bool(use_tanh)

    def forward(self, h: torch.Tensor):
        raw = self.proj(h)
        angles = self.angle_range * torch.tanh(raw) if self.use_tanh else raw
        yaw, pitch, roll = angles[..., 0], angles[..., 1], angles[..., 2]
        R = euler_zyx_to_rotmat(yaw, pitch, roll)
        return R, angles
