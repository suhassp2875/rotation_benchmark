from __future__ import annotations
import math
import torch
import torch.nn as nn
from rotation_benchmark.geometry.so3 import exp_map

class LieHead(nn.Module):
    def __init__(self, in_dim: int, max_angle: float = math.pi, use_tanh: bool = True):
        super().__init__()
        self.proj = nn.Linear(in_dim, 3)
        self.max_angle = float(max_angle)
        self.use_tanh = bool(use_tanh)

    def forward(self, h: torch.Tensor):
        w_raw = self.proj(h)
        w = self.max_angle * torch.tanh(w_raw) if self.use_tanh else w_raw
        R = exp_map(w)
        return R, w

class LieAxisAngleHead(nn.Module):
    def __init__(self, in_dim: int, max_angle: float = math.pi):
        super().__init__()
        self.proj_axis = nn.Linear(in_dim, 3)
        self.proj_angle = nn.Linear(in_dim, 1)
        self.max_angle = float(max_angle)

    def forward(self, h: torch.Tensor):
        axis_raw = self.proj_axis(h)
        axis = nn.functional.normalize(axis_raw, dim=-1, eps=1e-8)
        
        angle_raw = self.proj_angle(h)
        # Use sigmoid to bound strictly to [0, max_angle]
        angle = self.max_angle * torch.sigmoid(angle_raw)
        
        w = axis * angle
        R = exp_map(w)
        return R, w

