from __future__ import annotations
import torch
import torch.nn as nn
from rotation_benchmark.geometry.conversions import quat_to_rotmat, quat_normalize

class QuatHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, 4)

    def forward(self, h: torch.Tensor):
        q_raw = self.proj(h)
        q = quat_normalize(q_raw)
        R = quat_to_rotmat(q)
        return R, q
