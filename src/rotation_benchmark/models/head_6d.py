from __future__ import annotations
import torch
import torch.nn as nn
from rotation_benchmark.geometry.conversions import sixd_to_rotmat

class SixDHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, 6)

    def forward(self, h: torch.Tensor):
        a = self.proj(h)
        R = sixd_to_rotmat(a)
        return R, a
