from __future__ import annotations
import torch
import torch.nn as nn
from rotation_benchmark.geometry.metrics import geodesic_distance

class GeodesicLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be mean/sum/none")
        self.reduction = reduction

    def forward(self, R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
        d = geodesic_distance(R_pred, R_gt)  # (B,)
        if self.reduction == "mean":
            return d.mean()
        if self.reduction == "sum":
            return d.sum()
        return d
