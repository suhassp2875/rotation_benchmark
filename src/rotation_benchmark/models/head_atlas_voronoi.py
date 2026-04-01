
from __future__ import annotations
import math
import torch
import torch.nn as nn
from rotation_benchmark.geometry.so3 import exp_map

class AtlasHeadVoronoi(nn.Module):
    """
    Dynamic Atlas Head (Voronoi Variant).
    No learned router. Just K charts.
    Selection is handled by geometric proximity to anchors (Voronoi regions).
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_charts: int = 4, max_angle: float = 2.0):
        super().__init__()
        self.num_charts = num_charts
        self.max_angle = float(max_angle)
        
        # Separate MLPs for each chart
        self.chart_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 3)
            ) for _ in range(num_charts)
        ])
        
        self.register_buffer("anchors", self._generate_anchors(num_charts))
        
    def _generate_anchors(self, k):
        # Improved Anchor Distribution for K=4 (Tetrahedral)
        # Evenly spacing points on SO(3) is hard, but we can distribute them in R4 (quaternions)
        # For K=4, let's use: Identity + 3 Orthogonal Rotations (180 deg around X, Y, Z)
        # This forms a nice group V4 (Klein Four-Group).
        anchors = []
        anchors.append(torch.eye(3)) # Identity
        
        if k >= 4:
            # 180 deg rotations around principal axes
            # Rx(pi)
            anchors.append(torch.diag(torch.tensor([1., -1., -1.])))
            # Ry(pi)
            anchors.append(torch.diag(torch.tensor([-1., 1., -1.])))
            # Rz(pi)
            anchors.append(torch.diag(torch.tensor([-1., -1., 1.])))
        elif k == 3:
             # As before
             anchors.append(torch.tensor([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]))
             anchors.append(torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]))
             
        return torch.stack(anchors)

    def forward(self, h: torch.Tensor):
        B = h.shape[0]
        K = self.num_charts
        
        predictions = []
        raw_outputs = []
        
        for k in range(K):
            w_raw = self.chart_nets[k](h)
            w = self.max_angle * torch.tanh(w_raw)
            R_delta = exp_map(w)
            R_anchor = self.anchors[k].unsqueeze(0).expand(B, -1, -1)
            R_pred = torch.bmm(R_delta, R_anchor)
            predictions.append(R_pred)
            raw_outputs.append(w)
            
        R_stack = torch.stack(predictions, dim=1) # (B, K, 3, 3)
        w_stack = torch.stack(raw_outputs, dim=1) # (B, K, 3)
        
        return R_stack, w_stack
