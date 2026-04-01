
from __future__ import annotations
import math
import torch
import torch.nn as nn
from rotation_benchmark.geometry.so3 import exp_map

class AtlasHead(nn.Module):
    """
    Dynamic Atlas Regression Head.
    Outputs K rotation hypotheses (charts). 
    Each hypothesis is a small rotation 'delta' composed with a fixed 'anchor' rotation.
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_charts: int = 3, max_angle: float = 2.0):
        super().__init__()
        self.num_charts = num_charts
        self.max_angle = float(max_angle) # Restrict each chart to a safe radius (e.g. < pi)
        
        # We can share the feature extractor or have separate ones.
        # Let's have separate small MLPs for each chart to allow specialization
        self.chart_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 3)
            ) for _ in range(num_charts)
        ])
        
        # Fixed Anchors (Offsets) to cover SO(3)
        # 1. Identity
        # 2. 120 deg around Y
        # 3. 240 deg around Y (or -120)
        # This is a naive covering but better than single chart.
        # Ideally we'd use tetrahedrally symmetric rotations for K=4.
        # For K=3, let's just spread them out.
        
        # Selector / Router
        # Predicts log-probability (logits) for each chart being the best
        self.selector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_charts)
        )
        
        self.register_buffer("anchors", self._generate_anchors(num_charts))
        
    def _generate_anchors(self, k):
        # Generate K anchors maximizing spread on SO(3)
        anchors = []
        anchors.append(torch.eye(3)) # Identity
        
        if k == 2:
            # Antipodal: I, Rx(pi)
            anchors.append(torch.diag(torch.tensor([1., -1., -1.])))
        elif k == 3:
            # Equatorial Spacing
             anchors.append(torch.tensor([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]))
             anchors.append(torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]))
        elif k == 4:
            # Tetrahedral (Klein V4)
            anchors.append(torch.diag(torch.tensor([1., -1., -1.])))
            anchors.append(torch.diag(torch.tensor([-1., 1., -1.])))
            anchors.append(torch.diag(torch.tensor([-1., -1., 1.])))
        elif k == 6:
            # Cubic (Faces)
            # Rx(pi), Ry(pi), Rz(pi), Rx(pi/2), Ry(pi/2)
            # Just use 90 deg rotations
            anchors.append(torch.diag(torch.tensor([1., -1., -1.]))) # X_180
            anchors.append(torch.diag(torch.tensor([-1., 1., -1.]))) # Y_180
            anchors.append(torch.diag(torch.tensor([-1., -1., 1.]))) # Z_180
            
            # Plus 90 deg rotations
            Rx90 = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
            Ry90 = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
            anchors.append(Rx90)
            anchors.append(Ry90)
        else:
             # Fallback: Random rotations if K is weird
             for _ in range(k-1):
                 # Use random orthogonal matrices?
                 # For now, just identity to avoid bugs in training
                 anchors.append(torch.eye(3))
             
        return torch.stack(anchors)

    def forward(self, h: torch.Tensor):
        # h: (B, D)
        B = h.shape[0]
        K = self.num_charts
        
        # Predict selection logits
        logits = self.selector(h) # (B, K)
        
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
        
        return R_stack, w_stack, logits
