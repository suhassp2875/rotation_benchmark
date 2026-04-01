from __future__ import annotations
import math
import torch.nn as nn

from rotation_benchmark.models.backbone import MLPBackbone
from rotation_benchmark.models.head_euler import EulerHead
from rotation_benchmark.models.head_quat import QuatHead
from rotation_benchmark.models.head_6d import SixDHead
from rotation_benchmark.models.head_lie import LieHead

def build_model(
    rep: str,
    in_dim: int,
    hidden_dim: int,
    depth: int,
    dropout: float,
    angle_range: float = math.pi,
    euler_use_tanh: bool = True,
    lie_max_angle: float = math.pi,
    lie_use_tanh: bool = True,
    **kwargs
) -> nn.Module:
    backbone = MLPBackbone(in_dim=in_dim, hidden_dim=hidden_dim, depth=depth, dropout=dropout)

    rep = rep.lower()
    if rep == "euler":
        head = EulerHead(in_dim=hidden_dim, angle_range=angle_range, use_tanh=euler_use_tanh)
    elif rep == "quat":
        head = QuatHead(in_dim=hidden_dim)
    elif rep in ("6d", "sixd"):
        head = SixDHead(in_dim=hidden_dim)
    elif rep in ("5d", "fived"):
        from rotation_benchmark.models.head_5d import FiveDHead
        head = FiveDHead(in_dim=hidden_dim)
    elif rep in ("lie", "lie_loggeo", "lie_raw"):
        head = LieHead(in_dim=hidden_dim, max_angle=lie_max_angle, use_tanh=lie_use_tanh)
    elif rep in ("lie_hardened", "lie_fullfix"):
        from rotation_benchmark.models.head_lie import LieAxisAngleHead
        head = LieAxisAngleHead(in_dim=hidden_dim, max_angle=math.pi) # Standard principal range
    elif rep == "atlas":
        from rotation_benchmark.models.head_atlas import AtlasHead
        k = kwargs.get("num_charts", 3)
        head = AtlasHead(in_dim=hidden_dim, hidden_dim=hidden_dim, num_charts=k, max_angle=2.0)
    elif rep == "atlas_voronoi":
        from rotation_benchmark.models.head_atlas_voronoi import AtlasHeadVoronoi
        # Uses K=4 (Klein V4 Group) for better coverage
        head = AtlasHeadVoronoi(in_dim=hidden_dim, hidden_dim=hidden_dim, num_charts=4, max_angle=2.0)
    elif rep == "svd":
        from rotation_benchmark.models.head_svd import SVDHead
        head = SVDHead(in_dim=hidden_dim)
    else:

        raise ValueError(f"Unknown rep: {rep}")

    class Model(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            h = self.backbone(x)
            return self.head(h)  # (R_pred, rep_out)

    return Model(backbone, head)
