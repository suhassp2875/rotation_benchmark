
from __future__ import annotations
import torch
import torch.nn as nn

class SVDHead(nn.Module):
    """
    SVD-based Rotation Head.
    Predicts a 3x3 matrix A and projects it to SO(3) using SVD.
    R = U * diag(1, 1, det(U V^T)) * V^T
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 9)

    def forward(self, h: torch.Tensor):
        x = self.linear(h)
        A = x.view(-1, 3, 3)
        
        # SVD Orthogonalization
        # PyTorch SVD is differentiable
        U, S, Vh = torch.linalg.svd(A)
        # Vh is V^T in torch.linalg.svd
        
        # Standard Procrustes: R = U V^T
        # But we need to ensure det(R) = 1 (Rotation, not Reflection)
        # s = det(U V^T)
        det = torch.det(torch.matmul(U, Vh))
        
        # Construct correction diagonal
        diag_S = torch.ones_like(S)
        diag_S[..., -1] = torch.sign(det)
        
        # R = U * diag(s) * Vh
        R = torch.matmul(U * diag_S.unsqueeze(-2), Vh)
        
        # Pseudo-representation for consistency with other heads (return R, raw)
        return R, A
