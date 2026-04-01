from __future__ import annotations
import torch

from .so3 import log_map, exp_map
from .conversions import rotmat_to_euler

def geodesic_distance(R1: torch.Tensor, R2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Returns angle (rad) between rotations.
    R1, R2: (..., 3, 3)
    """
    R = R1.transpose(-1, -2) @ R2
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = (tr - 1.0) / 2.0
    # Stability: acos gradient is inf at +/- 1.0. 
    # Use 1e-7 margin for safe backprop.
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)
    return theta

def gimbal_proximity(R: torch.Tensor, convention="ZYX") -> torch.Tensor:
    """Returns absolute distance of pitch from +/- 90 degrees."""
    euler = rotmat_to_euler(R, convention=convention) # (..., 3) [yaw, pitch, roll]
    pitch = euler[..., 1]
    dist = torch.abs(torch.abs(pitch) - (torch.pi / 2.0))
    return dist

def antipodal_proximity(R: torch.Tensor) -> torch.Tensor:
    """Returns absolute distance of rotation angle from 180 degrees."""
    angle = torch.norm(log_map(R), dim=-1)
    return torch.abs(torch.pi - angle)

def discontinuity_rate(model, x: torch.Tensor, rot_gt: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """
    Measure how much the prediction jumps under small input perturbation.
    x: (B, N, 3)
    """
    model.eval()
    with torch.no_grad():
        pred1 = model(x)
        if isinstance(pred1, tuple): pred1 = pred1[0]
        if pred1.dim() == 4: pred1 = pred1[:, 0] # Take first chart for simplicity
        
        # Perturb input slightly
        x_perturbed = x + torch.randn_like(x) * eps
        pred2 = model(x_perturbed)
        if isinstance(pred2, tuple): pred2 = pred2[0]
        if pred2.dim() == 4: pred2 = pred2[:, 0]
        
        return geodesic_distance(pred1, pred2)

def grad_norm(parameters, norm_type: float = 2.0) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = torch.linalg.vector_norm(p.grad.detach(), ord=norm_type).item()
        total += param_norm ** norm_type
    return total ** (1.0 / norm_type) if total > 0 else 0.0
