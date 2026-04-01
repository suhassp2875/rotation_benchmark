from __future__ import annotations
import torch

def _safe_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.clamp(torch.linalg.norm(x, dim=-1, keepdim=True), min=eps)

def normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / _safe_norm(v, eps=eps)

def hat(w: torch.Tensor) -> torch.Tensor:
    """
    w: (..., 3)
    returns: (..., 3, 3) skew-symmetric matrix
    """
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    O = torch.zeros_like(wx)
    return torch.stack(
        [
            torch.stack([O, -wz, wy], dim=-1),
            torch.stack([wz, O, -wx], dim=-1),
            torch.stack([-wy, wx, O], dim=-1),
        ],
        dim=-2,
    )

def vee(W: torch.Tensor) -> torch.Tensor:
    """
    W: (..., 3, 3) skew-symmetric
    returns: (..., 3)
    """
    return torch.stack([W[..., 2, 1], W[..., 0, 2], W[..., 1, 0]], dim=-1)

def exp_map(w: torch.Tensor) -> torch.Tensor:
    """
    Exponential map so(3)->SO(3) via Rodrigues.
    w: (..., 3) axis-angle vector
    """
    device = w.device
    dtype = w.dtype
    I = torch.eye(3, device=device, dtype=dtype).expand(w.shape[:-1] + (3, 3))

    theta = _safe_norm(w)  # (...,1)
    W = hat(w)

    theta2 = theta * theta
    small = theta < 1e-4

    A = torch.sin(theta) / theta
    B = (1.0 - torch.cos(theta)) / theta2

    if small.any():
        t2 = theta2
        A_series = 1.0 - t2 / 6.0 + (t2 * t2) / 120.0
        B_series = 0.5 - t2 / 24.0 + (t2 * t2) / 720.0
        A = torch.where(small, A_series, A)
        B = torch.where(small, B_series, B)

    A = A[..., None]  # (...,1,1)
    B = B[..., None]
    return I + A * W + B * (W @ W)

def log_map(R: torch.Tensor) -> torch.Tensor:
    """
    Logarithm map SO(3)->so(3) returning axis-angle vector.
    R: (..., 3, 3)
    """
    import math
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)  # (...)

    Rt = R.transpose(-1, -2)
    skew = R - Rt
    v = vee(skew)

    sin_theta = torch.sin(theta)
    small = theta < 1e-4
    coeff = theta / (2.0 * torch.clamp(sin_theta, min=1e-8))
    w = coeff[..., None] * v
    if small.any():
        w_small = 0.5 * v
        w = torch.where(small[..., None], w_small, w)

    near_pi = (math.pi - theta).abs() < 1e-3
    if near_pi.any():
        Rn = R[near_pi]
        diag = torch.stack([Rn[..., 0, 0], Rn[..., 1, 1], Rn[..., 2, 2]], dim=-1)

        ax = torch.sqrt(torch.clamp((diag[..., 0] - diag[..., 1] - diag[..., 2] + 1.0) / 2.0, min=0.0))
        ay = torch.sqrt(torch.clamp((-diag[..., 0] + diag[..., 1] - diag[..., 2] + 1.0) / 2.0, min=0.0))
        az = torch.sqrt(torch.clamp((-diag[..., 0] - diag[..., 1] + diag[..., 2] + 1.0) / 2.0, min=0.0))

        axis = torch.stack([ax, ay, az], dim=-1)

        axis = torch.where(
            (Rn[..., 2, 1] - Rn[..., 1, 2])[..., None] < 0,
            axis * torch.tensor([-1.0, 1.0, 1.0], device=axis.device, dtype=axis.dtype),
            axis,
        )
        axis = torch.where(
            (Rn[..., 0, 2] - Rn[..., 2, 0])[..., None] < 0,
            axis * torch.tensor([1.0, -1.0, 1.0], device=axis.device, dtype=axis.dtype),
            axis,
        )
        axis = torch.where(
            (Rn[..., 1, 0] - Rn[..., 0, 1])[..., None] < 0,
            axis * torch.tensor([1.0, 1.0, -1.0], device=axis.device, dtype=axis.dtype),
            axis,
        )

        axis = normalize(axis, eps=1e-8)
        w_pi = theta[near_pi][..., None] * axis

        w2 = w.clone()
        w2[near_pi] = w_pi
        w = w2

    return w
