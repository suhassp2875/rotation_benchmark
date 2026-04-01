from __future__ import annotations
import torch
from .so3 import normalize

def quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    from .so3 import _safe_norm
    return q / _safe_norm(q, eps=eps)

def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    q: (..., 4) in (x, y, z, w)
    """
    q = quat_normalize(q)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = torch.stack(
        [
            torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
            torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], dim=-1),
            torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], dim=-1),
        ],
        dim=-2,
    )
    return R

def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    Returns q (...,4) in (x,y,z,w).
    """
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    q = torch.zeros(R.shape[:-2] + (4,), device=R.device, dtype=R.dtype)

    mask = tr > 0.0
    if mask.any():
        t = tr[mask]
        s = torch.sqrt(t + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[mask, 2, 1] - R[mask, 1, 2]) / s
        qy = (R[mask, 0, 2] - R[mask, 2, 0]) / s
        qz = (R[mask, 1, 0] - R[mask, 0, 1]) / s
        q[mask] = torch.stack([qx, qy, qz, qw], dim=-1)

    mask2 = ~mask
    if mask2.any():
        Rn = R[mask2]
        diag = torch.stack([Rn[..., 0, 0], Rn[..., 1, 1], Rn[..., 2, 2]], dim=-1)
        idx = torch.argmax(diag, dim=-1)

        q_sub = torch.zeros((Rn.shape[0], 4), device=R.device, dtype=R.dtype)
        for i in range(3):
            sel = idx == i
            if not sel.any():
                continue
            Rs = Rn[sel]
            if i == 0:
                s = torch.sqrt(1.0 + Rs[:, 0, 0] - Rs[:, 1, 1] - Rs[:, 2, 2]) * 2.0
                qx = 0.25 * s
                qy = (Rs[:, 0, 1] + Rs[:, 1, 0]) / s
                qz = (Rs[:, 0, 2] + Rs[:, 2, 0]) / s
                qw = (Rs[:, 2, 1] - Rs[:, 1, 2]) / s
            elif i == 1:
                s = torch.sqrt(1.0 - Rs[:, 0, 0] + Rs[:, 1, 1] - Rs[:, 2, 2]) * 2.0
                qx = (Rs[:, 0, 1] + Rs[:, 1, 0]) / s
                qy = 0.25 * s
                qz = (Rs[:, 1, 2] + Rs[:, 2, 1]) / s
                qw = (Rs[:, 0, 2] - Rs[:, 2, 0]) / s
            else:
                s = torch.sqrt(1.0 - Rs[:, 0, 0] - Rs[:, 1, 1] + Rs[:, 2, 2]) * 2.0
                qx = (Rs[:, 0, 2] + Rs[:, 2, 0]) / s
                qy = (Rs[:, 1, 2] + Rs[:, 2, 1]) / s
                qz = 0.25 * s
                qw = (Rs[:, 1, 0] - Rs[:, 0, 1]) / s

            q_sub[sel] = torch.stack([qx, qy, qz, qw], dim=-1)

        q[mask2] = q_sub

    return quat_normalize(q)

def sixd_to_rotmat(a: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    6D continuous representation -> SO(3) using Gram-Schmidt.
    a: (..., 6)
    """
    a1 = a[..., 0:3]
    a2 = a[..., 3:6]
    b1 = normalize(a1, eps=eps)
    proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = normalize(a2 - proj, eps=eps)
    b3 = torch.cross(b1, b2, dim=-1)
    R = torch.stack([b1, b2, b3], dim=-1)  # columns
    return R

def rotmat_to_sixd(R: torch.Tensor) -> torch.Tensor:
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)

def fived_to_rotmat(a: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    5D representation -> SO(3) using Gram-Schmidt.
    a: (..., 5) -> (x1, x2, x3, y1, y2)
    We pad with a fixed 0 to get 6D input. 
    """
    a1 = a[..., 0:3]
    a2 = torch.cat([a[..., 3:5], torch.zeros_like(a[..., 4:5])], dim=-1)
    b1 = normalize(a1, eps=eps)
    proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = normalize(a2 - proj, eps=eps)
    b3 = torch.cross(b1, b2, dim=-1)
    R = torch.stack([b1, b2, b3], dim=-1)
    return R

def euler_zyx_to_rotmat(yaw: torch.Tensor, pitch: torch.Tensor, roll: torch.Tensor) -> torch.Tensor:
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)

    R00 = cy * cp
    R01 = cy * sp * sr - sy * cr
    R02 = cy * sp * cr + sy * sr

    R10 = sy * cp
    R11 = sy * sp * sr + cy * cr
    R12 = sy * sp * cr - cy * sr

    R20 = -sp
    R21 = cp * sr
    R22 = cp * cr

    return torch.stack(
        [
            torch.stack([R00, R01, R02], dim=-1),
            torch.stack([R10, R11, R12], dim=-1),
            torch.stack([R20, R21, R22], dim=-1),
        ],
        dim=-2,
    )

def rotmat_to_euler_zyx(R: torch.Tensor) -> torch.Tensor:
    R20 = R[..., 2, 0]
    pitch = torch.asin(torch.clamp(-R20, -1.0, 1.0))
    cp = torch.cos(pitch)
    singular = cp.abs() < 1e-6

    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    roll = torch.atan2(R[..., 2, 1], R[..., 2, 2])

    if singular.any():
        yaw_s = torch.atan2(-R[..., 0, 1], R[..., 1, 1])
        roll_s = torch.zeros_like(roll)
        yaw = torch.where(singular, yaw_s, yaw)
        roll = torch.where(singular, roll_s, roll)

    return torch.stack([yaw, pitch, roll], dim=-1)

def euler_to_rotmat(euler: torch.Tensor, convention: str = "ZYX") -> torch.Tensor:
    if convention == "ZYX":
        return euler_zyx_to_rotmat(euler[..., 0], euler[..., 1], euler[..., 2])
    raise NotImplementedError(f"Convention {convention} not implemented")

def rotmat_to_euler(R: torch.Tensor, convention: str = "ZYX") -> torch.Tensor:
    if convention == "ZYX":
        return rotmat_to_euler_zyx(R)
    raise NotImplementedError(f"Convention {convention} not implemented")
