from __future__ import annotations
import math
import torch
import torch.nn as nn
from rotation_benchmark.geometry.so3 import log_map
from rotation_benchmark.geometry.metrics import geodesic_distance

def canonicalize_lie(w: torch.Tensor) -> torch.Tensor:
    """
    Enforce a deterministic hemisphere rule for axis-angle vectors.
    Rule: find component with max absolute value, force it to be positive.
    """
    theta = torch.linalg.norm(w, dim=-1, keepdim=True)
    axis = w / torch.clamp(theta, min=1e-8)
    
    # Find max absolute component index
    abs_axis = torch.abs(axis)
    max_idx = torch.argmax(abs_axis, dim=-1, keepdim=True)
    
    # Gather the sign of the max component
    max_val = torch.gather(axis, -1, max_idx)
    sign = torch.sign(max_val)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    
    # Flip axis and angle (equiv rotation)
    # Actually, for Lie vector w = theta * axis, 
    # to maintain same rotation, we can only flip (axis, theta) -> (-axis, -theta)
    # OR we are at theta = pi, where (axis, pi) == (-axis, pi).
    # Since our Lie heads usually project to a range (like [-pi, pi] or [0, pi]),
    # we target the canonical branch.
    
    return w * sign

class AntipodalLieLoss(nn.Module):
    def __init__(
        self, 
        lambda_theta: float = 0.1, 
        lambda_axis: float = 0.1,
        pi_start: float = 2.5, # ~143 deg
    ):
        super().__init__()
        self.lambda_theta = lambda_theta
        self.lambda_axis = lambda_axis
        self.pi_start = pi_start

    def forward(self, R_pred: torch.Tensor, R_gt: torch.Tensor, w_pred: torch.Tensor):
        """
        R_pred, R_gt: (B, 3, 3)
        w_pred: (B, 3) raw axis-angle vector from model head
        """
        # 1. Primary Geodesic Loss (radians)
        loss_geo = geodesic_distance(R_pred, R_gt).mean()
        
        # 2. Extract GT Lie parameters (theta, axis)
        w_gt = log_map(R_gt)
        theta_gt = torch.linalg.norm(w_gt, dim=-1, keepdim=True)
        u_gt = w_gt / torch.clamp(theta_gt, min=1e-8)
        
        # 3. Pred parameters
        theta_pred = torch.linalg.norm(w_pred, dim=-1, keepdim=True)
        u_pred = w_pred / torch.clamp(theta_pred, min=1e-8)
        
        # 4. Angle Loss (Critical for stability at pi)
        loss_theta = nn.functional.mse_loss(theta_pred, theta_gt)
        
        # 5. Axis Loss (Stability Checks)
        # Check 1: Gate on theta_gt ONLY
        # Smooth sin^2(theta_gt) kills axis supervision at 0 and pi
        w_sin2 = torch.sin(theta_gt)**2
        
        dot = (u_pred * u_gt).sum(dim=-1, keepdim=True)
        
        # Check 2: Smooth pi-ramp (no hard thresholds)
        # cosine-based ramp from pi_start to pi
        ramp = (theta_gt - self.pi_start) / (math.pi - self.pi_start)
        w_near_pi = torch.clamp(ramp, 0.0, 1.0)
        # Use smooth step or linear? linear is usually fine if wide. 
        # Let's use cosine ramp for maximum smoothness
        w_near_pi = 0.5 * (1.0 - torch.cos(w_near_pi * math.pi))
        
        # Check 3: Sign-invariant axis loss near pi (1 - |dot|)
        # Handles axis ambiguity (u ~ -u) without forcing discontinuities
        loss_axis_samples = w_sin2 * (1.0 - dot) + w_near_pi * (1.0 - torch.abs(dot))
        
        loss_axis = loss_axis_samples.mean()
        
        total_loss = loss_geo + self.lambda_theta * loss_theta + self.lambda_axis * loss_axis
        
        return total_loss
