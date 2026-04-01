import torch
from rotation_benchmark.losses.geodesic import GeodesicLoss
from rotation_benchmark.losses.antipodal import AntipodalLieLoss
from rotation_benchmark.geometry.so3 import exp_map

def test_geodesic_loss():
    loss_fn = GeodesicLoss(reduction="mean")
    R1 = exp_map(torch.randn(5, 3))
    R2 = R1.clone()
    
    loss = loss_fn(R1, R2)
    # Geodesic loss uses acos, which is sensitive near 1.0 (error proportional to sqrt(eps)).
    # 1e-3 rad is ~0.05 deg, acceptable for float32 identity test.
    assert loss.item() < 1e-3

def test_antipodal_loss_invariance():
    """Test that antipodal loss treats theta and theta + 2pi as similar if gated."""
    loss_fn = AntipodalLieLoss()
    
    # Near pi
    w_gt = torch.tensor([[3.14, 0, 0]])
    R_gt = exp_map(w_gt)
    
    # Prediction is near -pi (which is the same rotation)
    w_pred = torch.tensor([[-3.14, 0, 0]])
    R_pred = exp_map(w_pred)
    
    loss = loss_fn(R_pred, R_gt, w_pred)
    # The axis loss should trigger or the symmetry should be handled
    assert loss.item() < 0.1 
