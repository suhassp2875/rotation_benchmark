import torch
import pytest
from rotation_benchmark.geometry.so3 import exp_map, log_map
from rotation_benchmark.geometry.metrics import geodesic_distance
from rotation_benchmark.geometry.conversions import sixd_to_rotmat, rotmat_to_sixd

def test_exp_log_consistency():
    w = torch.randn(10, 3) * 0.5
    R = exp_map(w)
    w_recon = log_map(R)
    
    # Error should be small
    diff = torch.norm(w - w_recon, dim=-1)
    # log(exp(w)) could have axis flip if near pi, 
    # but randn*0.5 is far from pi (~randn[0, 1.5]).
    assert (diff < 1e-4).all()

def test_sixd_conversion():
    R = exp_map(torch.randn(10, 3))
    sixd = rotmat_to_sixd(R)
    R_recon = sixd_to_rotmat(sixd)
    
    err = geodesic_distance(R, R_recon)
    # acos is sensitive near 1.0; 1e-3 rad is ~0.05 deg.
    assert (err < 1e-3).all()

def test_geodesic_identity():
    R = torch.eye(3).unsqueeze(0)
    err = geodesic_distance(R, R)
    assert (err < 1e-3).all()
