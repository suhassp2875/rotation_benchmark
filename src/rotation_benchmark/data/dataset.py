from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset

from rotation_benchmark.data.uniform_so3 import random_uniform_rotmat
from rotation_benchmark.data.gimbal_lock import gimbal_lock_rotmat
from rotation_benchmark.data.noise_perturb import perturb_rotmat

class RotationPointCloudDataset(Dataset):
    """
    Observation: rotate a fixed canonical point set by R, add observation noise, flatten.
    Target: ground truth rotation matrix R_gt.
    """

    def __init__(
        self,
        n: int,
        mode: str = "uniform",
        num_points: int = 32,
        obs_noise_std: float = 0.01,
        rot_noise_std: float = 0.0,
        seed: int = 0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        gimbal_pitch_deg_low: float = 85.0,
        gimbal_pitch_deg_high: float = 95.0,
    ):
        super().__init__()
        self.n = int(n)
        self.mode = mode
        self.num_points = int(num_points)
        self.obs_noise_std = float(obs_noise_std)
        self.rot_noise_std = float(rot_noise_std)
        self.seed = int(seed)
        self.device = device
        self.dtype = dtype

        # Canonical point set (deterministic)
        rng = np.random.RandomState(0)
        pts = rng.normal(size=(self.num_points, 3))
        pts = pts / np.linalg.norm(pts, axis=-1, keepdims=True)
        self.P = torch.tensor(pts, device=device, dtype=dtype)  # (N,3)

        # Generate ground truth rotations
        from rotation_benchmark.geometry.sampling import sample_stratified_rotations
        
        # Mapping dataset modes to sampling region types
        mode_map = {
            "uniform": "uniform",
            "gimbal": "gimbal",
            "antipodal": "antipodal",
            "safe": "safe",
            "antipodal_heavy": "antipodal", # Legacy / variant
            "identity_heavy": "safe",     # Legacy / variant
        }
        
        region_type = mode_map.get(self.mode, self.mode)
        R = sample_stratified_rotations(self.n, region_type=region_type, seed=self.seed, device=device, dtype=dtype)
        angles = None # Could extract if needed, but not required for benchmark

        # Optional rotation noise (stress)
        if self.rot_noise_std > 0:
            R_obs = perturb_rotmat(R, sigma_rad=self.rot_noise_std, seed=self.seed + 1337)
        else:
            R_obs = R

        X = torch.einsum("bij,nj->bni", R_obs, self.P)  # (n,N,3)
        if self.obs_noise_std > 0:
            X = X + torch.randn_like(X) * self.obs_noise_std

        self.x = X.reshape(self.n, -1).contiguous()  # (n, N*3)
        self.R_gt = R.contiguous()
        self.angles = angles

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        item = {"x": self.x[idx], "R_gt": self.R_gt[idx]}
        if self.angles is not None:
            item["angles"] = self.angles[idx]
        return item
