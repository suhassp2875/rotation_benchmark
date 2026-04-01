import torch
import numpy as np
import os, sys
from pathlib import Path

# Add project roots
sys.path.insert(0, os.path.abspath("src"))

from rotation_benchmark.geometry.conversions import rotmat_to_euler_zyx

def check_z_sweep():
    axis = np.array([0.0, 0, 1.0])
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    
    angles = np.linspace(0, 180, 181)
    for a in angles:
        rad = np.radians(a)
        R_gt = np.eye(3) + np.sin(rad)*K + (1-np.cos(rad))*(K@K)
        euler = rotmat_to_euler_zyx(torch.from_numpy(R_gt).unsqueeze(0).float()).squeeze()
        yaw, pitch, roll = np.degrees(euler.numpy())
        if a > 165:
            print(f"Angle: {a:6.1f} | Yaw: {yaw:7.2f} | Pitch: {pitch:7.2f} | Roll: {roll:7.2f}")

if __name__ == "__main__":
    check_z_sweep()
