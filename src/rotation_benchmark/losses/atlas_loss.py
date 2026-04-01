
import torch
import torch.nn as nn
from rotation_benchmark.geometry.metrics import geodesic_distance

class AtlasLoss(nn.Module):
    """
    Winner-Take-All Loss for Dynamic Atlas Regression.
    Only backpropagates the error of the best-matching chart.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_R_stack, gt_R):
        # pred_R_stack: (B, K, 3, 3)
        # gt_R: (B, 3, 3)
        
        B, K, _, _ = pred_R_stack.shape
        
        # Expand GT to match K
        gt_R_exp = gt_R.unsqueeze(1).expand(-1, K, -1, -1) # (B, K, 3, 3)
        
        # Flatten batch and K dimension to compute pairwise distances
        pred_R_flat = pred_R_stack.reshape(B*K, 3, 3)
        gt_R_flat = gt_R_exp.reshape(B*K, 3, 3)
        
        # Compute losses
        losses_flat = geodesic_distance(pred_R_flat, gt_R_flat) # (B*K)
        losses = losses_flat.view(B, K) # (B, K)
        
        # Find min loss for each sample (Best Chart)
        min_losses, min_indices = torch.min(losses, dim=1) # (B,), (B,)
        
        # Return mean of best losses and the indices
        return min_losses.mean(), min_indices
