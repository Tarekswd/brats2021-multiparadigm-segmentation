"""
Focal Loss for addressing class imbalance in brain tumor subregions.
Down-weights easy background examples and focuses gradient on ambiguous tumor boundaries.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Sigmoid Focal Loss for multi-region 3D segmentation.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, C, D, H, W]
        target: [B, C, D, H, W]
        """
        prob = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

        p_t = prob * target + (1.0 - prob) * (1.0 - target)
        alpha_factor = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)

        focal = alpha_factor * modulating_factor * bce
        return torch.mean(focal)
