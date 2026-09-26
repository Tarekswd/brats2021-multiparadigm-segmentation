"""
Dice Loss implementations for 3D multi-region segmentation.
Supports:
  - Standard soft Dice loss
  - Batch Dice loss (treating the entire batch as one volume, as used by nnU-Net and the BraTS 2021 winners)
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice loss for binary/sigmoid multi-channel inputs (e.g. WT, TC, ET).
    """

    def __init__(self, smooth: float = 1e-5, batch_dice: bool = True):
        super().__init__()
        self.smooth = smooth
        self.batch_dice = batch_dice

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: [B, C, D, H, W] (probabilities after sigmoid, or logits if apply_sigmoid=True)
        target: [B, C, D, H, W] (binary targets)
        """
        if self.batch_dice:
            # Batch Dice: sum across batch as well as spatial dimensions
            dims = (0, 2, 3, 4)
            intersection = torch.sum(pred * target, dim=dims)
            cardinality = torch.sum(pred + target, dim=dims)
        else:
            # Sample-wise Dice: sum across spatial dimensions only, then average over batch
            dims = (2, 3, 4)
            intersection = torch.sum(pred * target, dim=dims)
            cardinality = torch.sum(pred + target, dim=dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        loss = 1.0 - dice
        return torch.mean(loss)
