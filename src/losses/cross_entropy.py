"""
Binary Cross-Entropy Loss for multi-region sigmoid outputs (WT, TC, ET).
"""

import torch
import torch.nn as nn


class BCELoss(nn.Module):
    """Numerically stable BCE loss with logits for multi-label region segmentation."""

    def __init__(self, pos_weight: torch.Tensor = None):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, C, D, H, W] (raw network outputs before sigmoid)
        target: [B, C, D, H, W] (binary targets)
        """
        return self.loss_fn(logits, target)
