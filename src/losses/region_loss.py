"""
Region-aware loss with decoupled per-region weighting:
  L = lambda_wt * L_wt + lambda_tc * L_tc + lambda_et * L_et
Allows prioritizing high-difficulty enhancing tumor (ET) or tumor core (TC).
"""

import torch
import torch.nn as nn
from typing import List, Optional


class RegionLoss(nn.Module):
    """
    Weighted region loss combining individual region losses (WT, TC, ET).
    """

    def __init__(
        self,
        base_loss_fn: nn.Module,
        weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        # Default weights: equal weighting [1.0, 1.0, 1.0] for WT, TC, ET
        self.weights = weights if weights is not None else [1.0, 1.0, 1.0]

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, 3, D, H, W]
        target: [B, 3, D, H, W]
        Channel 0: WT, Channel 1: TC, Channel 2: ET
        """
        num_channels = logits.shape[1]
        total_loss = 0.0
        w_sum = sum(self.weights[:num_channels])

        for c in range(num_channels):
            w = self.weights[c] if c < len(self.weights) else 1.0
            c_loss = self.base_loss_fn(logits[:, c:c+1], target[:, c:c+1])
            total_loss = total_loss + w * c_loss

        return total_loss / (w_sum if w_sum > 0 else 1.0)
