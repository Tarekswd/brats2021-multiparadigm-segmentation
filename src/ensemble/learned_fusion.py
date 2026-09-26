"""
Learned 3D Fusion Network for BraTS 2021 Ensembling (Section 32).
Compact parametric network trained strictly on out-of-fold probability maps.
Prevents overfitting by utilizing 1x1x1 point-wise convolutional mixing with residual connections.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedFusionNet(nn.Module):
    """
    Compact 3D fusion network.
    Inputs: [B, num_models * 3, D, H, W] (concatenated WT/TC/ET probabilities from each model)
    Outputs: [B, 3, D, H, W] (refined WT, TC, ET logits)
    """

    def __init__(
        self,
        num_models: int = 3,
        num_classes: int = 3,
        hidden_channels: int = 16,
        include_mri: bool = False
    ):
        super().__init__()
        in_channels = num_models * num_classes + (4 if include_mri else 0)
        self.num_models = num_models
        self.num_classes = num_classes

        # Point-wise 1x1x1 convolutional layers
        self.mixer = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=1),
            nn.GroupNorm(num_groups=min(4, hidden_channels), num_channels=hidden_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GroupNorm(num_groups=min(4, hidden_channels), num_channels=hidden_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(hidden_channels, num_classes, kernel_size=1)
        )

    def forward(self, prob_inputs: torch.Tensor, mri: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        prob_inputs: [B, num_models * 3, D, H, W]
        """
        if mri is not None:
            x = torch.cat([prob_inputs, mri], dim=1)
        else:
            x = prob_inputs

        residual = 0.0
        # Compute base average across the models for residual skip
        for m in range(self.num_models):
            residual = residual + prob_inputs[:, m * 3 : (m + 1) * 3]
        base_avg = residual / self.num_models

        delta = self.mixer(x)
        # Logit adjustment on top of base average
        return torch.logit(base_avg.clamp(1e-5, 1.0 - 1e-5)) + delta
