"""
SegResNet Architecture for BraTS 2021.
2nd Place Challenge Architecture (MONAI / Myronenko et al.).
Residual encoder-decoder network with Group Normalization and spatial dropout.
"""

from typing import Tuple, List, Optional
import torch
import torch.nn as nn
from monai.networks.nets import SegResNet as MonaiSegResNet


class SegResNet(nn.Module):
    """
    SegResNet wrapper configured for BraTS 2021 multi-modal brain tumor segmentation.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        init_filters: int = 16,
        blocks_down: Tuple[int, ...] = (1, 2, 2, 4),
        blocks_up: Tuple[int, ...] = (1, 1, 1),
        dropout_prob: float = 0.2
    ):
        super().__init__()
        self.model = MonaiSegResNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            init_filters=init_filters,
            blocks_down=blocks_down,
            blocks_up=blocks_up,
            dropout_prob=dropout_prob
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
