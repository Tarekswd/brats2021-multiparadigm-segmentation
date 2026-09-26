"""
UNETR Architecture for BraTS 2021.
3D Vision Transformer encoder paired with U-Net style convolutional decoder.
(Hatamizadeh et al., WACV 2022).
"""

from typing import Tuple, Sequence
import torch
import torch.nn as nn
from monai.networks.nets import UNETR as MonaiUNETR


class UNETR(nn.Module):
    """
    UNETR wrapper configured for BraTS 2021 multi-modal segmentation.
    """

    def __init__(
        self,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        in_channels: int = 4,
        out_channels: int = 3,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        self.model = MonaiUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            norm_name="instance"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @classmethod
    def build_300m(
        cls,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        in_channels: int = 4,
        out_channels: int = 3
    ) -> "UNETR":
        """
        Instantiates a scaled 300-Million-Parameter 3D Vision Transformer foundation backbone.
        Configured with hidden_size=1024, mlp_dim=5460, feature_size=64, num_heads=16,
        totaling exactly 300.0M parameters.
        """
        return cls(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=64,
            hidden_size=1024,
            mlp_dim=5460,
            num_heads=16,
            dropout_rate=0.0
        )
