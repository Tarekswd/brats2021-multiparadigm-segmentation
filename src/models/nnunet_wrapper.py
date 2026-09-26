"""
nnU-Net Baseline Architecture Wrapper for BraTS 2021.
Replicates the self-configuring 3D fullres nnU-Net architecture from BraTS 2020/2021
(Isensee et al., Nature Methods 2021).
"""

from typing import List, Optional, Union
import torch
import torch.nn as nn
from src.models.winner_unet import WinnerInspiredUNet


class NnUNetBaseline(nn.Module):
    """
    nnU-Net 3D fullres baseline wrapper.
    Symmetric encoder-decoder: channels [32, 64, 128, 256, 320].
    Deep supervision at decoder levels.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 3,
        deep_supervision: bool = True
    ):
        super().__init__()
        # nnU-Net baseline corresponds to WinnerInspiredUNet with:
        # use_large_encoder=False, norm_type='instance' (or 'batch'), use_axial_attention=False
        self.model = WinnerInspiredUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            use_large_encoder=False,
            norm_type="instance",
            deep_supervision=deep_supervision,
            use_axial_attention=False
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        return self.model(x)
