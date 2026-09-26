"""
Swin UNETR Architecture for BraTS 2021.
Hierarchical Shifted-Window 3D Vision Transformer combined with U-Net CNN Decoder.
(Hatamizadeh et al., MONAI Consortium).
"""

from typing import Tuple, Sequence, Optional
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR as MonaiSwinUNETR


class SwinUNETR(nn.Module):
    """
    Swin UNETR wrapper configured for BraTS 2021.
    Processes multi-modal 3D MRI volumes with shifted-window self-attention.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        feature_size: int = 24,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        norm_name: str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        img_size: Optional[Tuple[int, int, int]] = None
    ):
        super().__init__()
        self.model = MonaiSwinUNETR(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            depths=depths,
            num_heads=num_heads,
            norm_name=norm_name,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            use_checkpoint=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
