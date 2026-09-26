"""
Standard 3D U-Net Baseline for BraTS 2021 Brain Tumor Segmentation.
Follows clean encoder-decoder topology with skip connections and configurable depth/channels.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """Two consecutive 3D convs with normalization and LeakyReLU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "instance"
    ):
        super().__init__()
        norm_layer = nn.InstanceNorm3d if norm_type == "instance" else nn.BatchNorm3d
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = norm_layer(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = norm_layer(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x


class UNet3D(nn.Module):
    """
    Standard 3D U-Net baseline architecture.
    Inputs: [B, C_in, D, H, W] (e.g. C_in=4 for T1, T1ce, T2, FLAIR)
    Outputs: [B, num_classes, D, H, W] (e.g. num_classes=3 for WT, TC, ET)
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 3,
        channels: Optional[List[int]] = None,
        norm_type: str = "instance"
    ):
        super().__init__()
        self.channels = channels if channels is not None else [32, 64, 128, 256, 320]
        self.depth = len(self.channels)

        # Encoders
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_c = in_channels
        for c in self.channels:
            self.encoders.append(ConvBlock3D(prev_c, c, norm_type=norm_type))
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            prev_c = c

        # Decoders
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        rev_channels = list(reversed(self.channels))
        for i in range(len(rev_channels) - 1):
            curr_c = rev_channels[i]
            next_c = rev_channels[i + 1]
            self.upsamples.append(nn.ConvTranspose3d(curr_c, next_c, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock3D(next_c * 2, next_c, norm_type=norm_type))

        # Final 1x1x1 segmentation projection
        self.final_conv = nn.Conv3d(self.channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i in range(self.depth - 1):
            x = self.encoders[i](x)
            skips.append(x)
            x = self.pools[i](x)

        # Bottleneck
        x = self.encoders[-1](x)

        # Decoder path
        for i in range(len(self.decoders)):
            x = self.upsamples[i](x)
            skip = skips[-(i + 1)]
            # Align spatial sizes if needed
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=True)
            x = torch.cat([skip, x], dim=1)
            x = self.decoders[i](x)

        logits = self.final_conv(x)
        return logits
