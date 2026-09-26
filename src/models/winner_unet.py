"""
BraTS 2021 Winner-Inspired 3D U-Net Architecture.
Based on KAIST MRI Lab winning entry (Luu & Park, arXiv:2112.04653).

Key Innovations & Controlled Ablation Components:
  1. Asymmetrically Scaled Large Encoder (channels up to 512)
  2. Group Normalization (num_groups=32) for low-batch-size 3D stability
  3. Multi-Level Deep Supervision auxiliary segmentation heads
  4. 3D Axial Attention in the Decoder (independent attention along D, H, W axes)
  5. Configurable 4 or 5 channel inputs (optional brain foreground channel)
"""

from typing import List, Optional, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class AxialAttention3D(nn.Module):
    """
    3D Axial Attention module.
    Applies multi-head self-attention independently along each spatial axis (D, H, W),
    reducing complexity from O((DHW)^2) to O(DHW(D + H + W)).
    """

    def __init__(self, in_channels: int, num_heads: int = 4, dim_head: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim_head
        inner_dim = num_heads * dim_head

        self.to_qkv_d = nn.Conv3d(in_channels, inner_dim * 3, kernel_size=1, bias=False)
        self.to_qkv_h = nn.Conv3d(in_channels, inner_dim * 3, kernel_size=1, bias=False)
        self.to_qkv_w = nn.Conv3d(in_channels, inner_dim * 3, kernel_size=1, bias=False)

        self.proj = nn.Conv3d(inner_dim, in_channels, kernel_size=1)
        self.scale = dim_head ** -0.5

    def _axial_attn(self, qkv: torch.Tensor, axis: str, b: int, c: int, d: int, h: int, w: int) -> torch.Tensor:
        q, k, v = qkv.chunk(3, dim=1)
        h_heads = self.num_heads
        d_head = self.dim_head

        if axis == "d":
            # Reshape to (b * h * w, heads, d, dim_head)
            q = rearrange(q, "b (heads dhead) d h w -> (b h w) heads d dhead", heads=h_heads, dhead=d_head)
            k = rearrange(k, "b (heads dhead) d h w -> (b h w) heads d dhead", heads=h_heads, dhead=d_head)
            v = rearrange(v, "b (heads dhead) d h w -> (b h w) heads d dhead", heads=h_heads, dhead=d_head)
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = F.softmax(dots, dim=-1)
            out = torch.matmul(attn, v)
            out = rearrange(out, "(b h w) heads d dhead -> b (heads dhead) d h w", b=b, h=h, w=w)
        elif axis == "h":
            # Reshape to (b * d * w, heads, h, dim_head)
            q = rearrange(q, "b (heads dhead) d h w -> (b d w) heads h dhead", heads=h_heads, dhead=d_head)
            k = rearrange(k, "b (heads dhead) d h w -> (b d w) heads h dhead", heads=h_heads, dhead=d_head)
            v = rearrange(v, "b (heads dhead) d h w -> (b d w) heads h dhead", heads=h_heads, dhead=d_head)
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = F.softmax(dots, dim=-1)
            out = torch.matmul(attn, v)
            out = rearrange(out, "(b d w) heads h dhead -> b (heads dhead) d h w", b=b, d=d, w=w)
        else:  # 'w'
            # Reshape to (b * d * h, heads, w, dim_head)
            q = rearrange(q, "b (heads dhead) d h w -> (b d h) heads w dhead", heads=h_heads, dhead=d_head)
            k = rearrange(k, "b (heads dhead) d h w -> (b d h) heads w dhead", heads=h_heads, dhead=d_head)
            v = rearrange(v, "b (heads dhead) d h w -> (b d h) heads w dhead", heads=h_heads, dhead=d_head)
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = F.softmax(dots, dim=-1)
            out = torch.matmul(attn, v)
            out = rearrange(out, "(b d h) heads w dhead -> b (heads dhead) d h w", b=b, d=d, h=h)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        out_d = self._axial_attn(self.to_qkv_d(x), "d", b, c, d, h, w)
        out_h = self._axial_attn(self.to_qkv_h(x), "h", b, c, d, h, w)
        out_w = self._axial_attn(self.to_qkv_w(x), "w", b, c, d, h, w)
        out = self.proj(out_d + out_h + out_w)
        return x + out


class WinnerConvBlock(nn.Module):
    """
    Standard block for Winner U-Net: Conv3D -> Norm (GroupNorm or BatchNorm) -> LeakyReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "group",
        num_groups: int = 32
    ):
        super().__init__()
        if norm_type == "group":
            # Adjust groups if out_channels < num_groups
            g = num_groups if out_channels >= num_groups else out_channels
            norm_layer = lambda c: nn.GroupNorm(num_groups=g, num_channels=c)
        elif norm_type == "batch":
            norm_layer = lambda c: nn.BatchNorm3d(c)
        else:
            norm_layer = lambda c: nn.InstanceNorm3d(c)

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = norm_layer(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = norm_layer(out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm1(self.conv1(x)))
        x = self.act(self.norm2(self.conv2(x)))
        return x


class WinnerInspiredUNet(nn.Module):
    """
    Winner-Inspired 3D U-Net architecture for BraTS 2021.
    Supports all winning modifications and ablation configurations.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 3,
        use_large_encoder: bool = True,
        norm_type: str = "group",
        num_groups: int = 32,
        deep_supervision: bool = True,
        use_axial_attention: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.use_axial_attention = use_axial_attention

        # Channel configurations:
        # Large encoder: [64, 128, 256, 384, 512] vs Base: [32, 64, 128, 256, 320]
        if use_large_encoder:
            self.enc_channels = [64, 128, 256, 384, 512]
            self.dec_channels = [320, 256, 128, 64]
        else:
            self.enc_channels = [32, 64, 128, 256, 320]
            self.dec_channels = [256, 128, 64, 32]

        # Encoder path
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_c = in_channels
        for c in self.enc_channels:
            self.encoders.append(WinnerConvBlock(prev_c, c, norm_type=norm_type, num_groups=num_groups))
            self.pools.append(nn.MaxPool3d(2, 2))
            prev_c = c

        # Decoder path
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.axial_attns = nn.ModuleList() if use_axial_attention else None

        # Build decoders matching reversed encoder stages
        dec_in = self.enc_channels[-1]
        for i, dec_out in enumerate(self.dec_channels):
            skip_c = self.enc_channels[-(i + 2)]
            self.upsamples.append(nn.ConvTranspose3d(dec_in, dec_out, kernel_size=2, stride=2))

            if use_axial_attention and i < len(self.dec_channels) - 1:
                # Apply axial attention on lower-resolution decoder levels
                self.axial_attns.append(AxialAttention3D(dec_out, num_heads=4, dim_head=16))
            elif use_axial_attention:
                self.axial_attns.append(nn.Identity())

            self.decoders.append(WinnerConvBlock(dec_out + skip_c, dec_out, norm_type=norm_type, num_groups=num_groups))
            dec_in = dec_out

        # Final full-resolution segmentation head
        self.final_head = nn.Conv3d(self.dec_channels[-1], num_classes, kernel_size=1)

        # Deep supervision auxiliary heads
        if deep_supervision:
            self.aux_heads = nn.ModuleList([
                nn.Conv3d(c, num_classes, kernel_size=1) for c in self.dec_channels[:-1]
            ])
        else:
            self.aux_heads = None

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        skips = []
        for i in range(len(self.enc_channels) - 1):
            x = self.encoders[i](x)
            skips.append(x)
            x = self.pools[i](x)

        # Bottleneck
        x = self.encoders[-1](x)

        aux_outputs = []
        for i in range(len(self.decoders)):
            x = self.upsamples[i](x)
            if self.use_axial_attention and self.axial_attns is not None:
                x = self.axial_attns[i](x)

            skip = skips[-(i + 1)]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=True)
            x = torch.cat([skip, x], dim=1)
            x = self.decoders[i](x)

            if self.deep_supervision and i < len(self.decoders) - 1 and self.aux_heads is not None:
                aux_outputs.append(self.aux_heads[i](x))

        final_out = self.final_head(x)

        if self.training and self.deep_supervision:
            # Return list: [full_res, aux_level_1, aux_level_2, ...]
            return [final_out] + list(reversed(aux_outputs))
        return final_out
