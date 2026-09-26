"""
Model architecture factory for BraTS 2021 Brain Tumor Segmentation.
"""

from typing import Dict, Any
import torch.nn as nn

from src.models.unet3d import UNet3D
from src.models.nnunet_wrapper import NnUNetBaseline
from src.models.winner_unet import WinnerInspiredUNet
from src.models.segresnet import SegResNet
from src.models.unetr import UNETR
from src.models.swin_unetr import SwinUNETR


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """Instantiate model based on YAML config dictionary."""
    model_cfg = cfg.get("model", {})
    name = model_cfg.get("name", "winner_unet").lower()
    in_channels = model_cfg.get("in_channels", 4)
    num_classes = model_cfg.get("num_classes", 3)

    if name in ["unet3d", "baseline_unet", "standard_unet"]:
        channels = model_cfg.get("channels", [32, 64, 128, 256, 320])
        norm_type = model_cfg.get("norm_type", "instance")
        return UNet3D(in_channels=in_channels, num_classes=num_classes, channels=channels, norm_type=norm_type)

    elif name in ["nnunet", "nnunet_baseline"]:
        deep_sup = model_cfg.get("deep_supervision", True)
        return NnUNetBaseline(in_channels=in_channels, num_classes=num_classes, deep_supervision=deep_sup)

    elif name in ["winner_unet", "winner_inspired_unet", "winner"]:
        use_large = model_cfg.get("use_large_encoder", True)
        norm_type = model_cfg.get("norm_type", "group")
        num_groups = model_cfg.get("num_groups", 32)
        deep_sup = model_cfg.get("deep_supervision", True)
        axial_attn = model_cfg.get("axial_attention", False)
        return WinnerInspiredUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            use_large_encoder=use_large,
            norm_type=norm_type,
            num_groups=num_groups,
            deep_supervision=deep_sup,
            use_axial_attention=axial_attn
        )

    elif name in ["segresnet", "monai_segresnet"]:
        init_filters = model_cfg.get("init_filters", 16)
        dropout_prob = model_cfg.get("dropout_prob", 0.2)
        return SegResNet(in_channels=in_channels, out_channels=num_classes, init_filters=init_filters, dropout_prob=dropout_prob)

    elif name in ["swin_unetr", "swinunetr"]:
        img_size = tuple(model_cfg.get("img_size", [96, 96, 96]))
        feature_size = model_cfg.get("feature_size", 24)
        return SwinUNETR(img_size=img_size, in_channels=in_channels, out_channels=num_classes, feature_size=feature_size)

    elif name in ["unetr"]:
        img_size = tuple(model_cfg.get("img_size", [96, 96, 96]))
        feature_size = model_cfg.get("feature_size", 16)
        hidden_size = model_cfg.get("hidden_size", 768)
        return UNETR(img_size=img_size, in_channels=in_channels, out_channels=num_classes, feature_size=feature_size, hidden_size=hidden_size)

    else:
        raise ValueError(f"Unknown model name: {name}")
