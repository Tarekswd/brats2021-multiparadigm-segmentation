"""
Unified compound loss factory supporting:
  - Dice + BCE (BraTS 2021 Winning default)
  - Dice + Focal Loss
  - Compound Boundary Loss
  - Deep Supervision multi-scale loss aggregation
"""

from typing import Dict, Any, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.dice import DiceLoss
from src.losses.cross_entropy import BCELoss
from src.losses.focal import FocalLoss
from src.losses.boundary_loss import BoundaryLoss
from src.losses.region_loss import RegionLoss


class CompoundLoss(nn.Module):
    """
    Weighted combination of Dice, Cross-Entropy / Focal, and Boundary losses.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        focal_weight: float = 0.0,
        boundary_weight: float = 0.0,
        batch_dice: bool = True,
        region_weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight

        self.dice_loss = DiceLoss(batch_dice=batch_dice)
        self.bce_loss = BCELoss()
        self.focal_loss = FocalLoss() if focal_weight > 0 else None
        self.boundary_loss = BoundaryLoss() if boundary_weight > 0 else None
        self.region_weights = region_weights

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        logits: [B, C, D, H, W]
        target: [B, C, D, H, W]
        """
        prob = torch.sigmoid(logits)
        loss_dict = {}
        total = torch.tensor(0.0, device=logits.device)

        if self.dice_weight > 0:
            l_dice = self.dice_loss(prob, target)
            loss_dict["dice"] = l_dice
            total = total + self.dice_weight * l_dice

        if self.ce_weight > 0:
            l_ce = self.bce_loss(logits, target)
            loss_dict["ce"] = l_ce
            total = total + self.ce_weight * l_ce

        if self.focal_weight > 0 and self.focal_loss is not None:
            l_focal = self.focal_loss(logits, target)
            loss_dict["focal"] = l_focal
            total = total + self.focal_weight * l_focal

        if self.boundary_weight > 0 and self.boundary_loss is not None:
            l_bnd = self.boundary_loss(logits, target)
            loss_dict["boundary"] = l_bnd
            total = total + self.boundary_weight * l_bnd

        loss_dict["total"] = total
        return loss_dict


class DeepSupervisionLoss(nn.Module):
    """
    Wraps base loss to compute losses across multiple resolution decoder outputs.
    E.g., final_output (weight 0.5), aux1 (0.3), aux2 (0.2).
    """

    def __init__(self, base_loss: nn.Module, level_weights: Optional[List[float]] = None):
        super().__init__()
        self.base_loss = base_loss
        self.level_weights = level_weights if level_weights is not None else [0.5, 0.3, 0.2]

    def forward(
        self,
        preds: Union[torch.Tensor, List[torch.Tensor]],
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if isinstance(preds, torch.Tensor):
            return self.base_loss(preds, target)

        total_loss = torch.tensor(0.0, device=target.device)
        aggregated = {}

        # preds is ordered from full resolution down to lowest resolution
        for i, pred_level in enumerate(preds):
            w = self.level_weights[i] if i < len(self.level_weights) else (1.0 / len(preds))
            if pred_level.shape[2:] != target.shape[2:]:
                # Downsample target to match auxiliary head resolution
                target_down = F.interpolate(target, size=pred_level.shape[2:], mode="nearest")
            else:
                target_down = target

            level_dict = self.base_loss(pred_level, target_down)
            total_loss = total_loss + w * level_dict["total"]

        aggregated["total"] = total_loss
        return aggregated


def build_loss(cfg: Dict[str, Any]) -> nn.Module:
    """Build loss function based on YAML configuration."""
    loss_cfg = cfg.get("loss", {})
    name = loss_cfg.get("name", "dice_ce").lower()
    batch_dice = loss_cfg.get("batch_dice", True)
    deep_sup = loss_cfg.get("deep_supervision", False)
    ds_weights = loss_cfg.get("deep_supervision_weights", [0.5, 0.3, 0.2])

    if name == "dice_ce":
        base = CompoundLoss(dice_weight=1.0, ce_weight=1.0, batch_dice=batch_dice)
    elif name == "dice_focal":
        base = CompoundLoss(dice_weight=1.0, ce_weight=0.0, focal_weight=1.0, batch_dice=batch_dice)
    elif name == "dice_ce_boundary":
        base = CompoundLoss(dice_weight=1.0, ce_weight=1.0, boundary_weight=0.5, batch_dice=batch_dice)
    elif name == "dice_only":
        base = CompoundLoss(dice_weight=1.0, ce_weight=0.0, batch_dice=batch_dice)
    else:
        base = CompoundLoss(dice_weight=1.0, ce_weight=1.0, batch_dice=batch_dice)

    if deep_sup:
        return DeepSupervisionLoss(base, level_weights=ds_weights)
    return base
