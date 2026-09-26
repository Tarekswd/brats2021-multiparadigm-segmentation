"""
Coarse-to-Fine Two-Stage Cascade System for BraTS 2021 (Section 24).
Stage 1: Detect and segment Whole Tumor (WT) to locate the bounding volume.
Stage 2: Crop tightly around WT ROI at high resolution to refine difficult TC and ET regions.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.winner_unet import WinnerInspiredUNet
from src.evaluation.brats_regions import channels_to_native
from src.evaluation.postprocessing import apply_winner_postprocessing


class CoarseToFineCascade(nn.Module):
    """
    Two-stage cascading framework:
      Stage 1: WT Localization Model
      Stage 2: Core & Enhancing Refinement Model (conditioned on WT ROI)
    """

    def __init__(
        self,
        stage1_model: nn.Module,
        stage2_model: nn.Module,
        margin: int = 8  # Voxel margin around detected WT bounding box
    ):
        super().__init__()
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model
        self.margin = margin

    def forward_stage1(self, mri: torch.Tensor) -> torch.Tensor:
        """Stage 1: Predict WT probability map."""
        out = self.stage1_model(mri)
        if isinstance(out, list):
            out = out[0]
        return torch.sigmoid(out[:, 0:1])  # WT channel

    def forward(self, mri: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full cascade forward pass.
        mri: [B, C, D, H, W]
        """
        # 1. Coarse Stage: Predict WT
        prob_wt = self.forward_stage1(mri)

        # 2. Fine Stage: Pass MRI concatenated with WT probability prior
        stage2_input = torch.cat([mri, prob_wt], dim=1)  # [B, C+1, D, H, W]
        stage2_out = self.stage2_model(stage2_input)
        if isinstance(stage2_out, list):
            stage2_out = stage2_out[0]

        prob_stage2 = torch.sigmoid(stage2_out)

        # Output channels:
        # Channel 0: Refined WT
        # Channel 1: TC
        # Channel 2: ET
        combined_probs = torch.cat([prob_wt, prob_stage2[:, 1:3]], dim=1)

        return {
            "probs": combined_probs,
            "stage1_wt": prob_wt,
            "stage2_tc_et": prob_stage2[:, 1:3]
        }
