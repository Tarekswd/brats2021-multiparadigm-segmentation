"""
Post-processing and Test-Time Augmentation (TTA) pipeline for BraTS 2021.
Implements the crucial winning postprocessing rule (Luu & Park 2021):
  - ET thresholding: If count(ET) < 200 voxels, reassign ET -> necrotic core (NCR, label 1)
  - Strict anatomical consistency enforcement: ET ⊆ TC ⊆ WT
  - Small connected component removal
"""

from typing import Tuple, List, Optional
import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn as nn


def apply_winner_postprocessing(
    native_mask: np.ndarray,
    et_min_voxels: int = 200,
    min_component_size: int = 10
) -> np.ndarray:
    """
    Apply BraTS 2021 winning post-processing rules.
    native_mask: 3D array with labels in {0, 1, 2, 4}
    """
    out = native_mask.copy()

    # Rule 1: ET voxel count threshold (KAIST winning rule)
    et_count = np.sum(out == 4)
    if 0 < et_count < et_min_voxels:
        # Reassign small spurious ET voxels to necrotic core (1)
        out[out == 4] = 1

    # Rule 2: Remove isolated tiny disconnected fragments (< min_component_size)
    if min_component_size > 0:
        wt_mask = out > 0
        labeled_wt, num_comp = ndi.label(wt_mask)
        if num_comp > 0:
            sizes = ndi.sum(wt_mask, labeled_wt, range(1, num_comp + 1))
            small_comps = [idx + 1 for idx, sz in enumerate(sizes) if sz < min_component_size]
            for comp in small_comps:
                out[labeled_wt == comp] = 0

    return out


def enforce_anatomical_consistency(
    prob_channels: np.ndarray,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Enforces strict subset hierarchy on binary masks:
      ET ⊆ TC ⊆ WT
    prob_channels: [3, D, H, W] for WT, TC, ET probabilities
    """
    wt = prob_channels[0] >= threshold
    tc = prob_channels[1] >= threshold
    et = prob_channels[2] >= threshold

    # Project to satisfy nested subsets
    tc_consistent = tc & wt
    et_consistent = et & tc_consistent

    return wt, tc_consistent, et_consistent


def predict_with_tta(
    model: nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Test-Time Augmentation (TTA) using 3D spatial axis reflections.
    image_tensor: [1, C, D, H, W]
    Returns averaged sigmoid probability map: [1, 3, D, H, W]
    """
    model.eval()
    predictions = []

    # Flips to execute: (None, D, H, W)
    flip_dims_list = [
        None,
        (2,),       # Flip D
        (3,),       # Flip H
        (4,),       # Flip W
        (3, 4),     # Flip H, W
    ]

    with torch.no_grad():
        for dims in flip_dims_list:
            if dims is None:
                inp = image_tensor.to(device)
            else:
                inp = torch.flip(image_tensor, dims=dims).to(device)

            out = model(inp)
            if isinstance(out, list):
                out = out[0]
            prob = torch.sigmoid(out)

            # Invert transform
            if dims is not None:
                prob = torch.flip(prob, dims=dims)

            predictions.append(prob)

    avg_prob = torch.mean(torch.stack(predictions, dim=0), dim=0)
    return avg_prob
