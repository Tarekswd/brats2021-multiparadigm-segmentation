"""
BraTS 2021 Tumor Subregion Definitions and Conversions.

Native BraTS labels:
  0 = Background
  1 = Necrotic / Non-enhancing tumor (NCR / NET)
  2 = Peritumoral Edema (ED)
  4 = GD-Enhancing Tumor (ET)

Derived BraTS Evaluation Regions:
  WT (Whole Tumor) = labels {1, 2, 4}
  TC (Tumor Core)  = labels {1, 4}
  ET (Enhancing)   = label {4}
"""

import numpy as np
import torch
from typing import Tuple, Union


def native_to_regions(
    label_vol: Union[np.ndarray, torch.Tensor]
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Convert a 3D volume with native labels {0, 1, 2, 4} into binary masks for WT, TC, and ET.
    """
    if isinstance(label_vol, torch.Tensor):
        wt = (label_vol == 1) | (label_vol == 2) | (label_vol == 4)
        tc = (label_vol == 1) | (label_vol == 4)
        et = (label_vol == 4)
        return wt.float(), tc.float(), et.float()
    else:
        label_vol = np.asarray(label_vol)
        wt = np.isin(label_vol, [1, 2, 4]).astype(np.float32)
        tc = np.isin(label_vol, [1, 4]).astype(np.float32)
        et = (label_vol == 4).astype(np.float32)
        return wt, tc, et


def regions_to_channels(
    label_vol: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert native label volume [D, H, W] to 3-channel binary target [3, D, H, W]
    Channel 0: WT
    Channel 1: TC
    Channel 2: ET
    """
    wt, tc, et = native_to_regions(label_vol)
    if isinstance(label_vol, torch.Tensor):
        return torch.stack([wt, tc, et], dim=0)
    else:
        return np.stack([wt, tc, et], axis=0)


def channels_to_native(
    prob_channels: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5,
    enforce_hierarchy: bool = True
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert 3-channel sigmoid probabilities [3, D, H, W] back to native labels {0, 1, 2, 4}.
    Channel 0: WT prob
    Channel 1: TC prob
    Channel 2: ET prob

    Hierarchical decoding ensures ET ⊆ TC ⊆ WT:
      Background (0) if WT < threshold
      Edema (2) if WT >= threshold and TC < threshold
      Necrotic Core (1) if TC >= threshold and ET < threshold
      Enhancing Tumor (4) if ET >= threshold (and optionally TC/WT >= threshold)
    """
    is_torch = isinstance(prob_channels, torch.Tensor)
    if is_torch:
        p = prob_channels.detach().cpu().numpy()
    else:
        p = np.asarray(prob_channels)

    wt = p[0] >= threshold
    tc = p[1] >= threshold
    et = p[2] >= threshold

    if enforce_hierarchy:
        # Enforce ET subset of TC subset of WT
        tc = tc & wt
        et = et & tc

    native = np.zeros(p.shape[1:], dtype=np.uint8)
    # Where WT is positive: default to edema (2)
    native[wt] = 2
    # Where TC is positive: set to necrotic core (1)
    native[tc] = 1
    # Where ET is positive: set to enhancing tumor (4)
    native[et] = 4

    if is_torch:
        return torch.from_numpy(native).to(prob_channels.device)
    return native
