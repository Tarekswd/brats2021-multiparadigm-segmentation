"""
BraTS 2021 Dice Similarity Coefficient (DSC) evaluation metric.
Supports strict empty-region handling according to official BraTS benchmark rules:
  - Both GT and Pred empty -> Dice = 1.0 (True Negative)
  - One empty and other non-empty -> Dice = 0.0 (False Positive or False Negative)
"""

import numpy as np
import torch
from typing import Union, Tuple, Dict


def compute_dice(
    pred_mask: Union[np.ndarray, torch.Tensor],
    gt_mask: Union[np.ndarray, torch.Tensor],
    empty_score: float = 1.0
) -> float:
    """
    Compute Dice coefficient between two binary 3D masks.
    """
    if isinstance(pred_mask, torch.Tensor):
        p = pred_mask.detach().cpu().numpy().astype(bool)
    else:
        p = np.asarray(pred_mask, dtype=bool)

    if isinstance(gt_mask, torch.Tensor):
        g = gt_mask.detach().cpu().numpy().astype(bool)
    else:
        g = np.asarray(gt_mask, dtype=bool)

    p_sum = np.sum(p)
    g_sum = np.sum(g)

    if p_sum == 0 and g_sum == 0:
        return empty_score
    if p_sum == 0 or g_sum == 0:
        return 0.0

    intersection = np.sum(p & g)
    return float(2.0 * intersection / (p_sum + g_sum))


def compute_brats_dices(
    pred_regions: Tuple[np.ndarray, np.ndarray, np.ndarray],
    gt_regions: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> Dict[str, float]:
    """
    Compute WT, TC, and ET Dice scores.
    pred_regions: (pred_wt, pred_tc, pred_et)
    gt_regions: (gt_wt, gt_tc, gt_et)
    """
    p_wt, p_tc, p_et = pred_regions
    g_wt, g_tc, g_et = gt_regions

    d_wt = compute_dice(p_wt, g_wt)
    d_tc = compute_dice(p_tc, g_tc)
    d_et = compute_dice(p_et, g_et)
    d_avg = (d_wt + d_tc + d_et) / 3.0

    return {
        "dice_wt": d_wt,
        "dice_tc": d_tc,
        "dice_et": d_et,
        "dice_avg": d_avg
    }
