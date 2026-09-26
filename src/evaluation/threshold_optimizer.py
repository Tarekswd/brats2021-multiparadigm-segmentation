"""
Dynamic Per-Region Probability Threshold Optimizer for BraTS 2021 (Section 35).
Searches for optimal thresholds (theta_wt, theta_tc, theta_et) strictly using validation data.
"""

from typing import List, Tuple, Dict
import numpy as np

from src.evaluation.dice import compute_dice
from src.evaluation.brats_regions import channels_to_native
from src.evaluation.postprocessing import apply_winner_postprocessing


def optimize_region_thresholds(
    val_probs: List[np.ndarray],   # Each [3, D, H, W]
    val_targets: List[np.ndarray], # Each [3, D, H, W]
    wt_grid: np.ndarray = np.linspace(0.40, 0.60, 5),
    tc_grid: np.ndarray = np.linspace(0.40, 0.60, 5),
    et_grid: np.ndarray = np.linspace(0.45, 0.65, 5),
    et_min_voxels: int = 200
) -> Dict[str, float]:
    """
    Find optimal per-region decision thresholds on out-of-fold validation data.
    """
    best_thresholds = {"wt": 0.5, "tc": 0.5, "et": 0.5}
    best_dices = {"wt": 0.0, "tc": 0.0, "et": 0.0}

    # 1. Optimize WT threshold
    for t_wt in wt_grid:
        d_sum = 0.0
        for p, g in zip(val_probs, val_targets):
            pred_wt = (p[0] >= t_wt).astype(float)
            d_sum += compute_dice(pred_wt, g[0])
        score = d_sum / len(val_probs)
        if score > best_dices["wt"]:
            best_dices["wt"] = score
            best_thresholds["wt"] = float(t_wt)

    # 2. Optimize TC threshold
    for t_tc in tc_grid:
        d_sum = 0.0
        for p, g in zip(val_probs, val_targets):
            pred_tc = (p[1] >= t_tc).astype(float)
            d_sum += compute_dice(pred_tc, g[1])
        score = d_sum / len(val_probs)
        if score > best_dices["tc"]:
            best_dices["tc"] = score
            best_thresholds["tc"] = float(t_tc)

    # 3. Optimize ET threshold
    for t_et in et_grid:
        d_sum = 0.0
        for p, g in zip(val_probs, val_targets):
            pred_et = (p[2] >= t_et).astype(float)
            # Apply ET voxel volume threshold
            if 0 < np.sum(pred_et) < et_min_voxels:
                pred_et[:] = 0.0
            d_sum += compute_dice(pred_et, g[2])
        score = d_sum / len(val_probs)
        if score > best_dices["et"]:
            best_dices["et"] = score
            best_thresholds["et"] = float(t_et)

    return best_thresholds
