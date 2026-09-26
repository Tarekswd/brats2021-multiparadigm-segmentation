"""
Sensitivity (True Positive Rate) and Specificity (True Negative Rate) for BraTS 2021 regions.
"""

from typing import Union, Tuple, Dict
import numpy as np


def compute_sensitivity(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """True Positive Rate: TP / (TP + FN)."""
    p = np.asarray(pred_mask, dtype=bool)
    g = np.asarray(gt_mask, dtype=bool)
    g_sum = np.sum(g)
    if g_sum == 0:
        return 1.0 if np.sum(p) == 0 else 0.0
    tp = np.sum(p & g)
    return float(tp / g_sum)


def compute_specificity(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """True Negative Rate: TN / (TN + FP)."""
    p = np.asarray(pred_mask, dtype=bool)
    g = np.asarray(gt_mask, dtype=bool)
    tn = np.sum((~p) & (~g))
    fp = np.sum(p & (~g))
    denom = tn + fp
    return float(tn / denom) if denom > 0 else 1.0


def compute_brats_sensitivity_specificity(
    pred_regions: Tuple[np.ndarray, np.ndarray, np.ndarray],
    gt_regions: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> Dict[str, float]:
    p_wt, p_tc, p_et = pred_regions
    g_wt, g_tc, g_et = gt_regions

    return {
        "sens_wt": compute_sensitivity(p_wt, g_wt),
        "sens_tc": compute_sensitivity(p_tc, g_tc),
        "sens_et": compute_sensitivity(p_et, g_et),
        "spec_wt": compute_specificity(p_wt, g_wt),
        "spec_tc": compute_specificity(p_tc, g_tc),
        "spec_et": compute_specificity(p_et, g_et),
    }
