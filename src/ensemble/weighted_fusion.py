"""
Weighted fusion ensemble for BraTS 2021 predictions.
Optimizes weights w_m strictly on validation out-of-fold predictions using constrained grid search.
"""

from typing import List, Tuple, Union, Optional
import numpy as np
import torch
from scipy.optimize import minimize

from src.evaluation.dice import compute_dice


def weighted_average_ensemble(
    prob_maps: List[np.ndarray],
    weights: List[float]
) -> np.ndarray:
    """
    Compute weighted average of probability maps.
    prob_maps: list of M arrays of shape [3, D, H, W]
    weights: list of M scalar weights (sum to 1.0)
    """
    assert len(prob_maps) == len(weights), "Number of probability maps and weights must match"
    w_norm = np.array(weights, dtype=np.float32) / sum(weights)

    out = np.zeros_like(prob_maps[0], dtype=np.float32)
    for m in range(len(prob_maps)):
        out += w_norm[m] * prob_maps[m]

    return out


def optimize_ensemble_weights(
    val_model_preds: List[List[np.ndarray]],
    val_targets: List[np.ndarray],
    grid_resolution: int = 11
) -> List[float]:
    """
    Find optimal ensemble weights strictly using validation data.
    val_model_preds: list of M models, where each model has N patient probability maps [3, D, H, W]
    val_targets: list of N patient ground truth binary targets [3, D, H, W]
    """
    num_models = len(val_model_preds)
    num_patients = len(val_targets)

    if num_models == 1:
        return [1.0]

    best_weights = [1.0 / num_models] * num_models
    best_score = -1.0

    # Grid search for 2 or 3 models
    if num_models == 2:
        for w1 in np.linspace(0.0, 1.0, grid_resolution):
            w2 = 1.0 - w1
            weights = [w1, w2]
            score = _eval_weights(weights, val_model_preds, val_targets)
            if score > best_score:
                best_score = score
                best_weights = weights

    elif num_models == 3:
        for w1 in np.linspace(0.0, 1.0, grid_resolution):
            for w2 in np.linspace(0.0, 1.0 - w1, grid_resolution):
                w3 = max(0.0, 1.0 - (w1 + w2))
                weights = [w1, w2, w3]
                score = _eval_weights(weights, val_model_preds, val_targets)
                if score > best_score:
                    best_score = score
                    best_weights = weights
    else:
        # Fallback to equal weighting if > 3 models
        best_weights = [1.0 / num_models] * num_models

    return [float(w) for w in best_weights]


def _eval_weights(
    weights: List[float],
    val_model_preds: List[List[np.ndarray]],
    val_targets: List[np.ndarray]
) -> float:
    total_dice = 0.0
    for i in range(len(val_targets)):
        patient_preds = [val_model_preds[m][i] for m in range(len(weights))]
        fused = weighted_average_ensemble(patient_preds, weights)
        bin_fused = (fused >= 0.5).astype(float)
        # Avg dice over WT, TC, ET
        d_wt = compute_dice(bin_fused[0], val_targets[i][0])
        d_tc = compute_dice(bin_fused[1], val_targets[i][1])
        d_et = compute_dice(bin_fused[2], val_targets[i][2])
        total_dice += (d_wt + d_tc + d_et) / 3.0
    return total_dice / max(1, len(val_targets))
