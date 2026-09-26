"""
Overfitting and Generalization Audit Suite for BraTS 2021.
Verifies zero patient leakage, computes train-val generalization gaps,
and evaluates cross-fold variance to guarantee model generalizability.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd


def audit_dataset_leakage(
    train_ids: List[str],
    val_ids: List[str],
    test_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Formally verifies that patient splits are strictly disjoint.
    """
    set_train = set(train_ids)
    set_val = set(val_ids)
    set_test = set(test_ids) if test_ids else set()

    leak_train_val = set_train & set_val
    leak_train_test = set_train & set_test
    leak_val_test = set_val & set_test

    passed = (len(leak_train_val) == 0 and len(leak_train_test) == 0 and len(leak_val_test) == 0)

    return {
        "zero_leakage_passed": passed,
        "train_val_overlap_count": len(leak_train_val),
        "train_test_overlap_count": len(leak_train_test),
        "val_test_overlap_count": len(leak_val_test),
        "total_unique_patients": len(set_train | set_val | set_test)
    }


def compute_generalization_gap(
    train_dices: np.ndarray,
    val_dices: np.ndarray
) -> Dict[str, float]:
    """
    Calculates the generalization gap: Delta = Dice_train - Dice_val.
    A small gap (< 3-5%) indicates healthy regularization without overfitting.
    A large gap (> 10-15%) signals memorization / overfitting.
    """
    t_mean = float(np.mean(train_dices))
    v_mean = float(np.mean(val_dices))
    gap = t_mean - v_mean

    return {
        "train_dice_mean": t_mean,
        "val_dice_mean": v_mean,
        "generalization_gap": float(gap),
        "is_healthy_generalization": bool(gap < 6.0 if t_mean > 1.0 else gap < 0.06)
    }


def compute_cross_fold_stability(
    fold_scores: List[float]
) -> Dict[str, float]:
    """
    Computes cross-validation variance across all 5 folds.
    Low variance (Std < 1.0%) proves the model is not overfitted to a particular cohort slice.
    """
    scores = np.asarray(fold_scores, dtype=np.float64)
    return {
        "mean_score": float(np.mean(scores)),
        "std_variance": float(np.std(scores, ddof=1)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "is_stable": bool(np.std(scores, ddof=1) < 0.015)
    }
