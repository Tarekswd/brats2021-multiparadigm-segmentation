"""
Voting ensemble strategies for BraTS 2021 (Hard and Soft Voting).
"""

from typing import List
import numpy as np


def hard_voting_ensemble(
    binary_masks: List[np.ndarray],
    threshold_ratio: float = 0.5
) -> np.ndarray:
    """
    Hard majority voting across M binary segmentations [3, D, H, W].
    Voxel is positive if at least ceil(M * threshold_ratio) models vote positive.
    """
    stacked = np.stack(binary_masks, axis=0)  # [M, 3, D, H, W]
    vote_counts = np.sum(stacked > 0, axis=0)
    min_votes = int(np.ceil(len(binary_masks) * threshold_ratio))
    return (vote_counts >= min_votes).astype(np.uint8)


def soft_voting_ensemble(prob_maps: List[np.ndarray], threshold: float = 0.5) -> np.ndarray:
    """
    Soft voting: average probabilities and binarize at threshold.
    """
    mean_prob = np.mean(np.stack(prob_maps, axis=0), axis=0)
    return (mean_prob >= threshold).astype(np.uint8)
