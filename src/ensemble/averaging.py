"""
Simple probability averaging ensemble for BraTS 2021 multi-model predictions.
P_final = (1 / M) * sum(P_m)
"""

from typing import List, Union
import numpy as np
import torch


def simple_average_ensemble(
    prob_maps: List[Union[np.ndarray, torch.Tensor]]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Averages probability maps across M models.
    Each map has shape [3, D, H, W] (WT, TC, ET probabilities).
    """
    if not prob_maps:
        raise ValueError("prob_maps list must not be empty")

    is_torch = isinstance(prob_maps[0], torch.Tensor)
    if is_torch:
        stacked = torch.stack(prob_maps, dim=0)
        return torch.mean(stacked, dim=0)
    else:
        stacked = np.stack(prob_maps, axis=0)
        return np.mean(stacked, axis=0)
