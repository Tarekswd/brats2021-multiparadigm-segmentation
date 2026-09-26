"""
Reproducibility utilities for BraTS 2021 experiments.
Ensures deterministic operations across Python, NumPy, PyTorch, and CUDA.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """
    Set seeds across all random number generators.
    Enforces deterministic algorithms in PyTorch when specified.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            pass

    return seed
