"""
Training callbacks for BraTS 2021 experiments.
Supports early stopping and metric thresholds.
"""

from typing import Optional


class EarlyStopping:
    """Early stops training if monitored validation metric fails to improve."""

    def __init__(self, patience: int = 30, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def step(self, current_score: float) -> bool:
        """Returns True if score is an improvement; updates counter otherwise."""
        if self.best_score is None:
            self.best_score = current_score
            return True

        if self.mode == "max":
            improved = current_score > (self.best_score + self.min_delta)
        else:
            improved = current_score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
