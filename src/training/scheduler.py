"""
Learning rate schedulers for BraTS 2021 training.
Implements the polynomial learning rate decay used by nnU-Net and the winning KAIST system:
  lr = base_lr * (1 - epoch / max_epochs) ** 0.9
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler


class PolynomialLR(_LRScheduler):
    """
    Polynomial learning rate decay scheduler.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_epochs: int = 1000,
        power: float = 0.9,
        min_lr: float = 1e-6,
        last_epoch: int = -1
    ):
        self.max_epochs = max_epochs
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = max(0.0, 1.0 - (self.last_epoch / max(1, self.max_epochs))) ** self.power
        return [max(self.min_lr, base_lr * factor) for base_lr in self.base_lrs]
