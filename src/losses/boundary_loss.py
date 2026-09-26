"""
Boundary / Surface distance loss for sharp boundary delineation in brain tumor MRI.
Approximates surface distance using spatial gradient magnitude differences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryLoss(nn.Module):
    """
    Differentiable boundary approximation loss based on 3D spatial Sobel/Laplacian gradients.
    Penalizes discrepancies specifically at the tumor contour boundaries.
    """

    def __init__(self):
        super().__init__()
        # 3D spatial gradient filters
        kx = torch.tensor([[-1, 0, 1]], dtype=torch.float32).view(1, 1, 1, 1, 3)
        ky = torch.tensor([[[-1], [0], [1]]], dtype=torch.float32).view(1, 1, 1, 3, 1)
        kz = torch.tensor([[[[-1]], [[0]], [[1]]]], dtype=torch.float32).view(1, 1, 3, 1, 1)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
        self.register_buffer("kz", kz)

    def _gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        x_reshaped = x.view(b * c, 1, d, h, w)
        gx = F.conv3d(x_reshaped, self.kx, padding=(0, 0, 1))
        gy = F.conv3d(x_reshaped, self.ky, padding=(0, 1, 0))
        gz = F.conv3d(x_reshaped, self.kz, padding=(1, 0, 0))
        mag = torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2 + 1e-6)
        return mag.view(b, c, d, h, w)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        grad_pred = self._gradient_magnitude(prob)
        grad_target = self._gradient_magnitude(target.float())
        return F.l1_loss(grad_pred, grad_target)
