"""
3D Data Augmentation for BraTS 2021 MRI volumes.
Supports:
  - Random 3D spatial flips along all axes
  - Random 90-degree rotations
  - Random intensity contrast and gamma adjustments
  - Non-destructive nearest-neighbor preservation for segmentation masks
"""

import random
import numpy as np
import torch
from typing import Tuple, Optional


class BraTSAugmentor:
    """Configurable 3D augmentation pipeline for multi-channel MRI and segmentation masks."""

    def __init__(
        self,
        flip_prob: float = 0.5,
        rot90_prob: float = 0.3,
        gamma_prob: float = 0.3,
        gamma_range: Tuple[float, float] = (0.7, 1.5)
    ):
        self.flip_prob = flip_prob
        self.rot90_prob = rot90_prob
        self.gamma_prob = gamma_prob
        self.gamma_range = gamma_range

    def __call__(
        self,
        image: np.ndarray,
        seg: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        image shape: [C, D, H, W]
        seg shape: [D, H, W] or [3, D, H, W] (region channels)
        """
        # 1. Random spatial flips (axes: D=1, H=2, W=3 in image)
        for axis_idx in (1, 2, 3):
            if random.random() < self.flip_prob:
                image = np.flip(image, axis=axis_idx)
                if seg is not None:
                    seg_axis = axis_idx if seg.ndim == 4 else (axis_idx - 1)
                    seg = np.flip(seg, axis=seg_axis)

        # 2. Random 90-degree rotations in axial plane (H, W)
        if random.random() < self.rot90_prob:
            k = random.choice([1, 2, 3])
            image = np.rot90(image, k=k, axes=(2, 3))
            if seg is not None:
                seg_axes = (2, 3) if seg.ndim == 4 else (1, 2)
                seg = np.rot90(seg, k=k, axes=seg_axes)

        # 3. Random Gamma Adjustment per MRI modality
        if random.random() < self.gamma_prob:
            gamma = random.uniform(*self.gamma_range)
            # Only apply to MRI channels (first 4 channels, excluding binary foreground channel if present)
            num_mri_channels = min(4, image.shape[0])
            for c in range(num_mri_channels):
                c_min, c_max = np.min(image[c]), np.max(image[c])
                if c_max > c_min:
                    normalized = (image[c] - c_min) / (c_max - c_min + 1e-7)
                    image[c] = (normalized ** gamma) * (c_max - c_min) + c_min

        # Ensure contiguous memory layout
        image = np.ascontiguousarray(image)
        if seg is not None:
            seg = np.ascontiguousarray(seg)

        return image, seg
