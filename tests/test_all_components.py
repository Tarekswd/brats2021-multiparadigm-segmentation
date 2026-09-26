"""
Comprehensive unit test suite for BraTS 2021 Research System (Section 46).
Tests dataset loading, conversions, metrics, augmentations, ensembling, postprocessing, and models.
"""

import tempfile
import pytest
import numpy as np
import torch

from src.evaluation.brats_regions import native_to_regions, regions_to_channels, channels_to_native
from src.evaluation.dice import compute_dice, compute_brats_dices
from src.evaluation.hd95 import compute_hd95
from src.evaluation.sensitivity import compute_sensitivity, compute_specificity
from src.data.augmentation import BraTSAugmentor
from src.data.preprocessing import load_nifti, normalize_intensity
from src.ensemble.averaging import simple_average_ensemble
from src.ensemble.weighted_fusion import weighted_average_ensemble
from src.evaluation.postprocessing import apply_winner_postprocessing, enforce_anatomical_consistency
from src.models.unet3d import UNet3D
from src.models.winner_unet import WinnerInspiredUNet
from src.models.segresnet import SegResNet
from tests.create_mock_data import generate_mock_subject


def test_label_conversion_and_regions():
    """Verify native labels {0, 1, 2, 4} properly map to WT, TC, ET."""
    seg = np.zeros((10, 10, 10), dtype=np.uint8)
    seg[2:8, 2:8, 2:8] = 2  # ED
    seg[4:6, 4:6, 4:6] = 1  # NCR
    seg[5, 5, 5] = 4        # ET

    wt, tc, et = native_to_regions(seg)
    # WT = 1, 2, 4
    assert np.all(wt[seg == 2] == 1.0)
    assert np.all(wt[seg == 1] == 1.0)
    assert np.all(wt[seg == 4] == 1.0)
    assert np.all(wt[seg == 0] == 0.0)

    # TC = 1, 4
    assert np.all(tc[seg == 1] == 1.0)
    assert np.all(tc[seg == 4] == 1.0)
    assert np.all(tc[seg == 2] == 0.0)

    # ET = 4
    assert et[5, 5, 5] == 1.0
    assert np.sum(et) == 1

    # Test reverse conversion to native
    channels = regions_to_channels(seg)
    assert channels.shape == (3, 10, 10, 10)
    reconstructed = channels_to_native(channels, threshold=0.5)
    np.testing.assert_array_equal(reconstructed, seg)


def test_metrics_empty_region_handling():
    """Verify Dice and HD95 handle empty and non-empty ground truth correctly."""
    zeros = np.zeros((20, 20, 20), dtype=np.uint8)
    ones = np.ones((20, 20, 20), dtype=np.uint8)

    # Both empty -> Dice = 1.0, HD95 = 0.0
    assert compute_dice(zeros, zeros) == 1.0
    assert compute_hd95(zeros, zeros) == 0.0

    # One empty, one full -> Dice = 0.0, HD95 = penalty distance
    assert compute_dice(zeros, ones) == 0.0
    assert compute_hd95(zeros, ones) > 100.0

    # Sensitivity and Specificity
    assert compute_sensitivity(ones, ones) == 1.0
    assert compute_specificity(zeros, zeros) == 1.0


def test_anatomical_consistency_and_postprocessing():
    """Verify ET ⊆ TC ⊆ WT and ET < 200 voxel suppression rule."""
    prob_channels = np.zeros((3, 20, 20, 20), dtype=np.float32)
    # Put ET probability without WT probability (inconsistent)
    prob_channels[2, 5:8, 5:8, 5:8] = 0.9  # ET
    prob_channels[0, 5:8, 5:8, 5:8] = 0.1  # WT (low)

    wt, tc, et = enforce_anatomical_consistency(prob_channels, threshold=0.5)
    # Since WT is 0, ET must be forced to 0
    assert np.sum(et) == 0
    assert np.sum(tc) == 0

    # Test Winner ET < 200 rule within a realistic tumor mass
    seg = np.zeros((30, 30, 30), dtype=np.uint8)
    seg[8:14, 8:14, 8:14] = 2      # Outer edema mass (216 voxels > 10 min_component_size)
    seg[10:12, 10:12, 10:12] = 4  # 8 voxels of ET inside (< 200)
    cleaned = apply_winner_postprocessing(seg, et_min_voxels=200, min_component_size=10)
    assert np.sum(cleaned == 4) == 0
    assert np.sum(cleaned == 1) == 8  # Reassigned to necrotic core


def test_augmentation_pipeline():
    """Verify 3D spatial flip and gamma augmentation preserves shapes and types."""
    augmentor = BraTSAugmentor(flip_prob=1.0, rot90_prob=0.0, gamma_prob=1.0)
    img = np.random.uniform(0, 1, size=(4, 16, 16, 16)).astype(np.float32)
    seg = np.random.randint(0, 3, size=(16, 16, 16)).astype(np.uint8)

    aug_img, aug_seg = augmentor(img, seg)
    assert aug_img.shape == img.shape
    assert aug_seg.shape == seg.shape
    assert aug_img.dtype == np.float32
    assert aug_seg.dtype == np.uint8


def test_ensemble_averaging_and_weights():
    """Verify probability averaging and weighted fusion."""
    p1 = np.full((3, 10, 10, 10), 0.2, dtype=np.float32)
    p2 = np.full((3, 10, 10, 10), 0.8, dtype=np.float32)

    avg = simple_average_ensemble([p1, p2])
    np.testing.assert_allclose(avg, 0.5)

    weighted = weighted_average_ensemble([p1, p2], weights=[0.25, 0.75])
    expected = 0.25 * 0.2 + 0.75 * 0.8
    np.testing.assert_allclose(weighted, expected, atol=1e-5)


def test_model_forward_passes():
    """Verify forward passes of core architectures on small 3D batches."""
    x = torch.randn(1, 4, 32, 32, 32)

    # 1. 3D U-Net baseline
    unet = UNet3D(in_channels=4, num_classes=3, channels=[8, 16, 32])
    out_unet = unet(x)
    assert out_unet.shape == (1, 3, 32, 32, 32)

    # 2. Winner-Inspired U-Net (with deep supervision)
    winner = WinnerInspiredUNet(
        in_channels=4,
        num_classes=3,
        use_large_encoder=False,
        norm_type="group",
        num_groups=4,
        deep_supervision=True
    )
    winner.eval()
    out_winner = winner(x)
    assert out_winner.shape == (1, 3, 32, 32, 32)

    # 3. SegResNet
    segres = SegResNet(in_channels=4, out_channels=3, init_filters=8)
    out_segres = segres(x)
    assert out_segres.shape == (1, 3, 32, 32, 32)
