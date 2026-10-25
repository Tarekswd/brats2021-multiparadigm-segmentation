"""
Preprocessing pipeline for BraTS 2021 multi-modal MRI scans.
Supports:
  - Multi-sequence NIfTI reading (T1, T1ce, T2, FLAIR)
  - Non-zero foreground mask extraction and tight cropping
  - Foreground-based Z-score normalization
  - Robust percentile normalization (1st - 99th percentiles)
  - Optional 5th foreground mask channel
  - Coordinate tracking for exact restoration to original volume space
"""

import numpy as np
import nibabel as nib
from typing import Dict, Tuple, Optional, List


def load_nifti(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load NIfTI volume and return data array (float32) and affine matrix."""
    img = nib.load(path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    affine = img.affine
    return data, affine


def get_foreground_mask(modalities: np.ndarray) -> np.ndarray:
    """
    Extract non-zero foreground mask across modalities.
    modalities shape: [C, D, H, W]
    Returns binary mask: [D, H, W]
    """
    return (np.sum(np.abs(modalities), axis=0) > 1e-4).astype(np.uint8)


def get_bounding_box(mask: np.ndarray) -> Tuple[slice, slice, slice]:
    """Calculate tight 3D bounding box for non-zero voxels in mask."""
    indices = np.where(mask > 0)
    if len(indices[0]) == 0:
        return slice(0, mask.shape[0]), slice(0, mask.shape[1]), slice(0, mask.shape[2])
    bbox = tuple(slice(int(np.min(idx)), int(np.max(idx)) + 1) for idx in indices)
    return bbox


def normalize_intensity(
    volume: np.ndarray,
    mask: Optional[np.ndarray] = None,
    method: str = "zscore"
) -> np.ndarray:
    """
    Normalize 3D modality volume using foreground statistics.
    method: 'zscore' or 'robust_percentile'
    """
    if mask is None:
        mask = volume > 0

    fg = volume[mask > 0]
    if len(fg) == 0:
        return volume

    out = np.zeros_like(volume)
    if method == "robust_percentile":
        p_low, p_high = np.percentile(fg, (1.0, 99.0))
        fg_clipped = np.clip(fg, p_low, p_high)
        mean = np.mean(fg_clipped)
        std = np.std(fg_clipped)
        if std > 1e-6:
            out[mask > 0] = (np.clip(volume[mask > 0], p_low, p_high) - mean) / std
        else:
            out[mask > 0] = volume[mask > 0] - mean
    else:  # 'zscore'
        mean = np.mean(fg)
        std = np.std(fg)
        if std > 1e-6:
            out[mask > 0] = (volume[mask > 0] - mean) / std
        else:
            out[mask > 0] = volume[mask > 0] - mean

    return out


def preprocess_case(
    modality_paths: Dict[str, str],
    seg_path: Optional[str] = None,
    norm_method: str = "zscore",
    tight_crop: bool = True,
    add_foreground_channel: bool = False
) -> Dict[str, any]:
    """
    Preprocess all modalities for a BraTS patient case.
    modality_paths keys: 't1', 't1ce', 't2', 'flair'
    """
    sequences = ["t1", "t1ce", "t2", "flair"]
    vols = []
    affine = None

    for seq in sequences:
        if seq not in modality_paths:
            raise KeyError(f"Missing required modality: {seq}")
        data, aff = load_nifti(modality_paths[seq])
        vols.append(data)
        if affine is None:
            affine = aff

    stacked = np.stack(vols, axis=0)  # [4, D, H, W]
    original_shape = stacked.shape[1:]

    # Foreground brain mask
    fg_mask = get_foreground_mask(stacked)

    # Tight bounding box cropping
    bbox = get_bounding_box(fg_mask) if tight_crop else (
        slice(0, original_shape[0]),
        slice(0, original_shape[1]),
        slice(0, original_shape[2])
    )

    cropped = stacked[:, bbox[0], bbox[1], bbox[2]]
    cropped_fg = fg_mask[bbox[0], bbox[1], bbox[2]]

    # Modality-wise normalization
    normalized_channels = []
    for c in range(cropped.shape[0]):
        norm_c = normalize_intensity(cropped[c], cropped_fg, method=norm_method)
        normalized_channels.append(norm_c)

    if add_foreground_channel:
        # Append 5th channel (foreground mask)
        normalized_channels.append(cropped_fg.astype(np.float32))

    image = np.stack(normalized_channels, axis=0)

    seg_cropped = None
    if seg_path is not None:
        seg_data, _ = load_nifti(seg_path)
        seg_cropped = seg_data[bbox[0], bbox[1], bbox[2]].astype(np.uint8)

    return {
        "image": image,
        "seg": seg_cropped,
        "bbox": bbox,
        "original_shape": original_shape,
        "affine": affine
    }
