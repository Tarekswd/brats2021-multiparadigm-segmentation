"""
Error and failure mode visualizer for BraTS 2021 (Figure 12).
Generates multi-panel figures containing FLAIR, T1, T1ce, T2, Ground Truth, Prediction,
and True Positive / False Positive / False Negative spatial error maps.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.visualization.slices import create_brats_cmap


def plot_error_analysis_case(
    modalities: np.ndarray,
    gt_native: np.ndarray,
    pred_native: np.ndarray,
    subject_id: str,
    output_png_path: str,
    slice_idx: Optional[int] = None
) -> None:
    """
    modalities: [4, D, H, W]
    gt_native: [D, H, W]
    pred_native: [D, H, W]
    """
    if slice_idx is None:
        disagreement = (gt_native > 0) ^ (pred_native > 0)
        if np.sum(disagreement) > 0:
            slice_idx = int(np.argmax(np.sum(disagreement, axis=(1, 2))))
        else:
            slice_idx = gt_native.shape[0] // 2

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    seq_names = ["T1", "T1ce", "T2", "FLAIR"]

    # Top Row: 4 Modalities
    for i in range(4):
        axes[0, i].imshow(modalities[i, slice_idx], cmap="gray", origin="lower")
        axes[0, i].set_title(f"{seq_names[i]} (Slice {slice_idx})", fontsize=11, fontweight="bold")
        axes[0, i].axis("off")

    cmap = create_brats_cmap()

    # Bottom Row 1: Ground Truth
    axes[1, 0].imshow(modalities[3, slice_idx], cmap="gray", origin="lower")
    axes[1, 0].imshow(gt_native[slice_idx], cmap=cmap, origin="lower", vmin=0, vmax=4)
    axes[1, 0].set_title("Ground Truth", fontsize=11, fontweight="bold")
    axes[1, 0].axis("off")

    # Bottom Row 2: Prediction
    axes[1, 1].imshow(modalities[3, slice_idx], cmap="gray", origin="lower")
    axes[1, 1].imshow(pred_native[slice_idx], cmap=cmap, origin="lower", vmin=0, vmax=4)
    axes[1, 1].set_title("Prediction", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")

    # Bottom Row 3: Whole Tumor Error Map (TP, FP, FN)
    gt_wt = gt_native[slice_idx] > 0
    pred_wt = pred_native[slice_idx] > 0

    error_map = np.zeros_like(gt_wt, dtype=np.uint8)
    error_map[gt_wt & pred_wt] = 1   # TP: Cyan/Green
    error_map[(~gt_wt) & pred_wt] = 2  # FP: Magenta/Red
    error_map[gt_wt & (~pred_wt)] = 3  # FN: Blue

    err_colors = [
        (0.0, 0.0, 0.0, 0.0),  # 0: BG
        (0.1, 0.8, 0.1, 0.7),  # 1: TP (Green)
        (0.9, 0.1, 0.1, 0.7),  # 2: FP (Red)
        (0.1, 0.3, 0.9, 0.7),  # 3: FN (Blue)
    ]
    err_cmap = mcolors.ListedColormap(err_colors)

    axes[1, 2].imshow(modalities[3, slice_idx], cmap="gray", origin="lower")
    axes[1, 2].imshow(error_map, cmap=err_cmap, origin="lower", vmin=0, vmax=3)
    axes[1, 2].set_title("WT Error Map (Green=TP, Red=FP, Blue=FN)", fontsize=10, fontweight="bold")
    axes[1, 2].axis("off")

    # Bottom Row 4: Enhancing Tumor Error Map
    gt_et = gt_native[slice_idx] == 4
    pred_et = pred_native[slice_idx] == 4
    et_err_map = np.zeros_like(gt_et, dtype=np.uint8)
    et_err_map[gt_et & pred_et] = 1
    et_err_map[(~gt_et) & pred_et] = 2
    et_err_map[gt_et & (~pred_et)] = 3

    axes[1, 3].imshow(modalities[1, slice_idx], cmap="gray", origin="lower")
    axes[1, 3].imshow(et_err_map, cmap=err_cmap, origin="lower", vmin=0, vmax=3)
    axes[1, 3].set_title("ET Error Map (T1ce Background)", fontsize=10, fontweight="bold")
    axes[1, 3].axis("off")

    plt.suptitle(f"Subject Error Analysis: {subject_id}", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    out = Path(output_png_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()
