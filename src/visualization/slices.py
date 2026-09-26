"""
Multi-modal slice visualizer with color segmentation overlays for BraTS 2021 (Figures 2, 3, 4, 7, 8).
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def create_brats_cmap() -> mcolors.ListedColormap:
    """
    BraTS official color mapping:
      0 = Transparent (Background)
      1 = Red (Necrotic / Non-enhancing Core, NCR)
      2 = Green (Peritumoral Edema, ED)
      4 = Yellow (Enhancing Tumor, ET)
    """
    colors = [
        (0.0, 0.0, 0.0, 0.0),  # 0: Background
        (0.9, 0.1, 0.1, 0.7),  # 1: Red (NCR)
        (0.1, 0.8, 0.1, 0.7),  # 2: Green (ED)
        (0.0, 0.0, 0.0, 0.0),  # 3: Unused in BraTS 2021
        (1.0, 0.9, 0.1, 0.8),  # 4: Yellow (ET)
    ]
    return mcolors.ListedColormap(colors)


def plot_multimodal_slices(
    modalities: np.ndarray,
    seg_native: Optional[np.ndarray] = None,
    pred_native: Optional[np.ndarray] = None,
    slice_idx: Optional[int] = None,
    output_png_path: str = "results/figures/slice_view.png"
) -> None:
    """
    Plots the 4 MRI contrasts (T1, T1ce, T2, FLAIR) and optional GT / Prediction overlays.
    modalities: [4, D, H, W]
    """
    if slice_idx is None:
        # Select axial slice with maximum tumor content, or central slice
        if seg_native is not None and np.sum(seg_native > 0) > 0:
            slice_counts = np.sum(seg_native > 0, axis=(1, 2))
            slice_idx = int(np.argmax(slice_counts))
        else:
            slice_idx = modalities.shape[1] // 2

    fig, axes = plt.subplots(1, 6 if pred_native is not None else 5, figsize=(22, 4.5))
    seq_names = ["T1", "T1ce", "T2", "FLAIR"]

    for i in range(4):
        axes[i].imshow(modalities[i, slice_idx], cmap="gray", origin="lower")
        axes[i].set_title(f"Modality: {seq_names[i]} (Slice {slice_idx})", fontsize=11, fontweight="bold")
        axes[i].axis("off")

    cmap = create_brats_cmap()

    # Ground Truth Overlay on FLAIR
    axes[4].imshow(modalities[3, slice_idx], cmap="gray", origin="lower")
    if seg_native is not None:
        axes[4].imshow(seg_native[slice_idx], cmap=cmap, origin="lower", vmin=0, vmax=4)
    axes[4].set_title("Ground Truth (NCR/ED/ET)", fontsize=11, fontweight="bold")
    axes[4].axis("off")

    # Prediction Overlay on FLAIR
    if pred_native is not None:
        axes[5].imshow(modalities[3, slice_idx], cmap="gray", origin="lower")
        axes[5].imshow(pred_native[slice_idx], cmap=cmap, origin="lower", vmin=0, vmax=4)
        axes[5].set_title("Model Prediction", fontsize=11, fontweight="bold")
        axes[5].axis("off")

    plt.tight_layout()
    out = Path(output_png_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()
