"""
Confusion Matrix computation and visualization for BraTS 2021.
Computes:
  - 4x4 Multi-class voxel confusion matrix (Background, NCR, ED, ET)
  - 2x2 Binary region confusion matrices (WT, TC, ET)
  - Normalized confusion heatmaps with annotations
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from src.evaluation.brats_regions import native_to_regions


def compute_multiclass_confusion_matrix(
    pred_native: np.ndarray,
    gt_native: np.ndarray
) -> np.ndarray:
    """
    Computes 4x4 confusion matrix for native BraTS labels:
    Index 0: Background (0)
    Index 1: Necrotic / Non-enhancing Core (1)
    Index 2: Peritumoral Edema (2)
    Index 3: Enhancing Tumor (4)
    """
    label_map = {0: 0, 1: 1, 2: 2, 4: 3}
    y_true = np.vectorize(lambda x: label_map.get(x, 0))(gt_native.flatten())
    y_pred = np.vectorize(lambda x: label_map.get(x, 0))(pred_native.flatten())

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    return cm


def compute_region_confusion_matrix(
    pred_binary: np.ndarray,
    gt_binary: np.ndarray
) -> Dict[str, Any]:
    """
    Computes 2x2 confusion matrix (TP, FP, FN, TN) for a binary region mask (WT, TC, or ET).
    """
    p = pred_binary.astype(bool)
    g = gt_binary.astype(bool)

    tp = int(np.sum(p & g))
    fp = int(np.sum(p & (~g)))
    fn = int(np.sum((~p) & g))
    tn = int(np.sum((~p) & (~g)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    dice = 2.0 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "dice": float(dice),
        "matrix": np.array([[tn, fp], [fn, tp]])
    }


def plot_confusion_matrices(
    cm_4x4: np.ndarray,
    region_cms: Dict[str, Dict[str, Any]],
    output_png_path: str = "results/figures/figure_confusion_matrix.png"
) -> None:
    """
    Generates a publication-grade 4-panel confusion matrix figure:
    Panel 1: 4x4 Multi-class Normalized Heatmap
    Panels 2-4: 2x2 Binary Confusion Matrices for WT, TC, and ET
    """
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    labels = ["BG (0)", "NCR (1)", "ED (2)", "ET (4)"]

    # 1. Multi-class 4x4 Matrix (Row-normalized recall percentages)
    cm_norm = cm_4x4.astype("float") / (cm_4x4.sum(axis=1)[:, np.newaxis] + 1e-9) * 100.0

    im = axes[0].imshow(cm_norm, cmap="Blues", vmin=0, vmax=100)
    axes[0].set_title("Multi-Class Voxel Recall (%)", fontsize=12, fontweight="bold")
    axes[0].set_xticks(range(4))
    axes[0].set_yticks(range(4))
    axes[0].set_xticklabels(labels, rotation=30)
    axes[0].set_yticklabels(labels)
    axes[0].set_xlabel("Predicted Label", fontweight="bold")
    axes[0].set_ylabel("True Ground Truth", fontweight="bold")

    for i in range(4):
        for j in range(4):
            val = cm_norm[i, j]
            color = "white" if val > 50 else "black"
            axes[0].text(j, i, f"{val:.1f}%\n({cm_4x4[i,j]:,})", ha="center", va="center", color=color, fontsize=8)

    # 2-4. Region Matrices (WT, TC, ET)
    region_names = [("wt", "Whole Tumor (WT)"), ("tc", "Tumor Core (TC)"), ("et", "Enhancing Tumor (ET)")]
    for idx, (key, title) in enumerate(region_names, start=1):
        reg_info = region_cms[key]
        tp, fp, fn, tn = reg_info["tp"], reg_info["fp"], reg_info["fn"], reg_info["tn"]
        sens, prec, dice = reg_info["sensitivity"], reg_info["precision"], reg_info["dice"]

        mat = np.array([[tn, fp], [fn, tp]])
        mat_norm = mat.astype("float") / (mat.sum(axis=1)[:, np.newaxis] + 1e-9) * 100.0

        axes[idx].imshow(mat_norm, cmap="Greens", vmin=0, vmax=100)
        axes[idx].set_title(f"{title}\nDice: {dice*100:.1f}% | Sens: {sens*100:.1f}%", fontsize=11, fontweight="bold")
        axes[idx].set_xticks([0, 1])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_xticklabels(["Neg", "Pos"])
        axes[idx].set_yticklabels(["Neg", "Pos"])
        axes[idx].set_xlabel("Predicted", fontweight="bold")
        axes[idx].set_ylabel("True GT", fontweight="bold")

        axes[idx].text(0, 0, f"TN\n{tn:,}", ha="center", va="center", fontsize=9, fontweight="bold")
        axes[idx].text(1, 0, f"FP\n{fp:,}", ha="center", va="center", fontsize=9, fontweight="bold", color="red")
        axes[idx].text(0, 1, f"FN\n{fn:,}", ha="center", va="center", fontsize=9, fontweight="bold", color="red")
        axes[idx].text(1, 1, f"TP\n{tp:,}", ha="center", va="center", fontsize=9, fontweight="bold", color="white")

    plt.tight_layout()
    out = Path(output_png_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()
