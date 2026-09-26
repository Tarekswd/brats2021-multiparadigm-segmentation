"""
Training curve visualizer for BraTS 2021 experiments (Section 40).
Generates publication-quality plots of Loss, WT/TC/ET Dice, and Learning Rate progression.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt


def plot_training_curves(
    log_csv_path: str,
    output_png_path: str,
    title_prefix: str = "BraTS 2021"
) -> None:
    """
    Reads training_log.csv and produces a multi-panel figure.
    """
    df = pd.read_csv(log_csv_path)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Loss curves
    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    if "val_loss" in df.columns:
        axes[0].plot(df["epoch"], df["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2)
    axes[0].set_title(f"{title_prefix}: Loss Curves", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Validation Dice Progression
    if "dice_wt" in df.columns:
        axes[1].plot(df["epoch"], df["dice_wt"], label="WT Dice", color="#2ca02c", linewidth=2)
        axes[1].plot(df["epoch"], df["dice_tc"], label="TC Dice", color="#d62728", linewidth=2)
        axes[1].plot(df["epoch"], df["dice_et"], label="ET Dice", color="#9467bd", linewidth=2)
        if "dice_avg" in df.columns:
            axes[1].plot(df["epoch"], df["dice_avg"], label="Avg Dice", color="#333333", linestyle="--", linewidth=2.5)
    axes[1].set_title("Validation Dice Overlap", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Score")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # Panel 3: Learning Rate
    if "learning_rate" in df.columns:
        axes[2].plot(df["epoch"], df["learning_rate"], label="Learning Rate", color="#8c564b", linewidth=2)
        axes[2].set_title("Learning Rate Decay", fontsize=13, fontweight="bold")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("LR")
        axes[2].grid(True, linestyle="--", alpha=0.6)
        axes[2].legend()

    plt.tight_layout()
    out = Path(output_png_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()
