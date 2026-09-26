"""
Metrics and comparative visualizer for BraTS 2021 benchmarking (Figures 10, 11, 13).
"""

from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_model_comparison_bar(
    comparison_df: pd.DataFrame,
    output_png_path: str,
    metric_type: str = "dice"  # 'dice' or 'hd95'
) -> None:
    """
    Generates comparative grouped bar plot across models for WT, TC, ET, and Avg.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    models = comparison_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.2

    if metric_type == "dice":
        cols = ["WT Dice", "TC Dice", "ET Dice"]
        colors = ["#2ca02c", "#d62728", "#9467bd"]
        y_label = "Dice Similarity Coefficient (%)"
        title = "Model Comparison: Dice Overlap Across Brain Tumor Subregions"
    else:
        cols = ["WT HD95", "TC HD95", "ET HD95"]
        colors = ["#17becf", "#e377c2", "#bcbd22"]
        y_label = "95th Percentile Hausdorff Distance (mm) [Lower is Better]"
        title = "Model Comparison: 95% Hausdorff Distance Boundary Precision"

    for i, col in enumerate(cols):
        values = comparison_df[col].astype(float).tolist()
        ax.bar(x + (i - 1) * width, values, width, label=col.split()[0], color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=11, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    out = Path(output_png_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()


def plot_accuracy_vs_cost(
    benchmark_df: pd.DataFrame,
    output_png_path: str
) -> None:
    """
    Generates Accuracy (Dice) vs Computational Cost (Params / Inference Time) scatter plot.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for _, row in benchmark_df.iterrows():
        model = row["Model"]
        params = float(row["Params_M"])
        dice = float(row["Avg_Dice"])
        ax.scatter(params, dice, s=160, label=model)
        ax.annotate(
            model,
            (params, dice),
            textcoords="offset points",
            xytext=(7, 7),
            fontweight="bold"
        )

    ax.set_xlabel("Parameters (Millions)", fontsize=12)
    ax.set_ylabel("Average Dice Score (%)", fontsize=12)
    ax.set_title("BraTS 2021 Pareto Frontier: Accuracy vs. Model Capacity", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out = Path(output_png_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()
