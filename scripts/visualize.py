"""
Pipeline diagram generator for BraTS 2021 (Figure 1).
Usage:
    python scripts/visualize.py [--output-dir results/figures]
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import get_logger


def generate_pipeline_figure(output_path: str):
    """Figure 1: Overall System Architecture and Pipeline Flowchart."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")

    # Draw boxes
    boxes = [
        {"name": "Multi-Modal MRI\n(T1, T1ce, T2, FLAIR)", "xy": (0.05, 0.65), "wh": (0.16, 0.22), "color": "#e0f2fe"},
        {"name": "Preprocessing\n(Tight Crop, Z-Score,\n5th FG Channel)", "xy": (0.26, 0.65), "wh": (0.16, 0.22), "color": "#fef3c7"},
        {"name": "Winner-Inspired U-Net\n(Asym Enc, GroupNorm,\nDeep Supervision)", "xy": (0.48, 0.73), "wh": (0.19, 0.18), "color": "#dcfce7"},
        {"name": "SegResNet Baseline\n(Residual Regularization)", "xy": (0.48, 0.48), "wh": (0.19, 0.18), "color": "#f3e8ff"},
        {"name": "Swin UNETR\n(Shifted-Window 3D ViT)", "xy": (0.48, 0.23), "wh": (0.19, 0.18), "color": "#fee2e2"},
        {"name": "Out-of-Fold Fusion\n(Learned / Weighted Avg)", "xy": (0.72, 0.48), "wh": (0.16, 0.18), "color": "#ffedd5"},
        {"name": "Post-Processing &\nAnatomical Constraint\n(ET<200, ET⊆TC⊆WT)", "xy": (0.72, 0.20), "wh": (0.16, 0.20), "color": "#ecfdf5"},
        {"name": "Final BraTS Output\n(WT, TC, ET Delineation)", "xy": (0.72, 0.73), "wh": (0.16, 0.18), "color": "#cffafe"},
    ]

    for b in boxes:
        rect = patches.FancyBboxPatch(
            b["xy"], b["wh"][0], b["wh"][1],
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=b["color"], edgecolor="#334155", linewidth=2
        )
        ax.add_patch(rect)
        cx = b["xy"][0] + b["wh"][0] / 2
        cy = b["xy"][1] + b["wh"][1] / 2
        ax.text(cx, cy, b["name"], ha="center", va="center", fontsize=9.5, fontweight="bold", color="#0f172a")

    # Arrows
    arrow_props = dict(facecolor="#475569", edgecolor="#334155", width=1.5, headwidth=7)
    ax.annotate("", xy=(0.26, 0.76), xytext=(0.21, 0.76), arrowprops=arrow_props)
    ax.annotate("", xy=(0.48, 0.82), xytext=(0.42, 0.76), arrowprops=arrow_props)
    ax.annotate("", xy=(0.48, 0.57), xytext=(0.42, 0.76), arrowprops=arrow_props)
    ax.annotate("", xy=(0.48, 0.32), xytext=(0.42, 0.76), arrowprops=arrow_props)

    ax.annotate("", xy=(0.72, 0.57), xytext=(0.67, 0.82), arrowprops=arrow_props)
    ax.annotate("", xy=(0.72, 0.57), xytext=(0.67, 0.57), arrowprops=arrow_props)
    ax.annotate("", xy=(0.72, 0.57), xytext=(0.67, 0.32), arrowprops=arrow_props)

    ax.annotate("", xy=(0.80, 0.40), xytext=(0.80, 0.48), arrowprops=arrow_props)
    ax.annotate("", xy=(0.80, 0.73), xytext=(0.80, 0.40), arrowprops=arrow_props)

    ax.set_title("Figure 1: BraTS 2021 Research-Grade Winning-Architecture & Ensemble Segmentation System", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    logger = get_logger("VisualizeCLI")
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Rendering pipeline diagram...")

    # Figure 1: Pipeline
    generate_pipeline_figure(str(out_dir / "figure1_pipeline.png"))

    logger.info("Figures generated in results/figures/")


if __name__ == "__main__":
    main()
