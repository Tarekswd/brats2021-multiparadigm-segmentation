"""
Model Complementarity Analysis for BraTS 2021 Ensembling (Section 28).
Calculates per-subject correlations, error overlap, and failure case divergence
between CNN (Winner U-Net, SegResNet) and Transformer (Swin UNETR).
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def analyze_model_complementarity(
    model_a_results: pd.DataFrame,
    model_b_results: pd.DataFrame,
    model_a_name: str = "WinnerUNet",
    model_b_name: str = "SwinUNETR"
) -> Dict[str, Any]:
    """
    Compare per-patient results of two models to measure diversity and complementarity.
    Expects DataFrames with 'subject_id' and 'dice_wt', 'dice_tc', 'dice_et', 'dice_avg'.
    """
    merged = pd.merge(
        model_a_results, model_b_results,
        on="subject_id",
        suffixes=(f"_{model_a_name}", f"_{model_b_name}")
    )

    col_a = f"dice_avg_{model_a_name}"
    col_b = f"dice_avg_{model_b_name}"

    if len(merged) < 2:
        return {"error": "Insufficient subjects for complementarity analysis"}

    p_corr, _ = pearsonr(merged[col_a], merged[col_b])
    s_corr, _ = spearmanr(merged[col_a], merged[col_b])

    # Cases where Model A is substantially better than Model B (> 5% Dice difference)
    diff = merged[col_a] - merged[col_b]
    a_wins = merged[diff > 0.05]["subject_id"].tolist()
    b_wins = merged[diff < -0.05]["subject_id"].tolist()

    return {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "total_compared_subjects": len(merged),
        "pearson_correlation": float(p_corr),
        "spearman_correlation": float(s_corr),
        "mean_diff": float(diff.mean()),
        "model_a_superior_count": len(a_wins),
        "model_b_superior_count": len(b_wins),
        "complementary_subjects": {
            f"{model_a_name}_superior_cases": a_wins,
            f"{model_b_name}_superior_cases": b_wins
        }
    }
