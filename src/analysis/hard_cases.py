"""
Automatic Hard-Case and Failure-Mode Mining for BraTS 2021 (Sections 25 & 38).
Identifies Top 10 Best and Worst cases for WT, TC, ET, and HD95.
"""

from typing import Dict, List, Any
import pandas as pd


def mine_best_and_worst_cases(
    eval_df: pd.DataFrame,
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Extract top best and worst performing subject IDs across all BraTS subregions.
    """
    results: Dict[str, Any] = {}

    metrics = [
        ("dice_wt", True),   # True: higher is better
        ("dice_tc", True),
        ("dice_et", True),
        ("dice_avg", True),
        ("hd95_wt", False),  # False: lower is better
        ("hd95_tc", False),
        ("hd95_et", False),
    ]

    for metric, higher_is_better in metrics:
        if metric not in eval_df.columns:
            continue

        sorted_df = eval_df.sort_values(by=metric, ascending=not higher_is_better)
        best_cases = sorted_df.head(top_k)[["subject_id", metric]].to_dict(orient="records")
        worst_cases = sorted_df.tail(top_k).iloc[::-1][["subject_id", metric]].to_dict(orient="records")

        results[metric] = {
            f"top_{top_k}_best": best_cases,
            f"top_{top_k}_worst": worst_cases
        }

    return results
