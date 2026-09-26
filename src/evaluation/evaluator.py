"""
Comprehensive Patient-Level Evaluator for BraTS 2021.
Aggregates Dice, HD95, Sensitivity, and Specificity across subjects.
Computes Mean, Median, and Standard Deviation statistics.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

from src.evaluation.brats_regions import native_to_regions
from src.evaluation.dice import compute_brats_dices
from src.evaluation.hd95 import compute_brats_hd95s
from src.evaluation.sensitivity import compute_brats_sensitivity_specificity


class BraTSEvaluator:
    """Computes and tracks patient-level evaluation metrics."""

    def __init__(self, voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.voxel_spacing = voxel_spacing
        self.patient_records: List[Dict[str, Any]] = []

    def evaluate_subject(
        self,
        subject_id: str,
        pred_native: np.ndarray,
        gt_native: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate single patient prediction against ground truth.
        Both inputs are 3D integer arrays with labels in {0, 1, 2, 4}.
        """
        pred_regions = native_to_regions(pred_native)
        gt_regions = native_to_regions(gt_native)

        dice_metrics = compute_brats_dices(pred_regions, gt_regions)
        hd95_metrics = compute_brats_hd95s(pred_regions, gt_regions, self.voxel_spacing)
        sens_spec_metrics = compute_brats_sensitivity_specificity(pred_regions, gt_regions)

        record = {
            "subject_id": subject_id,
            **dice_metrics,
            **hd95_metrics,
            **sens_spec_metrics
        }
        self.patient_records.append(record)
        return record

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Return raw per-patient dataframe."""
        return pd.DataFrame(self.patient_records)

    def get_aggregated_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate Mean, Median, and Std across all evaluated subjects.
        """
        if not self.patient_records:
            return {}

        df = pd.DataFrame(self.patient_records)
        metric_cols = [
            "dice_wt", "dice_tc", "dice_et", "dice_avg",
            "hd95_wt", "hd95_tc", "hd95_et", "hd95_avg",
            "sens_wt", "sens_tc", "sens_et",
            "spec_wt", "spec_tc", "spec_et"
        ]

        summary = {}
        for col in metric_cols:
            if col in df:
                summary[col] = {
                    "mean": float(df[col].mean()),
                    "median": float(df[col].median()),
                    "std": float(df[col].std()) if len(df) > 1 else 0.0
                }

        return summary
