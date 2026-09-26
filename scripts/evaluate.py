"""
Evaluation CLI script for BraTS 2021 segmentation results.
Usage:
    python scripts/evaluate.py --predictions results/predictions --ground-truth data/BraTS2021 [--output-dir results/evaluation]
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np
import nibabel as nib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluator import BraTSEvaluator
from src.utils.logging import get_logger


def main():
    parser = argparse.ArgumentParser(description="BraTS 2021 Evaluation CLI")
    parser.add_argument("--predictions", type=str, required=True, help="Directory containing prediction NIfTIs")
    parser.add_argument("--ground-truth", type=str, required=True, help="Directory containing GT cases")
    parser.add_argument("--output-dir", type=str, default="results/evaluation", help="Output directory")
    args = parser.parse_args()

    logger = get_logger("EvaluateCLI")
    pred_dir = Path(args.predictions)
    gt_dir = Path(args.ground_truth)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_files = list(pred_dir.glob("*pred.nii*"))
    logger.info(f"Found {len(pred_files)} predictions to evaluate.")

    evaluator = BraTSEvaluator()

    for p_file in pred_files:
        # e.g. BraTS2021_00000_pred.nii.gz -> subject_id: BraTS2021_00000
        sub_id = p_file.name.replace("_pred.nii.gz", "").replace("_pred.nii", "")
        # Find corresponding GT segmentation
        sub_gt_dir = gt_dir / sub_id
        if not sub_gt_dir.is_dir():
            candidates = list(gt_dir.glob(f"*{sub_id}*seg.nii*"))
            if candidates:
                gt_file = candidates[0]
            else:
                logger.warning(f"Could not find ground truth for {sub_id}")
                continue
        else:
            candidates = list(sub_gt_dir.glob("*seg.nii*"))
            if not candidates:
                logger.warning(f"Could not find seg file inside {sub_gt_dir}")
                continue
            gt_file = candidates[0]

        pred_data = np.asarray(nib.load(p_file).dataobj, dtype=np.uint8)
        gt_data = np.asarray(nib.load(gt_file).dataobj, dtype=np.uint8)

        rec = evaluator.evaluate_subject(sub_id, pred_data, gt_data)
        logger.info(f"{sub_id} | WT Dice: {rec['dice_wt']:.3f} | TC: {rec['dice_tc']:.3f} | ET: {rec['dice_et']:.3f} | Avg: {rec['dice_avg']:.3f}")

    df = evaluator.get_summary_dataframe()
    df.to_csv(out_dir / "patient_metrics.csv", index=False)

    stats = evaluator.get_aggregated_statistics()
    with open(out_dir / "summary_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("Evaluation complete! Metrics written to results.")


if __name__ == "__main__":
    main()
