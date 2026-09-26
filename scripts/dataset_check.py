"""
Dataset Validation CLI Script for BraTS 2021.
Usage:
    python scripts/dataset_check.py --data-root /path/to/BraTS2021 --output-dir results/
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.validation import DatasetValidator
from src.utils.logging import get_logger


def main():
    parser = argparse.ArgumentParser(description="BraTS 2021 Dataset Integrity Validation")
    parser.add_argument("--data-root", type=str, required=True, help="Path to BraTS 2021 dataset root directory")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save validation reports")
    args = parser.parse_args()

    logger = get_logger("DatasetCheck")
    logger.info(f"Initiating BraTS 2021 dataset inspection on: {args.data_root}")

    validator = DatasetValidator(args.data_root)
    summary = validator.run_full_validation(args.output_dir)

    logger.info(f"Inspection complete. Total: {summary['total_subjects']}, Valid: {summary['valid_subjects']}")
    logger.info(f"Reports saved to: {args.output_dir}/dataset_report.[json|csv|md]")

    if summary["corrupted_or_missing_subjects"] > 0:
        logger.warning(f"Found {summary['corrupted_or_missing_subjects']} corrupted/incomplete subject(s)!")
        sys.exit(1)
    else:
        logger.info("Dataset verification PASSED with zero integrity errors.")


if __name__ == "__main__":
    main()
