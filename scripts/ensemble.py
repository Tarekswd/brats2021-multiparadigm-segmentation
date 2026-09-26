"""
Ensemble CLI script for fusing multi-model predictions in BraTS 2021.
Usage:
    python scripts/ensemble.py --config configs/ensemble.yaml [--output-dir results/ensemble]
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np
import nibabel as nib
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.ensemble.averaging import simple_average_ensemble
from src.ensemble.weighted_fusion import weighted_average_ensemble
from src.ensemble.voting import hard_voting_ensemble
from src.evaluation.brats_regions import channels_to_native
from src.evaluation.postprocessing import apply_winner_postprocessing
from src.utils.logging import get_logger


def main():
    parser = argparse.ArgumentParser(description="BraTS 2021 Multi-Model Ensemble CLI")
    parser.add_argument("--config", type=str, default="configs/ensemble.yaml", help="Path to ensemble config")
    parser.add_argument("--output-dir", type=str, default="results/ensemble", help="Output directory")
    args = parser.parse_args()

    logger = get_logger("EnsembleCLI")
    cfg = load_config(args.config)
    ens_cfg = cfg.get("ensemble", {})
    strategy = ens_cfg.get("strategy", "weighted_average")
    models_info = ens_cfg.get("models", [])

    logger.info(f"Initiating ensemble fusion using strategy: {strategy}")
    logger.info(f"Number of component architectures: {len(models_info)}")

    weights = [m.get("initial_weight", 1.0 / len(models_info)) for m in models_info]
    logger.info(f"Ensemble weighting distribution: {weights}")

    out_p = Path(args.output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    summary = {
        "strategy": strategy,
        "models": [m["name"] for m in models_info],
        "weights": weights,
        "postprocessing": cfg.get("postprocessing", {})
    }
    with open(out_p / "ensemble_manifest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Ensemble configuration manifest saved to: {out_p}/ensemble_manifest.json")


if __name__ == "__main__":
    main()
