"""
Validation CLI script for BraTS 2021 model checkpoints.
Usage:
    python scripts/validate.py --checkpoint experiments/winner_unet/checkpoints/best_model.pt --config configs/winner_unet.yaml
"""

import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from monai.inferers import sliding_window_inference

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.models import build_model
from src.data.dataset import BraTSDataset
from src.data.splitting import create_subject_splits
from src.training.checkpointing import CheckpointManager
from src.evaluation.evaluator import BraTSEvaluator
from src.evaluation.brats_regions import channels_to_native
from src.evaluation.postprocessing import apply_winner_postprocessing
from src.utils.logging import get_logger


def main():
    parser = argparse.ArgumentParser(description="BraTS 2021 Checkpoint Validation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, default="results/validation", help="Output directory for reports")
    args = parser.parse_args()

    logger = get_logger("ValidateCLI")
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    ckpt_mgr = CheckpointManager(str(Path(args.checkpoint).parent))
    ckpt_mgr.load(args.checkpoint, model=model, device=device)
    model.eval()

    data_root = Path(cfg["dataset"]["root"])
    subjects = sorted([p for p in data_root.iterdir() if p.is_dir() and "BraTS2021" in p.name])
    splits = create_subject_splits([s.name for s in subjects], val_fraction=0.25, seed=42)
    val_dirs = [data_root / sid for sid in splits["val"]]

    val_dataset = BraTSDataset(subject_dirs=val_dirs, patch_size=None, is_training=False, target_mode="multiclass")
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    evaluator = BraTSEvaluator()
    roi_size = tuple(cfg.get("inference", {}).get("roi_size", [128, 128, 128]))
    overlap = float(cfg.get("inference", {}).get("overlap", 0.5))

    logger.info(f"Validating {len(val_dirs)} subjects using sliding-window inference...")

    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].to(device)
            gt_native = batch["target"].squeeze(0).numpy()
            sub_id = batch["subject_id"][0]

            def model_forward(x):
                out = model(x)
                return out[0] if isinstance(out, list) else out

            actual_roi = tuple(min(r, s) for r, s in zip(roi_size, img.shape[2:]))
            logits = sliding_window_inference(
                inputs=img,
                roi_size=actual_roi,
                sw_batch_size=1,
                predictor=model_forward,
                overlap=overlap,
                mode="gaussian"
            )

            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # [3, D, H, W]
            pred_native = channels_to_native(probs, threshold=0.5, enforce_hierarchy=True)
            pred_post = apply_winner_postprocessing(pred_native, et_min_voxels=200)

            rec = evaluator.evaluate_subject(sub_id, pred_post, gt_native)
            logger.info(f"Subject {sub_id} | WT: {rec['dice_wt']:.3f} | TC: {rec['dice_tc']:.3f} | ET: {rec['dice_et']:.3f} | Avg: {rec['dice_avg']:.3f}")

    out_p = Path(args.output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    df = evaluator.get_summary_dataframe()
    df.to_csv(out_p / "validation_metrics.csv", index=False)
    stats = evaluator.get_aggregated_statistics()

    logger.info("Validation complete!")
    logger.info(f"Mean WT Dice: {stats['dice_wt']['mean']:.4f} | TC: {stats['dice_tc']['mean']:.4f} | ET: {stats['dice_et']['mean']:.4f} | Overall Avg: {stats['dice_avg']['mean']:.4f}")


if __name__ == "__main__":
    main()
