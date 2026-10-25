"""
Training CLI script for BraTS 2021 Brain Tumor Segmentation.
Usage:
    python scripts/train.py --config configs/winner_unet.yaml [--output-dir experiments/winner_unet]
"""

import argparse
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.hardware import recommend_training_parameters
from src.models import build_model
from src.losses.combined import build_loss
from src.data.dataset import BraTSDataset
from src.data.augmentation import BraTSAugmentor
from src.data.splitting import create_subject_splits
from src.training.trainer import BraTSTrainer
from src.utils.logging import get_logger


def main():
    parser = argparse.ArgumentParser(description="BraTS 2021 Model Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for logs and weights")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--data-root", type=str, default=None, help="Override dataset root (folder of BraTS2021_XXXXX cases)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.data_root is not None:
        cfg["dataset"]["root"] = args.data_root

    model_name = cfg.get("model", {}).get("name", "model")
    out_dir = args.output_dir or f"experiments/{model_name}"

    logger = get_logger("TrainCLI")
    logger.info(f"Loaded config: {args.config}")
    logger.info(f"Target experiment directory: {out_dir}")

    # Inspect device and memory recommendations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0.0
    rec = recommend_training_parameters(vram)
    logger.info(f"Compute device: {device} | Hardware tier: {rec['tier']}")

    # Build model and loss
    model = build_model(cfg)
    loss_fn = build_loss(cfg)

    # Locate dataset subjects
    data_root = Path(cfg["dataset"]["root"])
    if not data_root.exists() and args.data_root is not None:
        # never silently fall back to mock data when real data was requested
        raise FileNotFoundError(f"--data-root {data_root} does not exist")
    if not data_root.exists():
        logger.warning(f"Data root {data_root} does not exist. Creating synthetic mock data for validation pipeline.")
        data_root.mkdir(parents=True, exist_ok=True)
        # Create minimal mock subjects for dry run verification
        from tests.create_mock_data import generate_mock_subject
        for i in range(4):
            generate_mock_subject(data_root / f"BraTS2021_{i:05d}")

    subjects = sorted([p for p in data_root.iterdir() if p.is_dir() and "BraTS2021" in p.name])
    splits = create_subject_splits([s.name for s in subjects], val_fraction=0.25, seed=args.seed)

    train_dirs = [data_root / sid for sid in splits["train"]]
    val_dirs = [data_root / sid for sid in splits["val"]]

    if rec["tier"] == "cpu_or_low_memory":
        patch_size = tuple(rec["patch_size"])
        batch_size = rec["batch_size"]
        cfg.setdefault("inference", {})
        cfg["inference"]["roi_size"] = list(patch_size)
        cfg["inference"]["overlap"] = 0.0
    else:
        patch_size = tuple(cfg["training"].get("patch_size", rec["patch_size"]))
        batch_size = cfg["training"].get("batch_size", rec["batch_size"])

    augmentor = BraTSAugmentor() if cfg.get("augmentation", {}).get("enabled", True) else None

    train_dataset = BraTSDataset(
        subject_dirs=train_dirs,
        patch_size=patch_size,
        is_training=True,
        augmentor=augmentor,
        norm_method=cfg["dataset"].get("norm_method", "zscore"),
        add_foreground_channel=cfg["dataset"].get("add_foreground_channel", False)
    )
    val_dataset = BraTSDataset(
        subject_dirs=val_dirs,
        patch_size=None,  # Full volume for sliding window validation
        is_training=False,
        norm_method=cfg["dataset"].get("norm_method", "zscore"),
        add_foreground_channel=cfg["dataset"].get("add_foreground_channel", False)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    trainer = BraTSTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        cfg=cfg,
        output_dir=out_dir,
        device=device
    )

    results = trainer.fit()
    logger.info(f"Training successfully finished. Result summary: {results}")


if __name__ == "__main__":
    main()
