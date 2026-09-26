"""
Prediction CLI script for running inference on single patient cases.
Usage:
    python scripts/predict.py --input path/to/case --checkpoint path/to/model.pt --config configs/winner_unet.yaml --output results/predictions/
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from monai.inferers import sliding_window_inference

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import load_config
from src.models import build_model
from src.data.preprocessing import preprocess_case
from src.evaluation.brats_regions import channels_to_native
from src.evaluation.postprocessing import apply_winner_postprocessing, predict_with_tta
from src.training.checkpointing import CheckpointManager
from src.utils.logging import get_logger


def main():
    parser = argparse.ArgumentParser(description="BraTS 2021 Case Prediction")
    parser.add_argument("--input", type=str, required=True, help="Path to patient case directory")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output", type=str, default="results/predictions", help="Output directory")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation")
    args = parser.parse_args()

    logger = get_logger("PredictCLI")
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    ckpt_mgr = CheckpointManager(str(Path(args.checkpoint).parent))
    ckpt_mgr.load(args.checkpoint, model=model, device=device)
    model.eval()

    case_dir = Path(args.input)
    sub_id = case_dir.name
    logger.info(f"Predicting for patient case: {sub_id}")

    mod_paths = {}
    for m in ["flair", "t1", "t1ce", "t2"]:
        matches = list(case_dir.glob(f"*{m}.nii*"))
        if not matches:
            raise FileNotFoundError(f"Missing {m} in {case_dir}")
        mod_paths[m] = str(matches[0])

    preprocessed = preprocess_case(
        modality_paths=mod_paths,
        norm_method=cfg["dataset"].get("norm_method", "zscore"),
        tight_crop=True,
        add_foreground_channel=cfg["dataset"].get("add_foreground_channel", False)
    )

    image_cropped = preprocessed["image"]
    bbox = preprocessed["bbox"]
    orig_shape = preprocessed["original_shape"]
    affine = preprocessed["affine"]

    inp_tensor = torch.from_numpy(image_cropped).unsqueeze(0).float()
    roi_size = tuple(cfg.get("inference", {}).get("roi_size", [128, 128, 128]))
    overlap = float(cfg.get("inference", {}).get("overlap", 0.5))

    with torch.no_grad():
        if args.tta:
            logger.info("Applying 3D Test-Time Augmentation (TTA)...")
            prob_cropped = predict_with_tta(model, inp_tensor, device).squeeze(0).cpu().numpy()
        else:
            def model_forward(x):
                out = model(x.to(device))
                return out[0] if isinstance(out, list) else out

            actual_roi = tuple(min(r, s) for r, s in zip(roi_size, inp_tensor.shape[2:]))
            logits = sliding_window_inference(
                inputs=inp_tensor.to(device),
                roi_size=actual_roi,
                sw_batch_size=1,
                predictor=model_forward,
                overlap=overlap,
                mode="gaussian"
            )
            prob_cropped = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    # Decode channels to native labels {0, 1, 2, 4}
    pred_native_cropped = channels_to_native(prob_cropped, threshold=0.5, enforce_hierarchy=True)
    pred_post_cropped = apply_winner_postprocessing(pred_native_cropped, et_min_voxels=200)

    # Reconstruct back into original image space
    full_pred = np.zeros(orig_shape, dtype=np.uint8)
    full_pred[bbox[0], bbox[1], bbox[2]] = pred_post_cropped

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{sub_id}_pred.nii.gz"

    nii = nib.Nifti1Image(full_pred, affine)
    nib.save(nii, out_file)
    logger.info(f"Prediction successfully saved to: {out_file}")


if __name__ == "__main__":
    main()
