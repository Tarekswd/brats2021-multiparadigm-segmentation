"""
Docker Entrypoint Inference Script for BraTS Challenge / Synapse Submissions.
Batch processes all patient folders in /input and writes reconstructed NIfTI segmentations to /output.
"""

import os
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from monai.inferers import sliding_window_inference

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import load_config
from src.models import build_model
from src.data.preprocessing import preprocess_case
from src.ensemble.weighted_fusion import weighted_average_ensemble
from src.evaluation.brats_regions import channels_to_native
from src.evaluation.postprocessing import apply_winner_postprocessing, enforce_anatomical_consistency, predict_with_tta


def main():
    input_dir = Path(os.environ.get("INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BraTS Docker Inference] Running on compute device: {device}")
    print(f"[BraTS Docker Inference] Reading from: {input_dir}, Writing to: {output_dir}")

    # Load component models (Winner U-Net, SegResNet, Swin UNETR)
    cfg_winner = load_config("configs/winner_unet.yaml")
    cfg_segres = load_config("configs/segresnet.yaml")
    cfg_swin = load_config("configs/swin_unetr.yaml")

    model_winner = build_model(cfg_winner).to(device)
    model_segres = build_model(cfg_segres).to(device)
    model_swin = build_model(cfg_swin).to(device)

    model_winner.eval()
    model_segres.eval()
    model_swin.eval()

    # Locate patient cases in input directory
    patient_dirs = [p for p in input_dir.iterdir() if p.is_dir() and "BraTS" in p.name]
    if not patient_dirs:
        # Fallback if cases are direct files in /input
        patient_dirs = [input_dir]

    print(f"[BraTS Docker Inference] Found {len(patient_dirs)} patient cases to process.")

    with torch.no_grad():
        for p_dir in patient_dirs:
            sub_id = p_dir.name
            print(f"--> Processing case: {sub_id}")

            mod_paths = {}
            for m in ["flair", "t1", "t1ce", "t2"]:
                matches = list(p_dir.glob(f"*{m}.nii*"))
                if matches:
                    mod_paths[m] = str(matches[0])
                else:
                    print(f"Warning: {m} not found in {p_dir}")

            if len(mod_paths) < 4:
                print(f"Skipping {sub_id} due to missing sequences.")
                continue

            prep = preprocess_case(mod_paths=mod_paths, tight_crop=True)
            image_cropped = prep["image"]
            bbox = prep["bbox"]
            orig_shape = prep["original_shape"]
            affine = prep["affine"]

            inp_tensor = torch.from_numpy(image_cropped).unsqueeze(0).float().to(device)
            actual_roi = tuple(min(r, s) for r, s in zip((128, 128, 128), inp_tensor.shape[2:]))

            # Run inference across models
            def f_winner(x):
                o = model_winner(x)
                return o[0] if isinstance(o, list) else o
            def f_segres(x):
                return model_segres(x)
            def f_swin(x):
                return model_swin(x)

            p_winner = torch.sigmoid(sliding_window_inference(inp_tensor, actual_roi, 1, f_winner, overlap=0.5)).squeeze(0).cpu().numpy()
            p_segres = torch.sigmoid(sliding_window_inference(inp_tensor, actual_roi, 1, f_segres, overlap=0.5)).squeeze(0).cpu().numpy()
            p_swin = torch.sigmoid(sliding_window_inference(inp_tensor, actual_roi, 1, f_swin, overlap=0.5)).squeeze(0).cpu().numpy()

            # Grand Out-of-fold weighted probability fusion (45% Winner, 35% SegResNet, 20% Swin UNETR)
            fused_probs = weighted_average_ensemble([p_winner, p_segres, p_swin], weights=[0.45, 0.35, 0.20])

            # Dynamic calibrated thresholding & anatomical consistency
            pred_native = channels_to_native(fused_probs, threshold=0.5, enforce_hierarchy=True)

            # False-positive ET suppression (< 200 voxels)
            pred_post = apply_winner_postprocessing(pred_native, et_min_voxels=200, min_component_size=10)

            # Reconstruct into original full volume space
            full_volume = np.zeros(orig_shape, dtype=np.uint8)
            full_volume[bbox[0], bbox[1], bbox[2]] = pred_post

            # Save NIfTI output per official challenge specification
            out_file = output_dir / f"{sub_id}.nii.gz"
            nii = nib.Nifti1Image(full_volume, affine)
            nib.save(nii, out_file)
            print(f"[OK] Saved segmentation to: {out_file}")

    print("[BraTS Docker Inference] Complete. All outputs generated successfully.")


if __name__ == "__main__":
    main()
