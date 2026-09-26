"""
Synthetic Mock BraTS Patient Generator for testing and pipeline validation.
Creates small 3D NIfTI files with all 4 modalities and valid segmentation labels.
"""

from pathlib import Path
import numpy as np
import nibabel as nib


def generate_mock_subject(
    subject_dir: Path,
    shape: tuple = (48, 64, 64)
) -> None:
    """Generate mock NIfTI files for a single BraTS subject."""
    subject_dir.mkdir(parents=True, exist_ok=True)
    sub_id = subject_dir.name
    affine = np.eye(4)

    d, h, w = shape
    # Modalities
    t1 = np.random.uniform(0.1, 0.9, size=shape).astype(np.float32)
    t1ce = np.random.uniform(0.1, 0.9, size=shape).astype(np.float32)
    t2 = np.random.uniform(0.1, 0.9, size=shape).astype(np.float32)
    flair = np.random.uniform(0.1, 0.9, size=shape).astype(np.float32)

    # Segmentation: labels {0, 1, 2, 4}
    seg = np.zeros(shape, dtype=np.uint8)
    # Tumor sphere
    seg[16:32, 20:44, 20:44] = 2  # Edema
    seg[20:28, 26:38, 26:38] = 1  # Necrotic Core
    seg[22:26, 28:34, 28:34] = 4  # Enhancing Tumor

    nib.save(nib.Nifti1Image(t1, affine), subject_dir / f"{sub_id}_t1.nii.gz")
    nib.save(nib.Nifti1Image(t1ce, affine), subject_dir / f"{sub_id}_t1ce.nii.gz")
    nib.save(nib.Nifti1Image(t2, affine), subject_dir / f"{sub_id}_t2.nii.gz")
    nib.save(nib.Nifti1Image(flair, affine), subject_dir / f"{sub_id}_flair.nii.gz")
    nib.save(nib.Nifti1Image(seg, affine), subject_dir / f"{sub_id}_seg.nii.gz")
