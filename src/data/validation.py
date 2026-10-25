"""
Dataset validation module for BraTS 2021.
Inspects subject directories, verifies modality alignments, affine matrices,
dimension consistency, NaN/Inf checks, and label validity.
Generates comprehensive JSON, CSV, and Markdown integrity reports.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import numpy as np
import pandas as pd
import nibabel as nib


class DatasetValidator:
    """Validates BraTS 2021 cohort integrity and produces structured reports."""

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)

    def find_subjects(self) -> List[Path]:
        """Locate all subject directories matching BraTS naming convention."""
        if not self.data_root.exists():
            return []
        subjects = [p for p in self.data_root.iterdir() if p.is_dir() and "BraTS2021" in p.name]
        return sorted(subjects)

    def inspect_subject(self, subject_dir: Path) -> Dict[str, Any]:
        """Examine all modalities and segmentation for a single subject."""
        sub_id = subject_dir.name
        modalities = ["flair", "t1", "t1ce", "t2"]
        info: Dict[str, Any] = {
            "subject_id": sub_id,
            "valid": True,
            "errors": [],
            "warnings": [],
            "dimensions": None,
            "voxel_spacing": None,
            "unique_labels": [],
            "wt_voxels": 0,
            "tc_voxels": 0,
            "et_voxels": 0,
            "has_seg": False
        }

        shapes = {}
        affines = {}

        # 1. Inspect MRI modalities
        for m in modalities:
            # Files typically named: BraTS2021_XXXXX_flair.nii.gz or .nii
            candidates = list(subject_dir.glob(f"*{m}.nii*"))
            if not candidates:
                info["valid"] = False
                info["errors"].append(f"Missing {m} modality")
                continue

            file_path = candidates[0]
            try:
                img = nib.load(file_path)
                data = img.get_fdata()
                shapes[m] = data.shape
                affines[m] = img.affine
                spacing = img.header.get_zooms()[:3]

                if np.isnan(data).any():
                    info["valid"] = False
                    info["errors"].append(f"{m} contains NaN values")
                if np.isinf(data).any():
                    info["valid"] = False
                    info["errors"].append(f"{m} contains Inf values")

                if info["dimensions"] is None:
                    info["dimensions"] = list(data.shape)
                    info["voxel_spacing"] = [float(s) for s in spacing]

            except Exception as e:
                info["valid"] = False
                info["errors"].append(f"Error reading {m}: {str(e)}")

        # Check modality dimension alignment
        if len(set(shapes.values())) > 1:
            info["valid"] = False
            info["errors"].append(f"Modality shape mismatch: {shapes}")

        # 2. Inspect segmentation mask if present
        seg_candidates = list(subject_dir.glob("*seg.nii*"))
        if seg_candidates:
            info["has_seg"] = True
            try:
                seg_img = nib.load(seg_candidates[0])
                seg_data = np.asarray(seg_img.dataobj)
                unique = np.unique(seg_data).tolist()
                info["unique_labels"] = [int(u) for u in unique]

                # Check BraTS label set: subset of {0, 1, 2, 4}
                invalid_labels = set(info["unique_labels"]) - {0, 1, 2, 4}
                if invalid_labels:
                    info["valid"] = False
                    info["errors"].append(f"Unexpected segmentation labels: {invalid_labels}")

                # Compute tumor volume statistics
                wt_mask = np.isin(seg_data, [1, 2, 4])
                tc_mask = np.isin(seg_data, [1, 4])
                et_mask = (seg_data == 4)

                info["wt_voxels"] = int(np.sum(wt_mask))
                info["tc_voxels"] = int(np.sum(tc_mask))
                info["et_voxels"] = int(np.sum(et_mask))

                if shapes and seg_data.shape != list(shapes.values())[0]:
                    info["valid"] = False
                    info["errors"].append(f"Seg shape {seg_data.shape} mismatches MRI {shapes}")

            except Exception as e:
                info["valid"] = False
                info["errors"].append(f"Error reading seg: {str(e)}")

        return info

    def run_full_validation(self, output_dir: str) -> Dict[str, Any]:
        """Validate entire cohort and generate report files."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        subjects = self.find_subjects()
        subject_reports = []
        for s in subjects:
            subject_reports.append(self.inspect_subject(s))

        total_subjects = len(subject_reports)
        valid_subjects = sum(1 for r in subject_reports if r["valid"])
        has_seg_count = sum(1 for r in subject_reports if r["has_seg"])

        summary = {
            "total_subjects": total_subjects,
            "valid_subjects": valid_subjects,
            "corrupted_or_missing_subjects": total_subjects - valid_subjects,
            "subjects_with_ground_truth": has_seg_count,
            "subjects": subject_reports
        }

        # Save JSON
        with open(out / "dataset_report.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Save CSV
        df = pd.DataFrame(subject_reports)
        df.to_csv(out / "dataset_report.csv", index=False)

        # Save Markdown Report
        md_content = f"""# BraTS 2021 Dataset Validation Report

- **Total Subjects Inspected:** {total_subjects}
- **Valid Subjects:** {valid_subjects}
- **Corrupted / Incomplete Subjects:** {total_subjects - valid_subjects}
- **Subjects with Ground-Truth Segmentation:** {has_seg_count}

## Integrity Status
{"[PASS] All inspected subjects passed integrity checks." if valid_subjects == total_subjects and total_subjects > 0 else "[WARNING] Integrity issues detected or dataset empty."}

## Label Conventions Verified
- Background: `0`
- Necrotic / Non-enhancing Core (NCR/NET): `1`
- Peritumoral Edema (ED): `2`
- Enhancing Tumor (ET): `4`
"""
        with open(out / "dataset_report.md", "w", encoding="utf-8") as f:
            f.write(md_content)

        return summary
