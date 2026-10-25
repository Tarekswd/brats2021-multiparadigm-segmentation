"""
PyTorch Dataset implementation for BraTS 2021 Brain Tumor Segmentation.
Supports patch-based training with foreground oversampling, full-volume inference,
augmentation, and 3-region sigmoid targets (WT, TC, ET).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import preprocess_case
from src.data.augmentation import BraTSAugmentor
from src.evaluation.brats_regions import regions_to_channels


class BraTSDataset(Dataset):
    """
    BraTS 2021 Dataset supporting patch-based training and full-volume evaluation.
    """

    def __init__(
        self,
        subject_dirs: List[Union[str, Path]],
        patch_size: Optional[Tuple[int, int, int]] = (128, 128, 128),
        is_training: bool = True,
        norm_method: str = "zscore",
        add_foreground_channel: bool = False,
        augmentor: Optional[BraTSAugmentor] = None,
        fg_oversample_prob: float = 0.67,  # nnU-Net foreground patch sampling ratio
        target_mode: str = "regions"  # 'regions' (3 sigmoid channels) or 'multiclass' (0,1,2,4)
    ):
        self.subject_dirs = [Path(p) for p in subject_dirs]
        self.patch_size = patch_size
        self.is_training = is_training
        self.norm_method = norm_method
        self.add_foreground_channel = add_foreground_channel
        self.augmentor = augmentor if (is_training and augmentor) else None
        self.fg_oversample_prob = fg_oversample_prob
        self.target_mode = target_mode

    def __len__(self) -> int:
        return len(self.subject_dirs)

    def _get_modality_paths(self, subject_dir: Path) -> Tuple[Dict[str, str], Optional[str]]:
        """Resolve file paths for the four modalities and segmentation."""
        mod_paths = {}
        for m in ["flair", "t1", "t1ce", "t2"]:
            matches = list(subject_dir.glob(f"*{m}.nii*"))
            if matches:
                mod_paths[m] = str(matches[0])
            else:
                raise FileNotFoundError(f"Missing {m} in {subject_dir}")

        seg_matches = list(subject_dir.glob("*seg.nii*"))
        seg_path = str(seg_matches[0]) if seg_matches else None
        return mod_paths, seg_path

    def _sample_patch(
        self,
        image: np.ndarray,
        seg: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract a 3D patch of shape self.patch_size from image [C, D, H, W]
        and seg [D, H, W]. Oversamples tumor foreground voxels when possible.
        """
        p_d, p_h, p_w = self.patch_size
        _, d, h, w = image.shape

        # Pad volume if smaller than patch size
        pad_d = max(0, p_d - d)
        pad_h = max(0, p_h - h)
        pad_w = max(0, p_w - w)

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            pad_width = ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w))
            image = np.pad(image, pad_width, mode="constant", constant_values=0)
            if seg is not None:
                seg = np.pad(seg, ((0, pad_d), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
            _, d, h, w = image.shape

        # Choose patch origin
        sample_fg = (seg is not None) and (np.sum(seg > 0) > 0) and (random.random() < self.fg_oversample_prob)
        if sample_fg:
            fg_indices = np.where(seg > 0)
            rand_idx = random.randint(0, len(fg_indices[0]) - 1)
            c_d, c_h, c_w = fg_indices[0][rand_idx], fg_indices[1][rand_idx], fg_indices[2][rand_idx]

            start_d = np.clip(c_d - p_d // 2, 0, d - p_d)
            start_h = np.clip(c_h - p_h // 2, 0, h - p_h)
            start_w = np.clip(c_w - p_w // 2, 0, w - p_w)
        else:
            start_d = random.randint(0, d - p_d) if d > p_d else 0
            start_h = random.randint(0, h - p_h) if h > p_h else 0
            start_w = random.randint(0, w - p_w) if w > p_w else 0

        patch_img = image[:, start_d : start_d + p_d, start_h : start_h + p_h, start_w : start_w + p_w]
        patch_seg = seg[start_d : start_d + p_d, start_h : start_h + p_h, start_w : start_w + p_w] if seg is not None else None

        return patch_img, patch_seg

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        subject_dir = self.subject_dirs[idx]
        mod_paths, seg_path = self._get_modality_paths(subject_dir)

        # Preprocessing: load, tight crop, normalize
        preprocessed = preprocess_case(
            modality_paths=mod_paths,
            seg_path=seg_path,
            norm_method=self.norm_method,
            tight_crop=self.is_training,  # crop tightly during training
            add_foreground_channel=self.add_foreground_channel
        )

        image = preprocessed["image"]
        seg = preprocessed["seg"]

        if self.is_training and self.patch_size is not None:
            image, seg = self._sample_patch(image, seg)

        if self.augmentor is not None and self.is_training:
            image, seg = self.augmentor(image, seg)

        item = {
            "image": torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)),
            "subject_id": subject_dir.name
        }

        if seg is not None:
            if self.target_mode == "regions":
                # Convert to [3, D, H, W] for WT, TC, ET
                target_channels = regions_to_channels(seg)
                item["target"] = torch.from_numpy(np.ascontiguousarray(target_channels, dtype=np.float32))
            else:
                item["target"] = torch.from_numpy(np.ascontiguousarray(seg, dtype=np.int64))

        return item
