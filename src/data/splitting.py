"""
Subject-level dataset splitting for BraTS 2021.
Ensures zero data leakage across train, validation, and test subsets.
All splits operate strictly at the patient/subject level.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.model_selection import KFold


def create_subject_splits(
    subject_ids: List[str],
    val_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 42,
    output_dir: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    Split subject IDs into train, validation, and optional test sets.
    """
    rng = np.random.RandomState(seed)
    shuffled = list(subject_ids)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_test = int(round(n_total * test_fraction))
    n_val = int(round(n_total * val_fraction))

    test_ids = sorted(shuffled[:n_test])
    val_ids = sorted(shuffled[n_test : n_test + n_val])
    train_ids = sorted(shuffled[n_test + n_val :])

    # Integrity verification: Mutual exclusivity
    assert len(set(train_ids) & set(val_ids)) == 0, "Data leakage detected: train and val overlap!"
    assert len(set(train_ids) & set(test_ids)) == 0, "Data leakage detected: train and test overlap!"
    assert len(set(val_ids) & set(test_ids)) == 0, "Data leakage detected: val and test overlap!"

    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for split_name, ids in splits.items():
            split_file = out / f"{split_name}.json"
            with open(split_file, "w", encoding="utf-8") as f:
                json.dump({"subjects": ids, "count": len(ids), "seed": seed}, f, indent=2)

    return splits


def create_kfold_splits(
    subject_ids: List[str],
    n_splits: int = 5,
    seed: int = 42,
    output_dir: Optional[str] = None
) -> List[Dict[str, List[str]]]:
    """
    Generate deterministic 5-fold cross validation subject-level splits.
    """
    subject_ids = sorted(subject_ids)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(subject_ids)):
        train_ids = [subject_ids[i] for i in train_idx]
        val_ids = [subject_ids[i] for i in val_idx]

        assert len(set(train_ids) & set(val_ids)) == 0, f"Fold {fold_idx} has train/val overlap!"

        fold_data = {
            "fold": fold_idx,
            "train": train_ids,
            "val": val_ids
        }
        folds.append(fold_data)

        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            with open(out / f"fold_{fold_idx}.json", "w", encoding="utf-8") as f:
                json.dump(fold_data, f, indent=2)

    return folds
