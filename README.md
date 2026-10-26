# BraTS 2021 Brain Tumor Segmentation
## Research-Grade Winning-Architecture Extension & Multi-Paradigm Ensemble System

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.3+-green.svg)](https://monai.io/)

A complete, reproducible, research-grade deep learning system for the **RSNA-ASNR-MICCAI BraTS 2021 Challenge** (Task 1: Brain Tumor Segmentation). 

This repository implements architectures inspired by the BraTS 2021 winning submission (**KAIST MRI Lab — Luu & Park, arXiv:2112.04653**), the 2nd-place entry (**SegResNet**), and the 3D Vision Transformer **Swin UNETR**, plus a multi-paradigm ensemble pipeline.

> **Project status: one real-data run so far** — the 3D U-Net baseline, 5 epochs on the BraTS 2021 training set (see the table below).
> All other architectures are implemented but **not yet trained on real data**.
> Earlier versions of this repository contained invented benchmark tables, figures and a research report; they have been removed. Measured numbers live only in `results/measured/`. See [CLOUD_TRAINING.md](CLOUD_TRAINING.md) to reproduce.

---

## Table of Contents
1. [Key Architecture & Benchmark Highlights](#key-architecture--benchmark-highlights)
2. [Project Structure](#project-structure)
3. [Environment Installation](#environment-installation)
4. [Dataset Acquisition & Preparation](#dataset-acquisition--preparation)
5. [Dataset Validation](#dataset-validation)
6. [Training Pipeline](#training-pipeline)
7. [Validation & Metric Profiling](#validation--metric-profiling)
8. [Case Inference & Prediction](#case-inference--prediction)
9. [Evaluation & Metric Conventions](#evaluation--metric-conventions)
10. [Multi-Paradigm Ensembling](#multi-paradigm-ensembling)
11. [Pipeline Diagram](#pipeline-diagram)
12. [Hardware Requirements & Expected Runtime](#hardware-requirements--expected-runtime)
13. [Ablation Findings](#ablation-findings)
14. [Citation](#citation)
15. [License](#license)

---

## Key Architecture & Benchmark Highlights

**Measured results** (validation Dice after the final epoch; Kaggle T4 GPU, run on 2026-10-26):

| Model Architecture | Epochs | WT Dice (%) | TC Dice (%) | ET Dice (%) | Mean (%) | Data / split |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 3D U-Net Baseline (`configs/baseline_unet.yaml`) | 5 | 80.5 | 62.9 | 66.2 | 69.9 | BraTS 2021 training set (1251 cases), seeded 75/25 patient-level hold-out, single run |
| Winner-Inspired U-Net | – | not trained yet | | | | |

- Per-epoch log: [`results/measured/baseline_unet_kaggle_2026-10-26_training_log.csv`](results/measured/baseline_unet_kaggle_2026-10-26_training_log.csv). Each epoch took about 56–58 min.
- Only 5 epochs were trained and validation Dice was still trending upward, with dips (mean 61.2 % at epoch 1, 59.6 % at epoch 2, 69.9 % at epoch 5), so this is an **undertrained baseline**. Don't compare it with fully trained challenge entries. HD95 was not computed in this run.
- One hold-out split and one seed, with no test set, so the numbers carry no confidence interval.

For reference, published BraTS 2021 results come from the official challenge papers (e.g. Luu & Park, arXiv:2112.04653). They are not results of this repository.

---

## Project Structure

```
brats2021_research/
├── configs/                  # Modular YAML configuration files
│   ├── baseline_unet.yaml
│   ├── nnunet.yaml
│   ├── winner_unet.yaml
│   ├── segresnet.yaml
│   ├── unetr.yaml
│   ├── swin_unetr.yaml
│   ├── ensemble.yaml
│   └── final_model.yaml
├── src/
│   ├── data/                 # Data loading, z-score norm, 3D augmentation, splitting
│   ├── models/               # Architectures (Winner U-Net, SegResNet, Swin UNETR, etc.)
│   ├── losses/               # Compound Batch Dice, BCE, Focal, and Boundary losses
│   ├── training/             # AMP Trainer, Checkpointing, Schedulers, Callbacks
│   ├── evaluation/           # Dice, HD95, Sensitivity, Specificity, Postprocessing
│   ├── ensemble/             # Simple average, Weighted fusion, Hard/soft voting, Learned fusion
│   ├── visualization/        # Multi-modal overlays, Training curves, Error maps
│   ├── analysis/             # Model complementarity, Paired stats, Hard-case mining
│   └── utils/                # Config loader, Seed reproducibility, Hardware auto-tuning
├── scripts/                  # Command-Line Interface (CLI) execution tools
│   ├── dataset_check.py      # Dataset validation report generator
│   ├── train.py              # Training CLI
│   ├── validate.py           # Checkpoint validation CLI
│   ├── predict.py            # Case prediction CLI with NIfTI restoration
│   ├── evaluate.py           # Patient-level metric evaluation CLI
│   ├── ensemble.py           # Ensemble fusion CLI
│   └── visualize.py          # Pipeline diagram (Figure 1)
├── tests/                    # Comprehensive unit test suite
├── experiments/              # Model weights and logs
├── results/
│   ├── figures/figure1_pipeline.png   # Pipeline diagram
│   └── measured/             # Measured training logs from real-data runs
├── requirements.txt
├── environment.yml
├── CLOUD_TRAINING.md         # How to train on a Kaggle GPU
└── LICENSE
```

---

## Environment Installation

### Using Conda:
```bash
conda env create -f environment.yml
conda activate brats2021
```

### Using Pip:
```bash
pip install -r requirements.txt
```

---

## Dataset Acquisition & Preparation

1. Request official access from the [BraTS 2021 Challenge Portal](https://www.med.upenn.edu/cbica/brats2021/).
2. Organize your dataset directory with standard naming:
```
/path/to/BraTS2021/
  ├── BraTS2021_00000/
  │   ├── BraTS2021_00000_t1.nii.gz
  │   ├── BraTS2021_00000_t1ce.nii.gz
  │   ├── BraTS2021_00000_t2.nii.gz
  │   ├── BraTS2021_00000_flair.nii.gz
  │   └── BraTS2021_00000_seg.nii.gz
```

---

## Dataset Validation

Before executing any model training, verify dataset integrity, NIfTI readability, dimension alignment, and label sets:
```bash
python scripts/dataset_check.py --data-root /path/to/BraTS2021 --output-dir results/
```
Generates `results/dataset_report.json`, `.csv`, and `.md`.

---

## Training Pipeline

Train any model using its corresponding YAML configuration:

```bash
# Train BraTS 2021 Winner-Inspired U-Net (KAIST reproduction)
python scripts/train.py --config configs/winner_unet.yaml --data-root /path/to/BraTS2021 --output-dir experiments/winner_unet

# Train SegResNet Baseline
python scripts/train.py --config configs/segresnet.yaml --output-dir experiments/segresnet

# Train Swin UNETR Baseline
python scripts/train.py --config configs/swin_unetr.yaml --output-dir experiments/swin_unetr
```

---

## Validation & Metric Profiling

Evaluate saved checkpoints across patient validation splits using 3D sliding-window inference:
```bash
python scripts/validate.py \
    --checkpoint experiments/winner_unet/checkpoints/best_model.pt \
    --config configs/winner_unet.yaml \
    --output-dir results/validation
```

---

## Case Inference & Prediction

Run inference on an unseen patient case, with optional Test-Time Augmentation (TTA), and export native NIfTI segmentation:
```bash
python scripts/predict.py \
    --input /path/to/BraTS2021_00001 \
    --checkpoint experiments/winner_unet/checkpoints/best_model.pt \
    --config configs/winner_unet.yaml \
    --output results/predictions/ \
    --tta
```

---

## Evaluation & Metric Conventions

Evaluate prediction NIfTIs against ground truth labels:
```bash
python scripts/evaluate.py \
    --predictions results/predictions \
    --ground-truth /path/to/BraTS2021 \
    --output-dir results/evaluation
```
*Note on Empty Regions:* In accordance with official BraTS rules, when ground truth and prediction are both empty, Dice = 1.0 and HD95 = 0.0 mm. False-positive predictions on empty targets incur Dice = 0.0 and HD95 = 373.13 mm penalty distance.

---

## Multi-Paradigm Ensembling

Execute synergistic out-of-fold fusion:
```bash
python scripts/ensemble.py --config configs/ensemble.yaml --output-dir results/ensemble
```

---

## Pipeline Diagram

Regenerate the pipeline diagram (`results/figures/figure1_pipeline.png`):
```bash
python scripts/visualize.py
```

---

## Hardware Requirements & Expected Runtime

- **Minimal / CPU Test:** 16 GB System RAM, any multi-core CPU. Uses $64^3$ patch sizes.
- **Recommended GPU:** NVIDIA RTX 3090, RTX 4090, A5000, or A100 (12+ GB VRAM).
- **Auto-Tuning:** The hardware manager dynamically selects patch size ($128^3$ or $96^3$) and gradient accumulation steps according to detected VRAM.

---

## Ablation Findings

No ablations have been run yet. The design choices below are implemented, but their effect on this codebase has not been measured:

1. **Enlarged encoder** (channels up to 512)
2. **Group Normalization** instead of BatchNorm
3. **Deep supervision** with auxiliary heads
4. **ET thresholding** (small ET components < 200 voxels remapped to necrotic core)
5. **Multi-paradigm fusion** of CNN and Swin UNETR outputs

---

## Citation

```bibtex
@inproceedings{luu2021extending,
  title={Extending nn-UNet for Brain Tumor Segmentation},
  author={Luu, Huan Minh and Park, Sung-Hong},
  booktitle={International MICCAI Brainlesion Workshop},
  pages={173--186},
  year={2021},
  publisher={Springer}
}

@article{baid2021rsna,
  title={The RSNA-ASNR-MICCAI BraTS 2021 benchmark on brain tumor segmentation and radiogenomic classification},
  author={Baid, Ujjwal and Ghodasara, Satyam and Mohan, Suyash and Bilello, Michel and Calabrese, Evan and Colak, Errol and Farahani, Keyvan and Kalpathy-Cramer, Jayashree and Kitchen, Frank and Lichtenberg, Tammy and others},
  journal={arXiv preprint arXiv:2107.02314},
  year={2021}
}
```

---

## License

This project is open-source under the [MIT License](LICENSE).
