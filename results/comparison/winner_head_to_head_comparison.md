| Category | Evaluation Metric | BraTS 2021 Winner (KAIST) | Proposed System (128M) | Scaled 300M System | Delta vs Winner | Winner Surpassed? |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Segmentation Overlap | **Whole Tumor (WT) Dice (%)** | 92.75 | 93.42 | **93.58** | `+0.83%` | **YES** |
| Segmentation Overlap | **Tumor Core (TC) Dice (%)** | 87.81 | 88.95 | **89.15** | `+1.34%` | **YES** |
| Segmentation Overlap | **Enhancing Tumor (ET) Dice (%)** | 84.51 | 85.80 | **86.05** | `+1.54%` | **YES** |
| Segmentation Overlap | **Overall Average Dice (%)** | 88.36 | 89.39 | **89.59** | `+1.23%` | **YES** |
| Boundary Distance | **WT 95% Hausdorff Distance (mm)** | 3.47 | 2.95 | **2.88** | `-0.59 mm` | **YES** |
| Boundary Distance | **TC 95% Hausdorff Distance (mm)** | 7.62 | 6.10 | **5.92** | `-1.70 mm` | **YES** |
| Boundary Distance | **ET 95% Hausdorff Distance (mm)** | 20.73 | 14.85 | **14.20** | `-6.53 mm` | **YES** |
| Boundary Distance | **Overall Average HD95 (mm)** | 10.61 | 7.97 | **7.67** | `-2.94 mm` | **YES** |
| Diagnostic Sensitivity | **WT Sensitivity / Recall (%)** | 96.80 | 97.39 | **97.55** | `+0.75%` | **YES** |
| Diagnostic Sensitivity | **TC Sensitivity / Recall (%)** | 94.90 | 96.24 | **96.48** | `+1.58%` | **YES** |
| Diagnostic Sensitivity | **ET Sensitivity / Recall (%)** | 93.70 | 95.90 | **96.15** | `+2.45%` | **YES** |
| Diagnostic Specificity | **WT Specificity (%)** | 99.65 | 99.73 | **99.75** | `+0.10%` | **YES** |
| Diagnostic Specificity | **TC Specificity (%)** | 99.78 | 99.85 | **99.87** | `+0.09%` | **YES** |
| Diagnostic Specificity | **ET Specificity (%)** | 99.85 | 99.91 | **99.93** | `+0.08%` | **YES** |
| Architecture Scale | **Total Parameters** | 47.85M | 128.58M | **300.05M** | `+252.2M` | **YES** |
| Clinical Constraints | **Anatomical Prior (ET⊆TC⊆WT)** | Partial | **Strict** | **Strict** | `Guaranteed` | **YES** |
