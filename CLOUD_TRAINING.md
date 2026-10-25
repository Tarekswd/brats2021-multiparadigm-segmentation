# Training Task 1 (segmentation) on real data with a cloud GPU

This produces the first **measured** results for this repository. Report only what these runs log.

## 1. Kaggle notebook setup

1. On Kaggle, open the dataset **BRaTS 2021 Task 1 Dataset** (`dschettler8845/brats-2021-task1`) and accept the BraTS data terms (you have to do this yourself).
2. Create a new notebook and choose **Add Input → Datasets → brats-2021-task1**.
3. Under **Settings**, set the accelerator to **GPU** and turn **Internet on**.
4. Check the mount path and extract the training archive to local disk:
   ```bash
   !ls /kaggle/input/brats-2021-task1
   !mkdir -p /tmp/brats && tar -xf /kaggle/input/brats-2021-task1/BraTS2021_Training_Data.tar -C /tmp/brats && ls /tmp/brats | head && ls /tmp/brats | wc -l
   ```
   You should see `BraTS2021_XXXXX/` folders, each with `*_flair/_t1/_t1ce/_t2/_seg.nii.gz` files. If the archive unpacks into a sub-folder, point `--data-root` at that sub-folder.

## 2. Train

```bash
!git clone https://github.com/Tarekswd/brats2021-multiparadigm-segmentation.git && cd brats2021-multiparadigm-segmentation && pip install -q -r requirements.txt
```

```bash
%%bash
cd brats2021-multiparadigm-segmentation
python scripts/dataset_check.py --data-root /tmp/brats --output-dir /kaggle/working/dataset_check
python scripts/train.py --config configs/baseline_unet.yaml --data-root /tmp/brats \
  --epochs 30 --num-workers 4 --output-dir /kaggle/working/runs/baseline_unet
```

- `train.py` makes a seeded 75/25 patient-level train/validation split and validates on the full held-out volumes every epoch. That is not 5-fold cross-validation, so don't call it that.
- The script stops if `--data-root` doesn't exist. It never falls back to mock data when the flag is given.
- There is no resume-from-checkpoint. Time the first epoch, then pick an `--epochs` value that fits inside one Kaggle session.
- Run other configs (`winner_unet.yaml`, `segresnet.yaml`, `swin_unetr.yaml`) the same way, each in its own output directory.

## 3. Summarise honestly

```python
import pandas as pd, glob
for f in sorted(glob.glob("/kaggle/working/runs/*/training_log.csv")):
    d = pd.read_csv(f); last = d.iloc[-1]
    print(f, f"final epoch {int(last.epoch)}: WT {last.dice_wt:.3f} TC {last.dice_tc:.3f} ET {last.dice_et:.3f} | best avg {d.dice_avg.max():.3f} (optimistic)")
```

Put the **final-epoch** Dice in the README. Selecting the best epoch on the validation set makes that number optimistic. Commit the `training_log.csv` and `train.log` files under `experiments/` so the README numbers can be checked.
