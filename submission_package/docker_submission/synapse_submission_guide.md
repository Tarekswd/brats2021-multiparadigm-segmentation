# BraTS Challenge Docker Submission Guide
**Author & Submitter:** Tarek Al Ahmadieh  
**Team / Algorithm:** Synergistic Multi-Paradigm Brain Tumor Segmentation Framework  
**Target:** Synapse.org BraTS Continuous Leaderboard / MedPerf / RSNA-MICCAI Evaluation  

---

## 1. Overview
This package builds a self-contained, reproducible Docker image conforming to the official BraTS Synapse challenge guidelines. The container automatically:
1. Mounts test volumes from `/input` (FLAIR, T1, T1ce, T2 mpMRI series).
2. Performs z-score intensity normalization on non-zero brain voxels.
3. Executes multi-paradigm sliding-window inference with test-time augmentation (TTA).
4. Applies strict anatomical prior projection ($ET \subseteq TC \subseteq WT$) and $<200$ voxel ET cleanup.
5. Saves predicted segmentation masks (`.nii.gz`) with labels 1 (NCR), 2 (ED), and 4 (ET) directly to `/output`.

---

## 2. Building the Docker Container Locally

From your terminal in `submission_package/docker_submission/`:
```bash
docker build -t tarek-brats2021-winner:latest .
```

---

## 3. Testing the Container Locally

Run a test inference on your local validation data:
```bash
docker run --rm \
  -v "C:/path/to/validation_cases:/input:ro" \
  -v "C:/path/to/predictions:/output:rw" \
  tarek-brats2021-winner:latest
```

Verify that the output directory contains matching `.nii.gz` files with standard BraTS label values (`0, 1, 2, 4`).

---

## 4. Submitting to Synapse.org Leaderboard

1. **Log into Synapse Docker Registry:**
   ```bash
   docker login docker.synapse.org -u <your_synapse_username>
   ```

2. **Tag your image with your Synapse Project ID:**
   ```bash
   docker tag tarek-brats2021-winner:latest docker.synapse.org/<synapse_project_id>/tarek-brats-segmentation:v1.0
   ```

3. **Push to Synapse Registry:**
   ```bash
   docker push docker.synapse.org/<synapse_project_id>/tarek-brats-segmentation:v1.0
   ```

4. **Submit via Synapse Web Portal:**
   - Navigate to your Synapse BraTS Challenge submission page.
   - Select Docker image `docker.synapse.org/<synapse_project_id>/tarek-brats-segmentation:v1.0`.
   - Author: **Tarek Al Ahmadieh**.
   - Click **Submit to Evaluation Queue**.
   - The automated evaluation server will run inference on the hidden test set and populate your scores on the public leaderboard.
