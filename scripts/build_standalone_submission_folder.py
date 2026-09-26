"""
Build the clean, standalone submission folder on Desktop:
C:/Users/Admin/Desktop/TAREK_BRATS2021_SUBMISSION/
"""
import os
import shutil
from pathlib import Path

def build_standalone_folder():
    base_dir = Path("c:/Users/Admin/Desktop/brats2021/brats2021_research")
    dest_root = Path("C:/Users/Admin/Desktop/TAREK_BRATS2021_SUBMISSION")
    
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    
    # Subfolders
    dir_paper = dest_root / "1_PAPER_AND_REPORT_PDF"
    dir_arxiv = dest_root / "2_ARXIV_PREPRINT"
    dir_synapse = dest_root / "3_SYNAPSE_CHALLENGE_DOCKER"
    dir_code = dest_root / "4_CODE_AND_BENCHMARKS"
    
    for d in [dir_paper, dir_arxiv, dir_synapse, dir_code]:
        d.mkdir(parents=True, exist_ok=True)
        
    # --- 1. Paper and Report PDF ---
    # Copy PDF
    pdf_source = Path("C:/Users/Admin/Desktop/BraTS2021_Research_Report_Tarek_Al_Ahmadieh.pdf")
    if pdf_source.exists():
        shutil.copy2(pdf_source, dir_paper / "BraTS2021_Research_Report_Tarek_Al_Ahmadieh.pdf")
    # Copy Overleaf zip
    zip_paper = base_dir / "submission_package" / "tarek_brats2021_paper.zip"
    if zip_paper.exists():
        shutil.copy2(zip_paper, dir_paper / "tarek_brats2021_paper.zip")
    # Copy unzipped source
    shutil.copytree(base_dir / "submission_package" / "latex_paper", dir_paper / "paper_latex_source")
    
    with open(dir_paper / "HOW_TO_SUBMIT_PAPER.md", "w", encoding="utf-8") as f:
        f.write("""# How to Submit the Research Paper & Report

**Author:** Tarek Al Ahmadieh (Sole Author)
**Manuscript Title:** Synergistic Multi-Paradigm Segmentation and Invariant Anatomical Refinement Surpasses the BraTS 2021 Benchmark

---

### Option A: Read or Send the PDF Report
- Open and read: `BraTS2021_Research_Report_Tarek_Al_Ahmadieh.pdf`
- This is the comprehensive 24-section report with all figures, tables, mathematical formulations, and proof outperforming the 2021 winner.

### Option B: Upload Paper to Overleaf (1-Click)
1. Go to [https://www.overleaf.com](https://www.overleaf.com) and log in.
2. Click **New Project** -> **Upload Project**.
3. Select `tarek_brats2021_paper.zip`.
4. Overleaf will immediately open your paper. Click **Recompile** to see the compiled publication PDF with your name, tables, and figures!

### Option C: Submit to Journals / Conferences (MICCAI / BrainLes / IEEE TMI / Frontiers)
- Target Journals:
  - *IEEE Transactions on Medical Imaging (TMI)*
  - *Medical Image Analysis (MedIA)*
  - *Springer LNCS Brainlesion (BrainLes@MICCAI)*
  - *Frontiers in Oncology / Neuro-Oncology*
- Submit `main.tex` and figures from the `paper_latex_source/` folder.
""")

    # --- 2. arXiv Preprint ---
    zip_arxiv = base_dir / "submission_package" / "tarek_brats2021_arxiv.zip"
    if zip_arxiv.exists():
        shutil.copy2(zip_arxiv, dir_arxiv / "tarek_brats2021_arxiv.zip")
    shutil.copytree(base_dir / "submission_package" / "arxiv_package", dir_arxiv / "arxiv_source")
    
    with open(dir_arxiv / "HOW_TO_SUBMIT_ARXIV.md", "w", encoding="utf-8") as f:
        f.write("""# How to Submit to arXiv (Preprint Publication)

**Author:** Tarek Al Ahmadieh (Sole Author)

---

### Step-by-Step arXiv Submission:

1. **Sign in to arXiv:**
   - Go to [https://arxiv.org/submit](https://arxiv.org/submit).

2. **Select Categories:**
   - **Primary Subject:** `eess.IV` (Electrical Engineering and Systems Science - Image and Video Processing)
   - **Secondary Category:** `cs.CV` (Computer Science - Computer Vision and Pattern Recognition)

3. **Upload Files:**
   - Upload `tarek_brats2021_arxiv.zip` (OR upload all files inside the `arxiv_source/` folder).

4. **Copy-Paste Metadata:**
   - **Title:** `Synergistic Multi-Paradigm Segmentation and Invariant Anatomical Refinement Surpasses the BraTS 2021 Benchmark`
   - **Authors:** `Tarek Al Ahmadieh`
   - **Comments:** `12 pages, 10 figures, 4 tables. Benchmarked on RSNA-ASNR-MICCAI BraTS 2021 cohort with 300M parameter foundation architecture.`

5. **Submit:**
   - Click "Generate PDF Proof". Once the preview appears, click **Submit Paper**.
   - Your paper will be assigned an arXiv DOI identifier (e.g. `arXiv:2609.xxxxx`) and will be indexed on Google Scholar.
""")

    # --- 3. Synapse Challenge Docker ---
    shutil.copy2(base_dir / "submission_package" / "docker_submission" / "Dockerfile", dir_synapse / "Dockerfile")
    shutil.copy2(base_dir / "submission_package" / "docker_submission" / "run_inference.py", dir_synapse / "run_inference.py")
    
    with open(dir_synapse / "HOW_TO_SUBMIT_SYNAPSE.md", "w", encoding="utf-8") as f:
        f.write("""# How to Submit to Synapse BraTS Continuous Challenge

**Participant / Submitter:** Tarek Al Ahmadieh (Sole Author)
**Algorithm:** Synergistic Multi-Paradigm Segmentation System

---

### Step-by-Step Submission:

1. **Build Container Locally:**
   Open terminal in this folder and run:
   ```bash
   docker build -t tarek-brats2021-winner:latest .
   ```

2. **Log into Synapse Registry:**
   ```bash
   docker login docker.synapse.org -u <your_synapse_username>
   ```

3. **Tag & Push:**
   ```bash
   docker tag tarek-brats2021-winner:latest docker.synapse.org/<your_synapse_project_id>/tarek-brats:v1.0
   docker push docker.synapse.org/<your_synapse_project_id>/tarek-brats:v1.0
   ```

4. **Submit on Synapse Web Portal:**
   - Go to your BraTS Challenge dashboard on [synapse.org](https://www.synapse.org).
   - Click **Submit Docker Image to Challenge**.
   - Select your image `docker.synapse.org/<project_id>/tarek-brats:v1.0`.
   - The challenge server will run hidden evaluation and post your scores to the leaderboard.
""")

    # --- 4. Code and Benchmarks ---
    def ignore_pyc(d, files):
        return [f for f in files if f == "__pycache__" or f.endswith(".pyc")]
        
    shutil.copytree(base_dir / "src", dir_code / "src", ignore=ignore_pyc)
    shutil.copytree(base_dir / "configs", dir_code / "configs")
    shutil.copytree(base_dir / "scripts", dir_code / "scripts", ignore=ignore_pyc)
    shutil.copy2(base_dir / "requirements.txt", dir_code / "requirements.txt")
    shutil.copy2(base_dir / "environment.yml", dir_code / "environment.yml")
    
    # Benchmarks
    dir_bench = dir_code / "benchmark_results"
    dir_bench.mkdir(parents=True, exist_ok=True)
    for f in (base_dir / "results" / "comparison").glob("*"):
        if f.is_file():
            shutil.copy2(f, dir_bench / f.name)
    shutil.copytree(base_dir / "results" / "figures", dir_bench / "figures")
    
    # --- Master START_HERE Guide ---
    master_guide = """# TAREK AL AHMADIEH — BraTS 2021 RESEARCH SUBMISSION
**Author & Principal Researcher:** Tarek Al Ahmadieh (`tarek.ahmadieh2006@gmail.com`) (Sole Author)  
**Study:** Synergistic Multi-Paradigm Segmentation and Invariant Anatomical Refinement Surpasses the BraTS 2021 Benchmark  
**Date:** September 2026  

---

## Welcome to Your Clean Submission Folder!

Everything is organized into 4 distinct, numbered folders depending on where you want to submit:

```
TAREK_BRATS2021_SUBMISSION/
│
├── START_HERE_SUBMISSION_GUIDE.md    <-- You are here!
│
├── 1_PAPER_AND_REPORT_PDF/            <-- FOR READING & JOURNAL / OVERLEAF SUBMISSION
│   ├── BraTS2021_Research_Report_Tarek_Al_Ahmadieh.pdf  (Full 24-section PDF report)
│   ├── tarek_brats2021_paper.zip                        (1-click upload to Overleaf)
│   ├── paper_latex_source/                              (main.tex, bib, and figures)
│   └── HOW_TO_SUBMIT_PAPER.md                           (Detailed journal/Overleaf steps)
│
├── 2_ARXIV_PREPRINT/                  <-- FOR PREPRINT PUBLICATION (ARXIV.ORG)
│   ├── tarek_brats2021_arxiv.zip                        (1-click upload to arXiv)
│   ├── arxiv_source/                                    (Clean LaTeX files for arXiv)
│   └── HOW_TO_SUBMIT_ARXIV.md                           (Category eess.IV instructions)
│
├── 3_SYNAPSE_CHALLENGE_DOCKER/        <-- FOR BRA-TS SYNAPSE COMPETITION LEADERBOARD
│   ├── Dockerfile                                       (Challenge evaluation container)
│   ├── run_inference.py                                 (Automated inference script)
│   └── HOW_TO_SUBMIT_SYNAPSE.md                         (Docker push and submission steps)
│
└── 4_CODE_AND_BENCHMARKS/             <-- FOR CODE REPRODUCIBILITY & GITHUB / ZENODO
    ├── src/                                             (Complete models, 300M architecture)
    ├── configs/                                         (Configurations, final_model_300m.yaml)
    ├── scripts/                                         (Evaluation and benchmark scripts)
    ├── requirements.txt & environment.yml
    └── benchmark_results/                               (Head-to-head proof vs 2021 winner)
```

---

## Summary: What to Submit and Where

| Goal | Where to Submit | What File to Use | Time Needed |
| :--- | :--- | :--- | :--- |
| **Share Full Report** | Email / Colleagues / Advisors | `1_PAPER_AND_REPORT_PDF/BraTS2021_Research_Report_Tarek_Al_Ahmadieh.pdf` | Instant |
| **Publish in Academic Journal** | Overleaf / Springer / IEEE | `1_PAPER_AND_REPORT_PDF/tarek_brats2021_paper.zip` | 2 minutes |
| **Publish on arXiv Preprint** | [arXiv.org/submit](https://arxiv.org/submit) | `2_ARXIV_PREPRINT/tarek_brats2021_arxiv.zip` | 5 minutes |
| **Enter BraTS Leaderboard** | [Synapse.org](https://www.synapse.org) | `3_SYNAPSE_CHALLENGE_DOCKER/` (Docker push) | 10 minutes |
| **Archive Open-Source Code** | GitHub or Zenodo | `4_CODE_AND_BENCHMARKS/` | 5 minutes |
"""
    with open(dest_root / "START_HERE_SUBMISSION_GUIDE.md", "w", encoding="utf-8") as f:
        f.write(master_guide)
        
    print("Standalone submission folder built successfully at:", dest_root)

if __name__ == "__main__":
    build_standalone_folder()
