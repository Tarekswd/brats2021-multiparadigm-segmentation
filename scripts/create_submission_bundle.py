"""
Script to assemble and compress the complete Master Submission Bundle for Tarek Al Ahmadieh.
Bundles LaTeX paper, arXiv package, Docker submission, full 24-section research report,
modular source code, configs, benchmark comparisons, and figures.
"""
import os
import shutil
import zipfile
from pathlib import Path

def create_bundle():
    base_dir = Path("c:/Users/Admin/Desktop/brats2021/brats2021_research")
    bundle_dir = base_dir / "tarek_submission_bundle"
    
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Master Submission Guide
    guide_content = """# Master Research Submission Guide
**Author & Principal Researcher:** Tarek Al Ahmadieh (tarek.ahmadieh2006@gmail.com) (Sole Author)  
**Study Title:** Synergistic Multi-Paradigm Segmentation and Invariant Anatomical Refinement Surpasses the BraTS 2021 Benchmark  
**Date:** September 2026  

---

## What is in this Submission Bundle?

This package contains everything required to submit your brain tumor segmentation research across all scientific and competition channels:

```
tarek_submission_bundle/
├── README_SUBMISSION_GUIDE.md        <-- This guide
├── 1_latex_paper/                     <-- Option 1: Journal / Overleaf submission package
│   ├── main.tex
│   ├── references.bib
│   └── figures/
├── 2_arxiv_package/                   <-- Option 2: arXiv preprint submission package
│   ├── main.tex
│   ├── references.bib
│   ├── arxiv_submission_guide.md
│   └── figures/
├── 3_docker_submission/               <-- Option 3: Synapse challenge evaluation container
│   ├── Dockerfile
│   ├── run_inference.py
│   └── synapse_submission_guide.md
├── 4_research_report/                 <-- Full 24-section research report
│   └── research_report.md
├── 5_code_and_configs/                <-- Complete reproducible modular codebase
│   ├── src/                           <-- Core models, losses, evaluation, ensemble
│   ├── configs/                       <-- Model & training configurations
│   ├── scripts/                       <-- Training and evaluation scripts
│   ├── environment.yml                <-- Conda environment definition
│   └── requirements.txt               <-- Pip dependencies
└── 6_benchmark_results/               <-- Verified quantitative results & figures
    ├── model_comparison.md & .csv     <-- 6-architecture comparison table
    ├── winner_head_to_head.md & .csv  <-- 14-metric proof outperforming 2021 winner
    └── figures/                       <-- All 10 publication-grade figures
```

---

## How to Submit: 3 Easy Paths

### Option 1: Upload to Overleaf or Submit to MICCAI / Springer LNCS
1. Go to [Overleaf.com](https://www.overleaf.com) and click **New Project -> Upload Project**.
2. Select `tarek_brats2021_paper.zip` (located inside `1_latex_paper/` or the root zip).
3. Overleaf will automatically compile the paper with your name, tables, citations, and figures.

### Option 2: Submit Preprint to arXiv (`eess.IV` / `cs.CV`)
1. Go to [arxiv.org/submit](https://arxiv.org/submit).
2. Select Primary Subject: **Electrical Engineering and Systems Science - Image and Video Processing (`eess.IV`)**.
3. Cross list with: **Computer Science - Computer Vision (`cs.CV`)**.
4. Upload all files from `2_arxiv_package/`.
5. Enter title, abstract, and list **Tarek Al Ahmadieh** as Lead Author.

### Option 3: Submit Container to Synapse BraTS Challenge
1. Open terminal in `3_docker_submission/`.
2. Follow `synapse_submission_guide.md` to build and push to Synapse registry:
   `docker build -t tarek-brats2021-winner:latest .`
   `docker tag tarek-brats2021-winner:latest docker.synapse.org/<project_id>/tarek-brats:v1.0`
   `docker push docker.synapse.org/<project_id>/tarek-brats:v1.0`
3. Enter your Docker repository URI into the Synapse challenge submission portal.

---
"""
    with open(bundle_dir / "README_SUBMISSION_GUIDE.md", "w", encoding="utf-8") as f:
        f.write(guide_content)
        
    # 2. Copy latex_paper
    shutil.copytree(base_dir / "submission_package" / "latex_paper", bundle_dir / "1_latex_paper")
    
    # 3. Copy arxiv_package
    shutil.copytree(base_dir / "submission_package" / "arxiv_package", bundle_dir / "2_arxiv_package")
    
    # 4. Copy docker_submission
    shutil.copytree(base_dir / "submission_package" / "docker_submission", bundle_dir / "3_docker_submission")
    
    # 5. Copy research report (both Markdown and PDF)
    rep_dir = bundle_dir / "4_research_report"
    rep_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_dir / "research_report.md", rep_dir / "research_report.md")
    pdf_path = base_dir / "research_report.pdf"
    if pdf_path.exists():
        shutil.copy2(pdf_path, rep_dir / "research_report.pdf")
    
    # 6. Copy code_and_configs
    code_dir = bundle_dir / "5_code_and_configs"
    code_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy src, configs, scripts, ignoring __pycache__
    def ignore_pycache(dir, files):
        return [f for f in files if f == "__pycache__" or f.endswith(".pyc")]
        
    shutil.copytree(base_dir / "src", code_dir / "src", ignore=ignore_pycache)
    shutil.copytree(base_dir / "configs", code_dir / "configs")
    shutil.copytree(base_dir / "scripts", code_dir / "scripts", ignore=ignore_pycache)
    shutil.copy2(base_dir / "requirements.txt", code_dir / "requirements.txt")
    shutil.copy2(base_dir / "environment.yml", code_dir / "environment.yml")
    
    # 7. Copy benchmark_results
    res_dir = bundle_dir / "6_benchmark_results"
    res_dir.mkdir(parents=True, exist_ok=True)
    for f in (base_dir / "results" / "comparison").glob("*"):
        if f.is_file():
            shutil.copy2(f, res_dir / f.name)
    shutil.copytree(base_dir / "results" / "figures", res_dir / "figures")
    
    # 8. Compress the entire bundle into a master ZIP file
    zip_dest = base_dir / "submission_package" / "TAREK_AL_AHMADIEH_BRATS2021_SUBMISSION_BUNDLE.zip"
    print(f"Compressing master bundle to {zip_dest}...")
    
    with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(bundle_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(bundle_dir)
                zipf.write(file_path, arcname)
                
    # Also place a copy on the user's Desktop for instant 1-click access
    desktop_dest = Path("c:/Users/Admin/Desktop/TAREK_AL_AHMADIEH_BRATS2021_SUBMISSION_BUNDLE.zip")
    shutil.copy2(zip_dest, desktop_dest)
    print(f"Created Desktop copy: {desktop_dest}")
    
    zip_size = zip_dest.stat().st_size / (1024 * 1024)
    print(f"Master Submission Bundle successfully created! Size: {zip_size:.2f} MB")

if __name__ == "__main__":
    create_bundle()
