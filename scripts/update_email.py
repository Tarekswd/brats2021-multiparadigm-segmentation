"""
Update contact email from tarek.ahmadieh2006@gmail.com to tarek.ahmadieh2006@gmail.com
across all codebase, submission packages, guides, LaTeX files, and reports.
"""
import os
from pathlib import Path

OLD_EMAIL = "tarek.ahmadieh2006@gmail.com"
NEW_EMAIL = "tarek.ahmadieh2006@gmail.com"

dirs_to_update = [
    Path("c:/Users/Admin/Desktop/brats2021/brats2021_research"),
    Path("C:/Users/Admin/Desktop/TAREK_BRATS2021_SUBMISSION")
]

extensions = [".tex", ".md", ".py", ".html", ".yaml", ".yml", ".txt", ".bib", "Dockerfile"]

updated_files = []

for base_dir in dirs_to_update:
    if not base_dir.exists():
        continue
    for root, dirs, files in os.walk(base_dir):
        if "__pycache__" in root or ".git" in root:
            continue
        for f in files:
            file_path = Path(root) / f
            if any(f.endswith(ext) or f == ext for ext in extensions):
                try:
                    with open(file_path, "r", encoding="utf-8") as file:
                        content = file.read()
                    if OLD_EMAIL in content:
                        new_content = content.replace(OLD_EMAIL, NEW_EMAIL)
                        with open(file_path, "w", encoding="utf-8") as file:
                            file.write(new_content)
                        updated_files.append(str(file_path))
                        print(f"Updated: {file_path}")
                except Exception as e:
                    pass

print(f"\nSuccessfully replaced email in {len(updated_files)} files.")
