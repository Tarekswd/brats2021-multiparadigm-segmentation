"""
Convert research_report.md to a publication-grade PDF using Markdown, MathJax styling,
and Headless Chrome / Edge.
"""
import os
import re
import subprocess
from pathlib import Path
import markdown

def convert_to_pdf():
    base_dir = Path("c:/Users/Admin/Desktop/brats2021/brats2021_research")
    report_md_path = base_dir / "research_report.md"
    temp_html_path = base_dir / "research_report.html"
    output_pdf_path = base_dir / "research_report.pdf"
    desktop_pdf_path = Path("c:/Users/Admin/Desktop/BraTS2021_Research_Report_Tarek_Al_Ahmadieh.pdf")
    
    with open(report_md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
        
    # Replace relative image paths with absolute file:/// URIs for headless browser rendering
    def replace_img_path(match):
        rel_path = match.group(1)
        abs_path = (base_dir / rel_path).resolve().as_uri()
        return f'src="{abs_path}"'
        
    # Convert markdown to HTML with tables, code, and header anchors
    html_body = markdown.markdown(
        md_text,
        extensions=['tables', 'fenced_code', 'toc', 'attr_list']
    )
    
    # Fix image sources in HTML body
    html_body = re.sub(r'src="(results/figures/[^"]+)"', replace_img_path, html_body)
    html_body = re.sub(r'src="(\./results/figures/[^"]+)"', replace_img_path, html_body)
    
    # Format math equations cleanly
    html_body = html_body.replace(r'$ET \subseteq TC \subseteq WT$', '<strong><em>ET &sube; TC &sube; WT</em></strong>')
    html_body = html_body.replace(r'$$ET \subseteq TC \subseteq WT$$', '<div class="equation"><em>ET &sube; TC &sube; WT</em></div>')
    
    styled_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>BraTS 2021 Brain Tumor Segmentation Research Report - Tarek Al Ahmadieh</title>
    <style>
        @page {{
            size: A4;
            margin: 20mm 15mm 20mm 15mm;
            @bottom-right {{
                content: counter(page);
            }}
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            color: #1e293b;
            line-height: 1.6;
            font-size: 11pt;
            background-color: #ffffff;
            margin: 0;
            padding: 0;
        }}
        h1 {{
            color: #0f172a;
            font-size: 22pt;
            font-weight: 700;
            border-bottom: 2px solid #2563eb;
            padding-bottom: 8px;
            margin-top: 0;
            margin-bottom: 12px;
            line-height: 1.25;
        }}
        .author-card {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #2563eb;
            padding: 12px 16px;
            border-radius: 4px;
            margin-bottom: 24px;
            font-size: 10.5pt;
        }}
        h2 {{
            color: #1e3a8a;
            font-size: 14pt;
            font-weight: 600;
            border-bottom: 1px solid #cbd5e1;
            padding-bottom: 4px;
            margin-top: 26px;
            margin-bottom: 10px;
            page-break-after: avoid;
        }}
        h3 {{
            color: #1e40af;
            font-size: 12pt;
            font-weight: 600;
            margin-top: 18px;
            margin-bottom: 6px;
            page-break-after: avoid;
        }}
        p {{
            margin-top: 0;
            margin-bottom: 10px;
            text-align: justify;
        }}
        strong {{
            color: #0f172a;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            font-size: 9.5pt;
            page-break-inside: avoid;
        }}
        th, td {{
            padding: 7px 10px;
            border: 1px solid #cbd5e1;
            text-align: left;
        }}
        th {{
            background-color: #f1f5f9;
            color: #0f172a;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #f8fafc;
        }}
        img {{
            max-width: 95%;
            height: auto;
            display: block;
            margin: 16px auto 6px auto;
            border: 1px solid #e2e8f0;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            page-break-inside: avoid;
        }}
        em {{
            display: block;
            text-align: center;
            font-size: 9pt;
            color: #64748b;
            margin-bottom: 16px;
        }}
        hr {{
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 24px 0;
        }}
        .equation {{
            text-align: center;
            font-size: 12pt;
            margin: 12px 0;
            font-weight: 600;
            color: #1e3a8a;
        }}
        code {{
            background-color: #f1f5f9;
            color: #0f172a;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: Consolas, monospace;
            font-size: 9pt;
        }}
        pre {{
            background-color: #0f172a;
            color: #f8fafc;
            padding: 12px;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 8.5pt;
            line-height: 1.4;
            page-break-inside: avoid;
        }}
        pre code {{
            background-color: transparent;
            color: inherit;
            padding: 0;
        }}
        ul, ol {{
            margin-top: 0;
            margin-bottom: 12px;
            padding-left: 24px;
        }}
        li {{
            margin-bottom: 4px;
        }}
    </style>
</head>
<body>
{html_body}
</body>
</html>
"""
    with open(temp_html_path, "w", encoding="utf-8") as f:
        f.write(styled_html)
    print(f"Generated intermediate HTML at {temp_html_path}")
    
    # Locate Chrome or Edge
    chrome_paths = [
        Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
        Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
        Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe")
    ]
    
    browser_exe = None
    for p in chrome_paths:
        if p.exists():
            browser_exe = p
            break
            
    if not browser_exe:
        raise RuntimeError("Neither Google Chrome nor Microsoft Edge could be found.")
        
    print(f"Using browser for PDF compilation: {browser_exe}")
    
    cmd = [
        str(browser_exe),
        "--headless",
        "--disable-gpu",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={output_pdf_path}",
        "--no-pdf-header-footer",
        temp_html_path.as_uri()
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Browser error:", result.stderr)
    else:
        print(f"Successfully generated PDF: {output_pdf_path}")
        
    # Copy to Desktop and to submission packages
    if output_pdf_path.exists():
        import shutil
        import time
        time.sleep(1.5)
        
        for dst in [desktop_pdf_path, base_dir / "submission_package" / "research_report.pdf", base_dir / "tarek_submission_bundle" / "4_research_report" / "research_report.pdf"]:
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(output_pdf_path, dst)
            except Exception as e:
                print(f"Notice when copying to {dst}: {e}")
                time.sleep(1.0)
                try:
                    shutil.copy2(output_pdf_path, dst)
                except Exception:
                    pass
        
        pdf_size_mb = output_pdf_path.stat().st_size / (1024 * 1024)
        print(f"Desktop PDF created: {desktop_pdf_path} ({pdf_size_mb:.2f} MB)")
        
        # Also update the Master Submission ZIP with the new PDF!
        from create_submission_bundle import create_bundle
        create_bundle()

if __name__ == "__main__":
    convert_to_pdf()
