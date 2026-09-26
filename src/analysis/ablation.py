"""
Ablation Study Aggregator and Report Generator for BraTS 2021 (Section 37).
Formats controlled modification tables into Markdown and CSV.
"""

from typing import Dict, List, Any
import pandas as pd


def generate_ablation_table(
    ablation_records: List[Dict[str, Any]],
    output_path: str = None
) -> str:
    """
    Format list of ablation results into a standard Markdown table.
    Each record must have: 'modification', 'wt_dice', 'tc_dice', 'et_dice', 'avg_hd95'.
    """
    df = pd.DataFrame(ablation_records)

    header = "| Modification | WT Dice (%) | TC Dice (%) | ET Dice (%) | Avg HD95 (mm) |\n"
    separator = "|:---|:---:|:---:|:---:|:---:|\n"
    rows = []

    for r in ablation_records:
        mod = r.get("modification", "Unknown")
        wt = f"{r.get('wt_dice', 0.0):.2f}"
        tc = f"{r.get('tc_dice', 0.0):.2f}"
        et = f"{r.get('et_dice', 0.0):.2f}"
        hd = f"{r.get('avg_hd95', 0.0):.2f}"
        rows.append(f"| {mod} | {wt} | {tc} | {et} | {hd} |")

    table_md = header + separator + "\n".join(rows) + "\n"

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(table_md)
        # Also export CSV
        df.to_csv(output_path.replace(".md", ".csv"), index=False)

    return table_md
