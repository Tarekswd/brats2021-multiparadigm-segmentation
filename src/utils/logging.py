"""
Structured logging utility for BraTS 2021 training, validation, and evaluation.
Outputs formatted console logs and appends structured rows to metrics CSV files.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd


def get_logger(name: str = "BraTS2021", log_file: str = None) -> logging.Logger:
    """Create and configure a multi-handler logger (stdout + optional log file)."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(p, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class CSVMetricsLogger:
    """Appends validation and training metrics to a CSV file across epochs."""

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.history: List[Dict[str, Any]] = []

    def log(self, metrics: Dict[str, Any]) -> None:
        self.history.append(metrics)
        df = pd.DataFrame(self.history)
        df.to_csv(self.csv_path, index=False)
