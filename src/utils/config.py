"""
Configuration management utility for BraTS 2021 Research System.
Supports YAML loading, validation, parameter overrides, and serialization.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load configuration from a YAML file with optional nested overrides."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if overrides:
        for k, v in overrides.items():
            keys = k.split(".")
            d = cfg
            for sub_k in keys[:-1]:
                if sub_k not in d or not isinstance(d[sub_k], dict):
                    d[sub_k] = {}
                d = d[sub_k]
            d[keys[-1]] = v

    return cfg


def save_config(cfg: Dict[str, Any], output_path: str) -> None:
    """Save configuration dictionary to a YAML file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
