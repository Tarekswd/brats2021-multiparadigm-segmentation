"""
Hardware detection and resource management for BraTS 2021 segmentation.
Profiles GPU/CPU memory and automatically configures safe batch sizes and patch sizes.
"""

import sys
import shutil
import psutil
import torch
from typing import Dict, Any, Tuple


def get_hardware_info() -> Dict[str, Any]:
    """Inspect and return complete system and compute device specifications."""
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(".")

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU / Integrated Graphics"
    device_count = torch.cuda.device_count() if cuda_available else 0

    vram_gb = 0.0
    if cuda_available:
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)

    return {
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_available": cuda_available,
        "device_count": device_count,
        "device_name": device_name,
        "vram_gb": vram_gb,
        "ram_total_gb": round(vm.total / (1024 ** 3), 2),
        "ram_available_gb": round(vm.available / (1024 ** 3), 2),
        "disk_total_gb": round(disk.total / (1024 ** 3), 2),
        "disk_free_gb": round(disk.free / (1024 ** 3), 2),
    }


def recommend_training_parameters(vram_gb: float = 0.0) -> Dict[str, Any]:
    """
    Recommend memory-safe hyperparameters (patch size, batch size, gradient accumulation)
    based on available VRAM or RAM capacity to avoid Out-Of-Memory (OOM) failures.
    """
    if vram_gb >= 24.0:
        # e.g., RTX 3090, A5000, A6000, H100
        return {
            "patch_size": [128, 128, 128],
            "batch_size": 2,
            "grad_accum_steps": 1,
            "amp": True,
            "tier": "high_end_gpu"
        }
    elif vram_gb >= 12.0:
        # e.g., RTX 3060, RTX 4070, V100
        return {
            "patch_size": [128, 128, 128],
            "batch_size": 1,
            "grad_accum_steps": 2,
            "amp": True,
            "tier": "mid_range_gpu"
        }
    elif vram_gb >= 6.0:
        # e.g., RTX 2060, GTX 1660, 8GB GPUs
        return {
            "patch_size": [96, 96, 96],
            "batch_size": 1,
            "grad_accum_steps": 4,
            "amp": True,
            "tier": "budget_gpu"
        }
    else:
        # CPU / Low-VRAM fallback
        return {
            "patch_size": [64, 64, 64],
            "batch_size": 1,
            "grad_accum_steps": 4,
            "amp": False,
            "tier": "cpu_or_low_memory"
        }
