"""
Model checkpoint management for BraTS 2021 experiments.
Saves and restores model weights, optimizer states, epoch counters, and validation records.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class CheckpointManager:
    """Manages saving and loading of best_model.pt and last_model.pt checkpoints."""

    def __init__(self, checkpoint_dir: str):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = -float("inf")

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        metrics: Dict[str, Any],
        is_best: bool = False
    ) -> None:
        """Save training state to last_model.pt and optionally best_model.pt."""
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
        }

        # Save latest checkpoint
        torch.save(state, self.dir / "last_model.pt")

        # Save best checkpoint
        if is_best:
            torch.save(state, self.dir / "best_model.pt")

    def load(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: torch.device = torch.device("cpu")
    ) -> Dict[str, Any]:
        """Restore state from checkpoint."""
        p = Path(checkpoint_path)
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {p}")

        state = torch.load(p, map_location=device)
        model.load_state_dict(state["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in state and state["optimizer_state_dict"]:
            optimizer.load_state_dict(state["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in state and state["scheduler_state_dict"]:
            scheduler.load_state_dict(state["scheduler_state_dict"])

        return state
