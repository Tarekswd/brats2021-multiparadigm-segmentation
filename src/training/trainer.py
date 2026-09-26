"""
Unified Research-Grade Trainer for BraTS 2021 Brain Tumor Segmentation.
Supports AMP mixed precision, gradient accumulation/clipping, sliding-window inference,
deep supervision losses, polynomial/cosine scheduling, and metrics logging.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.inferers import sliding_window_inference

from src.training.checkpointing import CheckpointManager
from src.training.callbacks import EarlyStopping
from src.training.scheduler import PolynomialLR
from src.utils.logging import get_logger, CSVMetricsLogger
from src.evaluation.dice import compute_dice


class BraTSTrainer:
    """
    Orchestrates training and validation for all BraTS models.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        cfg: Dict[str, Any],
        output_dir: str = "results/experiment",
        device: Optional[torch.device] = None
    ):
        self.cfg = cfg
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn.to(self.device)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger("BraTSTrainer", str(self.output_dir / "train.log"))
        self.csv_logger = CSVMetricsLogger(str(self.output_dir / "training_log.csv"))
        self.ckpt_manager = CheckpointManager(str(self.output_dir / "checkpoints"))

        train_cfg = cfg.get("training", {})
        self.epochs = train_cfg.get("epochs", 100)
        self.grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 1)
        self.max_grad_norm = train_cfg.get("gradient_clip_norm", 12.0)
        self.use_amp = train_cfg.get("amp", False) and torch.cuda.is_available()

        # Optimizer selection: SGD (Nesterov) or AdamW
        opt_name = train_cfg.get("optimizer", "adamw").lower()
        lr = float(train_cfg.get("learning_rate", 1e-4))
        weight_decay = float(train_cfg.get("weight_decay", 1e-5))

        if opt_name == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=lr, momentum=0.99, nesterov=True, weight_decay=weight_decay
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )

        # Scheduler selection
        sched_name = train_cfg.get("scheduler", "polynomial").lower()
        if sched_name == "polynomial":
            self.scheduler = PolynomialLR(self.optimizer, max_epochs=self.epochs, power=0.9)
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs, eta_min=1e-6
            )

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.early_stopping = EarlyStopping(
            patience=train_cfg.get("early_stopping_patience", 30),
            mode="max"
        )

        inf_cfg = cfg.get("inference", {})
        self.roi_size = tuple(inf_cfg.get("roi_size", [128, 128, 128]))
        self.overlap = float(inf_cfg.get("overlap", 0.5))

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            targets = batch["target"].to(self.device)

            with torch.amp.autocast("cuda" if self.device.type == "cuda" else "cpu", enabled=self.use_amp):
                preds = self.model(images)
                loss_dict = self.loss_fn(preds, targets)
                loss = loss_dict["total"] / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps

        return total_loss / max(1, len(self.train_loader))

    def validate(self) -> Dict[str, float]:
        self.model.eval()
        dices_wt = []
        dices_tc = []
        dices_et = []
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                targets = batch["target"].to(self.device)

                # Sliding window inference for full volume or large validation samples
                def model_forward(x):
                    out = self.model(x)
                    return out[0] if isinstance(out, list) else out

                actual_roi = tuple(min(r, s) for r, s in zip(self.roi_size, images.shape[2:]))
                pred_logits = sliding_window_inference(
                    inputs=images,
                    roi_size=actual_roi,
                    sw_batch_size=1,
                    predictor=model_forward,
                    overlap=self.overlap,
                    mode="gaussian"
                )

                # Compute loss
                loss_dict = self.loss_fn(pred_logits, targets)
                total_val_loss += loss_dict["total"].item()

                pred_prob = torch.sigmoid(pred_logits)
                p_bin = (pred_prob >= 0.5).float()

                # Calculate patient-wise Dices: channel 0=WT, 1=TC, 2=ET
                for b in range(p_bin.shape[0]):
                    dices_wt.append(compute_dice(p_bin[b, 0], targets[b, 0]))
                    dices_tc.append(compute_dice(p_bin[b, 1], targets[b, 1]))
                    dices_et.append(compute_dice(p_bin[b, 2], targets[b, 2]))

        val_wt = float(sum(dices_wt) / max(1, len(dices_wt)))
        val_tc = float(sum(dices_tc) / max(1, len(dices_tc)))
        val_et = float(sum(dices_et) / max(1, len(dices_et)))
        val_avg = (val_wt + val_tc + val_et) / 3.0
        val_loss = total_val_loss / max(1, len(self.val_loader))

        return {
            "val_loss": val_loss,
            "dice_wt": val_wt,
            "dice_tc": val_tc,
            "dice_et": val_et,
            "dice_avg": val_avg
        }

    def fit(self) -> Dict[str, Any]:
        self.logger.info(f"Starting training for {self.epochs} epochs on device: {self.device}")
        best_dice = -1.0

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate()
            self.scheduler.step()

            duration = time.time() - t0
            curr_lr = self.optimizer.param_groups[0]["lr"]

            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": curr_lr,
                "epoch_time_sec": duration,
                **val_metrics
            }
            self.csv_logger.log(metrics)

            is_best = val_metrics["dice_avg"] > best_dice
            if is_best:
                best_dice = val_metrics["dice_avg"]

            self.ckpt_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                metrics=metrics,
                is_best=is_best
            )

            self.logger.info(
                f"Epoch {epoch:03d}/{self.epochs:03d} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | "
                f"WT: {val_metrics['dice_wt']:.4f} | TC: {val_metrics['dice_tc']:.4f} | ET: {val_metrics['dice_et']:.4f} | "
                f"Avg Dice: {val_metrics['dice_avg']:.4f} | Time: {duration:.1f}s"
            )

            if self.early_stopping.step(val_metrics["dice_avg"]):
                if self.early_stopping.early_stop:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

        self.logger.info(f"Training completed. Best Validation Dice: {best_dice:.4f}")
        return {"best_dice_avg": best_dice}
