"""Trainer — training loop for MeshGraphNet on FEM data.

Handles:
- Training and validation epoch loops
- Multi-term loss (displacement, stress, strain correction, von Mises)
- Gradient clipping
- Learning rate scheduling (ReduceLROnPlateau)
- Checkpointing (best val loss)
- Logging (per-epoch metrics to console + CSV)

Designed to run on Colab with a single GPU or CPU fallback.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class Trainer:
    """MeshGraphNet training loop.

    Parameters
    ----------
    model : nn.Module
        The MeshGraphNet model.
    loss_fn : nn.Module
        MeshGraphNetLoss instance (with field_stds registered).
    train_loader : DataLoader
        PyG DataLoader for training.
    val_loader : DataLoader, optional
        PyG DataLoader for validation.
    lr : float
        Learning rate. Default 1e-3.
    grad_clip_norm : float
        Max gradient norm for clipping. Default 1.0.
    device : str
        ``"cuda"`` or ``"cpu"``. Auto-detected if not specified.
    checkpoint_dir : str or Path
        Directory for saving checkpoints.
    log_dir : str or Path
        Directory for CSV training logs.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        train_loader,
        val_loader=None,
        lr: float = 1e-3,
        grad_clip_norm: float = 1.0,
        device: str | None = None,
        checkpoint_dir: str | Path = "checkpoints",
        log_dir: str | Path = "logs",
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10,
        )
        self.grad_clip_norm = grad_clip_norm

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._best_val_loss = float("inf")
        self._log_file = self.log_dir / "training_log.csv"
        self._init_log()

    def _init_log(self) -> None:
        """Initialize the CSV log file with headers."""
        with open(self._log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "train_L_u", "train_L_sigma",
                "train_L_eps", "train_L_vm",
                "val_loss", "val_L_u", "val_L_sigma",
                "val_L_eps", "val_L_vm",
                "lr", "epoch_time_s",
            ])

    def train(self, num_epochs: int = 100) -> dict[str, list[float]]:
        """Run the full training loop.

        Parameters
        ----------
        num_epochs : int
            Number of training epochs.

        Returns
        -------
        dict[str, list[float]]
            History of per-epoch metrics.
        """
        history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [], "lr": [],
        }

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.perf_counter()

            # --- Train ---
            train_metrics = self._train_epoch()

            # --- Validate ---
            val_metrics = {}
            if self.val_loader is not None and len(self.val_loader) > 0:
                val_metrics = self._val_epoch()

            epoch_time = time.perf_counter() - epoch_start
            current_lr = self.optimizer.param_groups[0]["lr"]

            # --- Scheduler step ---
            val_loss = val_metrics.get("total", train_metrics["total"])
            self.scheduler.step(val_loss)

            # --- Checkpoint ---
            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, is_best=True)

            # --- Log ---
            self._log_epoch(epoch, train_metrics, val_metrics, current_lr, epoch_time)
            history["train_loss"].append(train_metrics["total"])
            history["val_loss"].append(val_loss)
            history["lr"].append(current_lr)

            # --- Console ---
            val_str = f"val={val_loss:.6f}" if val_metrics else "val=N/A"
            logger.info(
                "Epoch %3d/%d | train=%.6f %s | lr=%.2e | %.1fs",
                epoch, num_epochs, train_metrics["total"],
                val_str, current_lr, epoch_time,
            )

        return history

    def _train_epoch(self) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        accum: dict[str, float] = {}
        n_batches = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            # Forward
            preds = self.model(batch)

            # Build targets dict from batch
            targets = self._extract_targets(batch)

            # Loss
            losses = self.loss_fn(preds, targets)
            losses["total"].backward()

            # Gradient clipping
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm,
                )

            self.optimizer.step()

            # Accumulate
            for k, v in losses.items():
                accum[k] = accum.get(k, 0.0) + v.item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in accum.items()}

    @torch.no_grad()
    def _val_epoch(self) -> dict[str, float]:
        """Run one validation epoch."""
        self.model.eval()
        accum: dict[str, float] = {}
        n_batches = 0

        for batch in self.val_loader:
            batch = batch.to(self.device)
            preds = self.model(batch)
            targets = self._extract_targets(batch)
            losses = self.loss_fn(preds, targets)

            for k, v in losses.items():
                accum[k] = accum.get(k, 0.0) + v.item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in accum.items()}

    @staticmethod
    def _extract_targets(batch) -> dict[str, torch.Tensor]:
        """Extract target tensors from a PyG batch.

        Maps from our naming convention (``y_displacement``, ``y_stress``,
        ``y_von_mises``) to the loss function's expected keys
        (``u``, ``sigma``, ``vm``).
        """
        targets: dict[str, torch.Tensor] = {}
        if hasattr(batch, "y_displacement"):
            targets["u"] = batch.y_displacement
        if hasattr(batch, "y_stress"):
            targets["sigma"] = batch.y_stress
        if hasattr(batch, "y_von_mises"):
            targets["vm"] = batch.y_von_mises
        return targets

    def _save_checkpoint(
        self, epoch: int, val_loss: float, is_best: bool = False,
    ) -> None:
        """Save model checkpoint."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
        }
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(state, path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(state, best_path)
            logger.info(
                "  ★ New best model saved (val_loss=%.6f) → %s",
                val_loss, best_path,
            )

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        lr: float,
        epoch_time: float,
    ) -> None:
        """Append one row to the CSV log."""
        with open(self._log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics.get("total", 0),
                train_metrics.get("L_u", 0),
                train_metrics.get("L_sigma", 0),
                train_metrics.get("L_eps", 0),
                train_metrics.get("L_vm", 0),
                val_metrics.get("total", 0),
                val_metrics.get("L_u", 0),
                val_metrics.get("L_sigma", 0),
                val_metrics.get("L_eps", 0),
                val_metrics.get("L_vm", 0),
                lr,
                epoch_time,
            ])

    def load_checkpoint(self, path: str | Path) -> int:
        """Load a checkpoint and return the epoch number.

        Parameters
        ----------
        path : str or Path
            Path to ``.pt`` checkpoint file.

        Returns
        -------
        int
            The epoch number from the checkpoint.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self._best_val_loss = ckpt.get("val_loss", float("inf"))
        epoch = ckpt.get("epoch", 0)
        logger.info("Loaded checkpoint from epoch %d (val_loss=%.6f)", epoch, self._best_val_loss)
        return epoch
