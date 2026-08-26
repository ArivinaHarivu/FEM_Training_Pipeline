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
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class AdaptiveLossBalancer:
    """Auto-balances multi-term losses live, during training.

    This is the "tune hyperparameters during training itself" mechanism
    requested in place of a separate Optuna sweep (which would delay the
    first training run). It re-weights the loss terms returned by
    ``loss_fn`` — displacement, stress, strain, strain-correction,
    von Mises, whatever keys are present other than ``"total"`` — every
    training step, based on each term's own running-average magnitude:

        weight_k(t)   = 1 / (ema_k(t) + eps), then normalised so the
                        mean weight across terms is 1
        combined loss = sum_k weight_k(t) * raw_loss_k(t)

    Effect: a term that is numerically small (or has shrunk from earlier
    training) gets scaled up relative to the others, and a term that
    dominates gets scaled down — so no single term's raw numeric scale
    silently starves the rest of the gradient signal, without hand-
    tuning the static ``w_u`` / ``w_sigma`` / ... config weights.

    This intentionally does NOT replace ``loss_fn``'s own fixed weights;
    it re-balances on top of them. If a term is small partly *because*
    its static weight is small, this scheme will counteract that and
    impose its own dynamic balance instead — which is the desired
    behaviour here (dynamic > static), but worth knowing if you later
    compare numbers against a run using ``loss_fn``'s "total" directly.

    Parameters
    ----------
    momentum : float
        EMA momentum for the running per-term magnitude. Default 0.9
        (roughly a ~10-step effective window).
    eps : float
        Numerical floor to avoid divide-by-zero when a term is ~0.
    warmup_steps : int
        Number of initial steps to use plain unweighted (weight=1) sums,
        before EMA statistics have had a chance to stabilise.
    """

    def __init__(self, momentum: float = 0.9, eps: float = 1e-8, warmup_steps: int = 20):
        self.momentum = momentum
        self.eps = eps
        self.warmup_steps = warmup_steps
        self._ema: dict[str, float] = {}
        self._step = 0

    def combine(
        self, losses: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return (combined_loss_tensor, effective_weights_used)."""
        terms = {k: v for k, v in losses.items() if k != "total"}
        if not terms:
            return losses["total"], {}

        self._step += 1

        for k, v in terms.items():
            val = float(v.detach().item())
            if k not in self._ema:
                self._ema[k] = max(val, self.eps)
            else:
                self._ema[k] = self.momentum * self._ema[k] + (1 - self.momentum) * val

        if self._step <= self.warmup_steps:
            weights = {k: 1.0 for k in terms}
        else:
            inv = {k: 1.0 / (self._ema[k] + self.eps) for k in terms}
            mean_inv = sum(inv.values()) / len(inv)
            weights = {k: inv[k] / mean_inv for k in terms}

        combined = sum(weights[k] * terms[k] for k in terms)
        return combined, weights


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
        adaptive_loss_weighting: bool = True,
        loss_balance_momentum: float = 0.9,
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

        self.adaptive_loss_weighting = adaptive_loss_weighting
        self.loss_balancer = (
            AdaptiveLossBalancer(momentum=loss_balance_momentum)
            if adaptive_loss_weighting
            else None
        )

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._best_val_loss = float("inf")
        self._log_file = self.log_dir / "training_log.csv"
        self._weights_log_file = self.log_dir / "adaptive_weights_log.jsonl"
        self._last_train_weights: dict[str, float] = {}
        self._init_log()

    def _init_log(self) -> None:
        """Initialize the CSV log file with headers."""
        with open(self._log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "train_L_u", "train_L_sigma",
                "train_L_eps", "train_L_eps_corr", "train_L_vm",
                "val_loss", "val_L_u", "val_L_sigma",
                "val_L_eps", "val_L_eps_corr", "val_L_vm",
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
            train_metrics = self._train_epoch(epoch)

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
            self._log_adaptive_weights(epoch)
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
            if self._last_train_weights:
                weight_str = ", ".join(
                    f"{k}={w:.2f}" for k, w in sorted(self._last_train_weights.items())
                )
                logger.info("  adaptive loss weights (avg this epoch): %s", weight_str)

        return history

    def _train_epoch(self, epoch: int = 1) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        accum: dict[str, float] = {}
        weight_accum: dict[str, float] = {}
        n_batches = 0
        total_batches = len(self.train_loader)

        for i, batch in enumerate(self.train_loader):
            try:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                # Forward
                preds = self.model(batch)

                # Build targets dict from batch
                targets = self._extract_targets(batch)

                # Loss
                losses = self.loss_fn(preds, targets)

                if self.loss_balancer is not None:
                    backward_loss, weights = self.loss_balancer.combine(losses)
                    for k, w in weights.items():
                        weight_accum[k] = weight_accum.get(k, 0.0) + w
                else:
                    backward_loss = losses["total"]

                backward_loss.backward()

                # Gradient clipping
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm,
                    )

                self.optimizer.step()

                # Accumulate raw losses
                for k, v in losses.items():
                    accum[k] = accum.get(k, 0.0) + v.item()
                n_batches += 1

                if (i + 1) % 25 == 0 or (i + 1) == total_batches:
                    logger.info("  [Batch %3d/%d] current loss: %.4f", i + 1, total_batches, backward_loss.item())

                if (i + 1) % 100 == 0:
                    self._save_checkpoint(epoch=epoch, val_loss=0.0, filename="checkpoint_latest.pt")

            except torch.cuda.OutOfMemoryError:
                logger.warning("  ⚠ Batch %d/%d exceeded memory limits; skipping sample and freeing cache...", i + 1, total_batches)
                self.optimizer.zero_grad()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

        n = max(n_batches, 1)
        self._last_train_weights = {k: v / n for k, v in weight_accum.items()}
        return {k: v / n for k, v in accum.items()}

    @torch.no_grad()
    def _val_epoch(self) -> dict[str, float]:
        """Run one validation epoch."""
        self.model.eval()
        accum: dict[str, float] = {}
        n_batches = 0

        for batch in self.val_loader:
            try:
                batch = batch.to(self.device)
                preds = self.model(batch)
                targets = self._extract_targets(batch)
                losses = self.loss_fn(preds, targets)

                for k, v in losses.items():
                    accum[k] = accum.get(k, 0.0) + v.item()
                n_batches += 1
            except torch.cuda.OutOfMemoryError:
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

        return {k: v / max(n_batches, 1) for k, v in accum.items()}

    @staticmethod
    def _extract_targets(batch) -> dict[str, torch.Tensor]:
        """Extract target tensors from a PyG batch."""
        targets: dict[str, torch.Tensor] = {}
        if hasattr(batch, "y_displacement"):
            targets["u"] = batch.y_displacement
        if hasattr(batch, "y_stress"):
            targets["sigma"] = batch.y_stress
        if hasattr(batch, "y_strain"):
            targets["eps"] = batch.y_strain
        if hasattr(batch, "y_von_mises"):
            targets["vm"] = batch.y_von_mises
        return targets

    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False,
        filename: str | None = None,
    ) -> None:
        """Save model checkpoint."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss_balancer_state_dict": self.loss_balancer.state_dict() if self.loss_balancer else None,
            "val_loss": val_loss,
        }
        if filename:
            path = self.checkpoint_dir / filename
            torch.save(state, path)
            logger.info("  💾 Auto-saved checkpoint → %s", path)
            return

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
                train_metrics.get("L_eps_corr", 0),
                train_metrics.get("L_vm", 0),
                val_metrics.get("total", 0),
                val_metrics.get("L_u", 0),
                val_metrics.get("L_sigma", 0),
                val_metrics.get("L_eps", 0),
                val_metrics.get("L_eps_corr", 0),
                val_metrics.get("L_vm", 0),
                lr,
                epoch_time,
            ])

    def _log_adaptive_weights(self, epoch: int) -> None:
        """Append one JSON line with this epoch's effective loss weights."""
        if not self._last_train_weights:
            return
        record = {"epoch": epoch, **self._last_train_weights}
        with open(self._weights_log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def load_checkpoint(self, path: str | Path) -> int:
        """Load a checkpoint and return the epoch number."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self._best_val_loss = ckpt.get("val_loss", float("inf"))
        epoch = ckpt.get("epoch", 0)
        logger.info("Loaded checkpoint from epoch %d (val_loss=%.6f)", epoch, self._best_val_loss)
        return epoch
