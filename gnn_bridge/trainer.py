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
import random
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
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

    def state_dict(self) -> dict[str, Any]:
        """Return the state of the loss balancer for checkpointing."""
        return {
            "momentum": self.momentum,
            "eps": self.eps,
            "warmup_steps": self.warmup_steps,
            "ema": self._ema.copy(),
            "step": self._step,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the state of the loss balancer from a checkpoint."""
        self.momentum = state_dict.get("momentum", self.momentum)
        self.eps = state_dict.get("eps", self.eps)
        self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps)
        self._ema = state_dict.get("ema", {}).copy()
        self._step = state_dict.get("step", 0)


class Trainer:
    """MeshGraphNet training loop with intra-epoch batch resumption.

    Parameters
    ----------
    model : nn.Module
        The MeshGraphNet model.
    loss_fn : nn.Module
        MeshGraphNetLoss instance (with field_stds registered).
    train_loader : DataLoader
        PyG DataLoader for training (preferably with ResumableBatchSampler).
    val_loader : DataLoader, optional
        PyG DataLoader for validation.
    lr : float
        Learning rate. Default 1e-3.
    grad_clip_norm : float
        Max gradient norm for clipping. Default 1.0.
    device : str, optional
        ``"cuda"`` or ``"cpu"``. Auto-detected if not specified.
    checkpoint_dir : str or Path
        Directory for saving checkpoints.
    log_dir : str or Path
        Directory for CSV training logs.
    adaptive_loss_weighting : bool
        Whether to use AdaptiveLossBalancer. Default True.
    loss_balance_momentum : float
        EMA momentum for AdaptiveLossBalancer. Default 0.9.
    checkpoint_interval_batches : int
        Save a batch-level checkpoint every N batches. Default 50.
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
        checkpoint_interval_batches: int = 50,
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

        self.checkpoint_interval_batches = max(1, checkpoint_interval_batches)
        self._global_step: int = 0
        self._best_val_loss = float("inf")
        self._log_file = self.log_dir / "training_log.csv"
        self._weights_log_file = self.log_dir / "adaptive_weights_log.jsonl"
        self._last_train_weights: dict[str, float] = {}
        self._current_history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }
        self._init_log()

    def _init_log(self, force: bool = False) -> None:
        """Initialize the CSV log file with headers if it does not exist."""
        if not force and self._log_file.exists() and self._log_file.stat().st_size > 0:
            return
        with open(self._log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "train_L_u", "train_L_sigma",
                "train_L_eps", "train_L_eps_corr", "train_L_vm",
                "val_loss", "val_L_u", "val_L_sigma",
                "val_L_eps", "val_L_eps_corr", "val_L_vm",
                "lr", "epoch_time_s",
            ])

    def train(
        self,
        num_epochs: int = 100,
        start_epoch: int = 1,
        start_batch: int = 0,
        resume_state: dict[str, Any] | None = None,
    ) -> dict[str, list[float]]:
        """Run the full training loop with support for intra-epoch resumption.

        Parameters
        ----------
        num_epochs : int
            Target total number of training epochs.
        start_epoch : int
            Epoch to start/resume training at (1-indexed). Default 1.
        start_batch : int
            Intra-epoch batch index to resume at (0-indexed). Default 0.
        resume_state : dict[str, Any], optional
            Metadata dictionary returned by `load_checkpoint()`. If provided,
            `start_epoch`, `start_batch`, accumulators, and history are auto-populated.

        Returns
        -------
        dict[str, list[float]]
            History of per-epoch metrics.
        """
        initial_accum: dict[str, float] = {}
        initial_weight_accum: dict[str, float] = {}
        initial_n_batches: int = 0

        if resume_state is not None:
            start_epoch = resume_state.get("next_epoch", start_epoch)
            start_batch = resume_state.get("start_batch", start_batch)
            initial_accum = resume_state.get("epoch_accum", {})
            initial_weight_accum = resume_state.get("weight_accum", {})
            initial_n_batches = resume_state.get("n_batches", 0)
            if "history" in resume_state and resume_state["history"]:
                self._current_history = {
                    k: list(v) for k, v in resume_state["history"].items()
                }

        logger.info(
            "Starting training run: epochs %d to %d (starting at batch %d)...",
            start_epoch, num_epochs, start_batch,
        )

        try:
            for epoch in range(start_epoch, num_epochs + 1):
                epoch_start = time.perf_counter()

                batch_offset = start_batch if epoch == start_epoch else 0
                init_acc = initial_accum if epoch == start_epoch else None
                init_wt = initial_weight_accum if epoch == start_epoch else None
                init_nb = initial_n_batches if epoch == start_epoch else 0

                # --- Train Epoch ---
                train_metrics = self._train_epoch(
                    epoch=epoch,
                    start_batch=batch_offset,
                    initial_accum=init_acc,
                    initial_weight_accum=init_wt,
                    initial_n_batches=init_nb,
                )

                # --- Validate ---
                val_metrics = {}
                if self.val_loader is not None and len(self.val_loader) > 0:
                    val_metrics = self._val_epoch()

                epoch_time = time.perf_counter() - epoch_start
                current_lr = self.optimizer.param_groups[0]["lr"]

                # --- Scheduler step ---
                val_loss = val_metrics.get("total", train_metrics.get("total", 0.0))
                self.scheduler.step(val_loss)

                # --- Update History & Log ---
                self._current_history["train_loss"].append(train_metrics.get("total", 0.0))
                self._current_history["val_loss"].append(val_loss)
                self._current_history["lr"].append(current_lr)
                self._log_epoch(epoch, train_metrics, val_metrics, current_lr, epoch_time)
                self._log_adaptive_weights(epoch)

                # --- Checkpoint best model ---
                if val_loss < self._best_val_loss:
                    self._best_val_loss = val_loss
                    self._save_checkpoint(
                        epoch=epoch,
                        val_loss=val_loss,
                        is_best=True,
                        batch_idx=-1,
                        history=self._current_history,
                    )

                # --- Checkpoint epoch completion ---
                self._save_checkpoint(
                    epoch=epoch,
                    val_loss=val_loss,
                    is_best=False,
                    batch_idx=-1,
                    history=self._current_history,
                )
                # Also refresh checkpoint_latest.pt at end of epoch
                self._save_checkpoint(
                    epoch=epoch,
                    val_loss=val_loss,
                    is_best=False,
                    filename="checkpoint_latest.pt",
                    batch_idx=-1,
                    history=self._current_history,
                )

                # --- Console ---
                val_str = f"val={val_loss:.6f}" if val_metrics else "val=N/A"
                logger.info(
                    "Epoch %3d/%d | train=%.6f %s | lr=%.2e | %.1fs",
                    epoch, num_epochs, train_metrics.get("total", 0.0),
                    val_str, current_lr, epoch_time,
                )
                if self._last_train_weights:
                    weight_str = ", ".join(
                        f"{k}={w:.2f}" for k, w in sorted(self._last_train_weights.items())
                    )
                    logger.info("  adaptive loss weights (avg this epoch): %s", weight_str)

        except KeyboardInterrupt:
            logger.warning(
                "Training interrupted by user. Latest progress has been auto-saved to %s/checkpoint_latest.pt",
                self.checkpoint_dir,
            )

        return self._current_history

    def _train_epoch(
        self,
        epoch: int = 1,
        start_batch: int = 0,
        initial_accum: dict[str, float] | None = None,
        initial_weight_accum: dict[str, float] | None = None,
        initial_n_batches: int = 0,
    ) -> dict[str, float]:
        """Run one training epoch with optional start batch offset."""
        self.model.train()
        accum: dict[str, float] = initial_accum.copy() if initial_accum else {}
        weight_accum: dict[str, float] = initial_weight_accum.copy() if initial_weight_accum else {}
        n_batches = initial_n_batches

        sampler = getattr(self.train_loader, "batch_sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
            sampler.set_start_batch(start_batch)

        # Sync epoch to dataset for deterministic worker-independent augmentation
        dataset = getattr(self.train_loader, "dataset", None)
        if dataset is not None:
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)
            elif hasattr(dataset, "dataset") and hasattr(dataset.dataset, "set_epoch"):
                dataset.dataset.set_epoch(epoch)

        if sampler is not None and hasattr(sampler, "total_batches"):
            total_batches = sampler.total_batches()
        else:
            total_batches = len(self.train_loader) if self.train_loader is not None else 0

        if start_batch > 0:
            logger.info(
                "  ▶ Resuming epoch %d at batch %d/%d (skipping prior %d batches)...",
                epoch, start_batch + 1, total_batches, start_batch,
            )

        curr_batch_idx = start_batch
        for i, batch in enumerate(self.train_loader):
            # If standard loader without ResumableBatchSampler was passed, fast-forward
            if sampler is None or not hasattr(sampler, "set_start_batch"):
                if i < start_batch:
                    continue
                curr_batch_idx = i
            else:
                curr_batch_idx = start_batch + i

            try:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                # Forward
                preds = self.model(batch)

                # Targets
                targets = self._extract_targets(batch)

                # Loss
                losses = self.loss_fn(preds, targets)

                if self.loss_balancer is not None:
                    backward_loss, weights = self.loss_balancer.combine(losses)
                    for k, w in weights.items():
                        weight_accum[k] = weight_accum.get(k, 0.0) + w
                else:
                    backward_loss = losses["total"]

                if not torch.isfinite(backward_loss):
                    logger.warning(
                        "  ⚠ Batch %d/%d produced non-finite loss (NaN/Inf); skipping step to protect weights...",
                        curr_batch_idx + 1, total_batches,
                    )
                    self.optimizer.zero_grad()
                    continue

                backward_loss.backward()

                # Gradient clipping
                if self.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip_norm,
                    )

                self.optimizer.step()
                self._global_step += 1

                # Accumulate raw losses
                for k, v in losses.items():
                    accum[k] = accum.get(k, 0.0) + v.item()
                n_batches += 1

                if (curr_batch_idx + 1) == 1 or (curr_batch_idx + 1) % 10 == 0 or (curr_batch_idx + 1) == total_batches:
                    logger.info(
                        "  [Batch %3d/%d] current loss: %.4f",
                        curr_batch_idx + 1, total_batches, backward_loss.item(),
                    )

                # Periodic batch-level checkpointing
                if (curr_batch_idx + 1) % self.checkpoint_interval_batches == 0:
                    self._save_checkpoint(
                        epoch=epoch,
                        batch_idx=curr_batch_idx,
                        val_loss=0.0,
                        filename="checkpoint_latest.pt",
                        epoch_accum=accum,
                        weight_accum=weight_accum,
                        n_batches=n_batches,
                        history=self._current_history,
                    )

            except torch.cuda.OutOfMemoryError:
                logger.warning(
                    "  ⚠ Batch %d/%d exceeded memory limits; skipping sample and freeing cache...",
                    curr_batch_idx + 1, total_batches,
                )
                self.optimizer.zero_grad()
                
                # Explicitly delete local variables to break references 
                # so empty_cache() can actually free the fragmented memory
                if 'preds' in locals(): del preds
                if 'targets' in locals(): del targets
                if 'losses' in locals(): del losses
                if 'backward_loss' in locals(): del backward_loss
                if 'batch' in locals(): del batch
                
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            except KeyboardInterrupt:
                logger.warning(
                    "  ⚠ KeyboardInterrupt detected at epoch %d, batch %d/%d! Auto-saving checkpoint...",
                    epoch, curr_batch_idx + 1, total_batches,
                )
                self._save_checkpoint(
                    epoch=epoch,
                    batch_idx=curr_batch_idx,
                    val_loss=0.0,
                    filename="checkpoint_latest.pt",
                    epoch_accum=accum,
                    weight_accum=weight_accum,
                    n_batches=n_batches,
                    history=self._current_history,
                )
                raise

        # Reset sampler start_batch for clean subsequent epochs
        if sampler is not None and hasattr(sampler, "set_start_batch"):
            sampler.set_start_batch(0)

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
        val_loss: float = 0.0,
        batch_idx: int = -1,
        is_best: bool = False,
        filename: str | None = None,
        epoch_accum: dict[str, float] | None = None,
        weight_accum: dict[str, float] | None = None,
        n_batches: int = 0,
        history: dict[str, list[float]] | None = None,
    ) -> Path | None:
        """Save model checkpoint with atomic write protection and full training state."""
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            state = {
                "epoch": epoch,
                "batch_idx": batch_idx,
                "global_step": self._global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "loss_balancer_state_dict": (
                    self.loss_balancer.state_dict() if self.loss_balancer else None
                ),
                "val_loss": val_loss,
                "best_val_loss": self._best_val_loss,
                "epoch_accum": epoch_accum if epoch_accum is not None else {},
                "weight_accum": weight_accum if weight_accum is not None else {},
                "n_batches": n_batches,
                "history": history if history is not None else self._current_history,
                "rng_state": {
                    "torch": torch.get_rng_state(),
                    "cuda": (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else None
                    ),
                    "numpy": np.random.get_state(),
                    "python": random.getstate(),
                },
            }

            if filename:
                path = self.checkpoint_dir / filename
                temp_path = path.with_suffix(".tmp")
                torch.save(state, temp_path)
                temp_path.replace(path)
                logger.info("  💾 Auto-saved checkpoint → %s (epoch %d, batch %d)", path, epoch, batch_idx + 1 if batch_idx >= 0 else 0)
                return path

            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            temp_path = path.with_suffix(".tmp")
            torch.save(state, temp_path)
            temp_path.replace(path)

            if is_best:
                best_path = self.checkpoint_dir / "best_model.pt"
                best_temp = best_path.with_suffix(".tmp")
                torch.save(state, best_temp)
                best_temp.replace(best_path)
                logger.info(
                    "  ★ New best model saved (val_loss=%.6f) → %s",
                    val_loss, best_path,
                )
            return path
        except Exception as e:
            logger.error("  ⚠ Checkpoint save failed (%s); continuing training without crashing...", e)
            return None

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

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Load a checkpoint and return a metadata dictionary with resume info.

        Parameters
        ----------
        path : str or Path
            Path to the checkpoint file (.pt).

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - "epoch": int (checkpointed epoch)
            - "batch_idx": int (-1 if saved at end of epoch, or >=0 if mid-epoch)
            - "next_epoch": int (epoch to start/resume training)
            - "start_batch": int (batch index to start/resume training)
            - "global_step": int
            - "val_loss": float
            - "best_val_loss": float
            - "epoch_accum": dict
            - "weight_accum": dict
            - "n_batches": int
            - "history": dict
        """
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"]:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        if (
            "loss_balancer_state_dict" in ckpt
            and self.loss_balancer
            and ckpt["loss_balancer_state_dict"]
        ):
            self.loss_balancer.load_state_dict(ckpt["loss_balancer_state_dict"])

        self._best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
        self._global_step = ckpt.get("global_step", 0)

        if "history" in ckpt and ckpt["history"]:
            self._current_history = {
                k: list(v) for k, v in ckpt["history"].items()
            }

        # Restore RNG state
        rng = ckpt.get("rng_state")
        if rng:
            if "torch" in rng and rng["torch"] is not None:
                t_state = rng["torch"]
                torch.set_rng_state(t_state.cpu() if isinstance(t_state, torch.Tensor) else t_state)
            if "cuda" in rng and rng["cuda"] is not None and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state_all(rng["cuda"])
                except Exception as e:
                    logger.debug("Could not restore CUDA RNG state: %s", e)
            if "numpy" in rng and rng["numpy"] is not None:
                np.random.set_state(rng["numpy"])
            if "python" in rng and rng["python"] is not None:
                random.setstate(rng["python"])

        epoch = ckpt.get("epoch", 1)
        batch_idx = ckpt.get("batch_idx", -1)
        epoch_accum = ckpt.get("epoch_accum", {})
        weight_accum = ckpt.get("weight_accum", {})
        n_batches = ckpt.get("n_batches", 0)

        sampler = getattr(self.train_loader, "batch_sampler", None)
        if sampler is not None and hasattr(sampler, "total_batches"):
            total_batches = sampler.total_batches()
        else:
            total_batches = len(self.train_loader) if self.train_loader is not None else 0

        if batch_idx < 0 or (total_batches > 0 and batch_idx >= total_batches - 1):
            next_epoch = epoch + 1
            start_batch = 0
            epoch_accum = {}
            weight_accum = {}
            n_batches = 0
        else:
            next_epoch = epoch
            start_batch = batch_idx + 1

        logger.info(
            "Loaded checkpoint from %s: epoch %d (batch %d/%d, global_step=%d, best_val_loss=%.6f) → will resume at epoch %d, batch %d",
            path, epoch, batch_idx + 1 if batch_idx >= 0 else total_batches,
            total_batches, self._global_step, self._best_val_loss, next_epoch, start_batch,
        )

        return {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "next_epoch": next_epoch,
            "start_batch": start_batch,
            "global_step": self._global_step,
            "val_loss": ckpt.get("val_loss", float("inf")),
            "best_val_loss": self._best_val_loss,
            "epoch_accum": epoch_accum,
            "weight_accum": weight_accum,
            "n_batches": n_batches,
            "history": self._current_history,
        }

    def resume_latest(self) -> dict[str, Any] | None:
        """Auto-detect and load checkpoint_latest.pt from checkpoint_dir if it exists."""
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        if latest_path.exists():
            logger.info("Found existing checkpoint: %s. Loading state for auto-resumption...", latest_path)
            return self.load_checkpoint(latest_path)
        return None
