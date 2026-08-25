"""Evaluation metrics for MeshGraphNet predictions.

Computes:
- Per-field MAE, RMSE, normalised error
- Per-family breakdown
- Per-split breakdown
- Inference speed comparison (FEM solve_time vs GNN forward time)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-field metrics
# ---------------------------------------------------------------------------

def displacement_metrics(
    pred: np.ndarray,
    true: np.ndarray,
) -> dict[str, float]:
    """Compute displacement field metrics.

    Parameters
    ----------
    pred : np.ndarray
        Predicted displacement, shape ``(N, 3)``.
    true : np.ndarray
        Ground truth displacement, shape ``(N, 3)``.

    Returns
    -------
    dict[str, float]
        ``mae``, ``rmse``, ``max_error``, ``relative_error``.
    """
    err = pred - true
    abs_err = np.abs(err)

    mag_true = np.linalg.norm(true, axis=1)
    mag_err = np.linalg.norm(err, axis=1)
    denom = np.maximum(mag_true.max(), 1e-12)

    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "max_error": float(abs_err.max()),
        "relative_error": float(mag_err.mean() / denom),
    }


def stress_metrics(
    pred: np.ndarray,
    true: np.ndarray,
) -> dict[str, float]:
    """Compute stress field metrics (Voigt notation).

    Parameters
    ----------
    pred : np.ndarray
        Predicted stress, shape ``(N, 6)``.
    true : np.ndarray
        Ground truth stress, shape ``(N, 6)``.

    Returns
    -------
    dict[str, float]
        ``mae``, ``rmse``, ``relative_error``, ``p95_error``.
    """
    err = pred - true
    abs_err = np.abs(err)

    component_err = np.linalg.norm(err, axis=1)
    denom = np.maximum(np.linalg.norm(true, axis=1).max(), 1e-12)

    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "relative_error": float(component_err.mean() / denom),
        "p95_error": float(np.percentile(component_err, 95)),
    }


def von_mises_metrics(
    pred: np.ndarray,
    true: np.ndarray,
) -> dict[str, float]:
    """Compute von Mises stress metrics.

    Parameters
    ----------
    pred : np.ndarray
        Predicted VM stress, shape ``(N,)``.
    true : np.ndarray
        Ground truth VM stress, shape ``(N,)``.

    Returns
    -------
    dict[str, float]
        ``mae``, ``rmse``, ``relative_error``, ``correlation``,
        ``peak_error_pct``.
    """
    err = pred - true
    abs_err = np.abs(err)

    denom = np.maximum(true.max(), 1e-12)
    peak_pred = pred.max()
    peak_true = true.max()
    peak_err_pct = abs(peak_pred - peak_true) / max(peak_true, 1e-12) * 100

    corr = float(np.corrcoef(pred.ravel(), true.ravel())[0, 1])

    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "relative_error": float(abs_err.mean() / denom),
        "correlation": corr if not np.isnan(corr) else 0.0,
        "peak_error_pct": float(peak_err_pct),
    }


# ---------------------------------------------------------------------------
# Full evaluation pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Run evaluation on a DataLoader and return aggregate metrics.

    Parameters
    ----------
    model : nn.Module
        Trained MeshGraphNet.
    dataloader : DataLoader
        PyG DataLoader (val or test split).
    device : str or torch.device
        Device to run on.

    Returns
    -------
    dict[str, Any]
        Nested metrics: ``displacement``, ``stress``, ``von_mises``,
        ``timing``.
    """
    model.eval()
    device = torch.device(device)
    model = model.to(device)

    all_u_pred, all_u_true = [], []
    all_sigma_pred, all_sigma_true = [], []
    all_vm_pred, all_vm_true = [], []
    total_forward_time = 0.0
    n_samples = 0

    for batch in dataloader:
        batch = batch.to(device)

        t0 = time.perf_counter()
        preds = model(batch)
        torch.cuda.synchronize() if device.type == "cuda" else None
        total_forward_time += time.perf_counter() - t0

        n_samples += batch.num_graphs if hasattr(batch, "num_graphs") else 1

        # Displacement
        if hasattr(batch, "y_displacement"):
            all_u_pred.append(preds["u_pred"].cpu().numpy())
            all_u_true.append(batch.y_displacement.cpu().numpy())

        # Stress
        if hasattr(batch, "y_stress"):
            all_sigma_pred.append(preds["sigma_pred"].cpu().numpy())
            all_sigma_true.append(batch.y_stress.cpu().numpy())

        # Von Mises
        if hasattr(batch, "y_von_mises"):
            all_vm_pred.append(preds["vm"].cpu().numpy())
            all_vm_true.append(batch.y_von_mises.cpu().numpy())

    results: dict[str, Any] = {"n_samples": n_samples}

    if all_u_pred:
        results["displacement"] = displacement_metrics(
            np.concatenate(all_u_pred),
            np.concatenate(all_u_true),
        )

    if all_sigma_pred:
        results["stress"] = stress_metrics(
            np.concatenate(all_sigma_pred),
            np.concatenate(all_sigma_true),
        )

    if all_vm_pred:
        results["von_mises"] = von_mises_metrics(
            np.concatenate(all_vm_pred),
            np.concatenate(all_vm_true),
        )

    results["timing"] = {
        "total_forward_s": total_forward_time,
        "avg_forward_ms": (total_forward_time / max(n_samples, 1)) * 1000,
    }

    return results


def print_evaluation_report(
    results: dict[str, Any],
    split_name: str = "test",
) -> None:
    """Pretty-print evaluation results.

    Parameters
    ----------
    results : dict[str, Any]
        Output from ``evaluate_model()``.
    split_name : str
        Name of the split for the header.
    """
    print(f"\n{'=' * 60}")
    print(f"  EVALUATION REPORT — {split_name.upper()} SET")
    print(f"  ({results['n_samples']} samples)")
    print(f"{'=' * 60}")

    if "displacement" in results:
        d = results["displacement"]
        print(f"\n  Displacement:")
        print(f"    MAE:             {d['mae']:.6e}")
        print(f"    RMSE:            {d['rmse']:.6e}")
        print(f"    Max error:       {d['max_error']:.6e}")
        print(f"    Relative error:  {d['relative_error']:.4f}")

    if "stress" in results:
        s = results["stress"]
        print(f"\n  Stress (Voigt):")
        print(f"    MAE:             {s['mae']:.6e}")
        print(f"    RMSE:            {s['rmse']:.6e}")
        print(f"    Relative error:  {s['relative_error']:.4f}")
        print(f"    P95 error:       {s['p95_error']:.6e}")

    if "von_mises" in results:
        v = results["von_mises"]
        print(f"\n  Von Mises:")
        print(f"    MAE:             {v['mae']:.6e}")
        print(f"    RMSE:            {v['rmse']:.6e}")
        print(f"    Relative error:  {v['relative_error']:.4f}")
        print(f"    Correlation:     {v['correlation']:.6f}")
        print(f"    Peak error (%):  {v['peak_error_pct']:.2f}%")

    t = results["timing"]
    print(f"\n  Inference Speed:")
    print(f"    Total time:      {t['total_forward_s']:.2f}s")
    print(f"    Avg per sample:  {t['avg_forward_ms']:.1f}ms")
    print(f"{'=' * 60}\n")
