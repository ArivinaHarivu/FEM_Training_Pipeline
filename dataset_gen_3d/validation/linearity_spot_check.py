"""Linearity spot-check — empirical verification of linear scaling.

For a configurable fraction of base samples, re-solves at the highest
scaled load (lowest SF) and compares against the linearly-scaled result.
Detects solver tolerance issues or accidental nonlinear flags.

This module requires FEniCS (Colab only). The logic is written locally
but execution happens during the calibration phase on Colab.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LinearityCheckResult:
    """Result of a linearity spot-check for one sample.

    Attributes
    ----------
    base_sample_id : str
        The base sample being checked.
    family : str
        Geometry family.
    scale_bucket : str
        Scale bucket.
    target_sf : float
        The safety factor variant being checked (lowest SF = highest load).
    max_displacement_error : float
        Max relative error in displacement magnitude.
    max_von_mises_error : float
        Max relative error in von Mises stress.
    passed : bool
        Whether both errors are within tolerance.
    """

    base_sample_id: str
    family: str
    scale_bucket: str
    target_sf: float
    max_displacement_error: float
    max_von_mises_error: float
    passed: bool


def select_samples_for_verification(
    manifest_df: Any,
    sample_rate: float,
    seed: int = 42,
) -> list[str]:
    """Select base samples for linearity verification.

    Deterministic given a fixed seed.

    Parameters
    ----------
    manifest_df : pd.DataFrame
        The generation manifest.
    sample_rate : float
        Fraction of base samples to verify (e.g. 0.01 = 1%).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[str]
        base_sample_ids selected for verification.
    """
    import pandas as pd

    base_ids = manifest_df["base_sample_id"].unique()
    rng = np.random.default_rng(seed)

    n_select = max(1, int(len(base_ids) * sample_rate))
    selected = rng.choice(base_ids, size=n_select, replace=False)

    return sorted(selected.tolist())


def compare_scaled_vs_resolved(
    scaled_displacement: np.ndarray,
    scaled_von_mises: np.ndarray,
    resolved_displacement: np.ndarray,
    resolved_von_mises: np.ndarray,
    tolerance: float = 0.01,
    base_sample_id: str = "",
    family: str = "",
    scale_bucket: str = "",
    target_sf: float = 0.0,
) -> LinearityCheckResult:
    """Compare a linearly-scaled variant against an independent re-solve.

    Parameters
    ----------
    scaled_displacement : np.ndarray
        Displacement from linear scaling, shape (N, 3).
    scaled_von_mises : np.ndarray
        Von Mises from linear scaling, shape (N,).
    resolved_displacement : np.ndarray
        Displacement from independent FEM solve, shape (N, 3).
    resolved_von_mises : np.ndarray
        Von Mises from independent solve, shape (N,).
    tolerance : float
        Maximum acceptable relative error (default 1%).
    base_sample_id : str
        Sample identifier for logging.
    family : str
        Geometry family.
    scale_bucket : str
        Scale bucket.
    target_sf : float
        Target safety factor of the variant.

    Returns
    -------
    LinearityCheckResult
        Comparison result with pass/fail.
    """
    # Displacement error
    disp_mag_scaled = np.linalg.norm(scaled_displacement, axis=1)
    disp_mag_resolved = np.linalg.norm(resolved_displacement, axis=1)
    disp_denom = np.maximum(disp_mag_resolved, 1e-30)
    disp_errors = np.abs(disp_mag_scaled - disp_mag_resolved) / disp_denom
    max_disp_error = float(np.max(disp_errors))

    # Von Mises error
    vm_denom = np.maximum(resolved_von_mises, 1e-30)
    vm_errors = np.abs(scaled_von_mises - resolved_von_mises) / vm_denom
    max_vm_error = float(np.max(vm_errors))

    passed = max_disp_error <= tolerance and max_vm_error <= tolerance

    return LinearityCheckResult(
        base_sample_id=base_sample_id,
        family=family,
        scale_bucket=scale_bucket,
        target_sf=target_sf,
        max_displacement_error=max_disp_error,
        max_von_mises_error=max_vm_error,
        passed=passed,
    )


def generate_linearity_report(
    results: list[LinearityCheckResult],
    max_family_failure_rate: float = 0.05,
) -> dict[str, Any]:
    """Generate the linearity spot-check report.

    Parameters
    ----------
    results : list[LinearityCheckResult]
        All spot-check results.
    max_family_failure_rate : float
        Maximum acceptable failure rate per family.

    Returns
    -------
    dict[str, Any]
        Report with per-family stats and overall pass/fail.
    """
    report: dict[str, Any] = {
        "total_checked": len(results),
        "total_passed": sum(1 for r in results if r.passed),
        "total_failed": sum(1 for r in results if not r.passed),
        "per_family": {},
        "blocking_failures": [],
    }

    # Group by family
    families: dict[str, list[LinearityCheckResult]] = {}
    for r in results:
        families.setdefault(r.family, []).append(r)

    for family, family_results in families.items():
        n_total = len(family_results)
        n_failed = sum(1 for r in family_results if not r.passed)
        failure_rate = n_failed / n_total if n_total > 0 else 0.0
        worst_disp = max(r.max_displacement_error for r in family_results)
        worst_vm = max(r.max_von_mises_error for r in family_results)

        report["per_family"][family] = {
            "checked": n_total,
            "passed": n_total - n_failed,
            "failed": n_failed,
            "failure_rate": failure_rate,
            "worst_displacement_error": worst_disp,
            "worst_von_mises_error": worst_vm,
        }

        if failure_rate > max_family_failure_rate:
            report["blocking_failures"].append(
                f"Family '{family}' failure rate {failure_rate:.1%} "
                f"> max {max_family_failure_rate:.1%}"
            )

    report["all_passed"] = len(report["blocking_failures"]) == 0
    return report
