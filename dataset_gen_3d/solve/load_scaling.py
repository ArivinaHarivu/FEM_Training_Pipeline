"""Load scaling — exact linear scaling of FEM results.

Pure linear algebra, NO FEniCS dependency — locally testable.

For linear-elastic FEM, stress/displacement scale exactly linearly
with applied load. This module takes one reference FEM solve and
produces N load-level variants by scalar multiplication.

Applies three gates before accepting a scaled variant:
1. Peak stress cap: 0.9 × UTS (discards samples in plastic regime)
2. Geometric linearity — bulk: δ/L < threshold
3. Geometric linearity — bending: δ/t < threshold (half-thickness rule)

The mode-aware linearity gate (opus Task 2) uses governing_thickness
for bending families, characteristic_length for bulk families.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from dataset_gen_3d.solve.run_fem_3d import FEMResult


@dataclass
class ScaledSample:
    """One load-level variant of a base sample.

    Contains both node-averaged and element-level (DG0) fields.

    Attributes
    ----------
    target_sf : float
        Target safety factor this variant was scaled to.
    scale_factor : float
        Multiplier applied to the reference fields.
    displacement : np.ndarray
        Scaled displacement, shape (N, 3).
    stress_voigt : np.ndarray
        Scaled node-averaged Voigt stress, shape (N, 6).
    strain_voigt : np.ndarray
        Scaled node-averaged Voigt strain, shape (N, 6).
    stress_tensor : np.ndarray
        Scaled node-averaged full stress tensor, shape (N, 3, 3).
    von_mises : np.ndarray
        Scaled node-averaged von Mises stress, shape (N,).
    stress_voigt_elem : np.ndarray
        Scaled element-level DG0 Voigt stress, shape (E, 6).
    strain_voigt_elem : np.ndarray
        Scaled element-level DG0 Voigt strain, shape (E, 6).
    von_mises_elem : np.ndarray
        Scaled element-level DG0 von Mises, shape (E,).
    peak_von_mises : float
        Maximum von Mises stress in the scaled field.
    p95_von_mises : float
        95th percentile von Mises (used as scaling denominator).
    max_displacement : float
        Maximum displacement magnitude.
    accepted : bool
        Whether this variant passed all gates.
    rejection_reason : str
        Why rejected, if ``accepted`` is False.
    linearity_gate_type : str
        Which gate was applied: "bulk", "bending", or "both".
    """

    target_sf: float
    scale_factor: float
    displacement: np.ndarray
    stress_voigt: np.ndarray
    strain_voigt: np.ndarray
    stress_tensor: np.ndarray
    von_mises: np.ndarray
    stress_voigt_elem: np.ndarray
    strain_voigt_elem: np.ndarray
    von_mises_elem: np.ndarray
    peak_von_mises: float
    p95_von_mises: float
    max_displacement: float
    accepted: bool
    rejection_reason: str
    linearity_gate_type: str


def compute_scale_factors(
    reference_von_mises: np.ndarray,
    sigma_yield: float,
    target_safety_factors: list[float],
    stress_percentile: int = 95,
) -> list[tuple[float, float]]:
    """Compute scale factors to hit target safety factors.

    Uses the P95 (configurable) von Mises stress as the denominator,
    not the peak. This avoids singularity-driven suppression of the
    scale factor.

    scale_factor = (σ_yield / target_SF) / P95_vm

    Parameters
    ----------
    reference_von_mises : np.ndarray
        Von Mises stress from the reference solve, shape (N,).
    sigma_yield : float
        Material yield strength [Pa].
    target_safety_factors : list[float]
        Target safety factors (all >= 1.0 for v1).
    stress_percentile : int
        Percentile of von Mises to use as denominator (default 95).

    Returns
    -------
    list[tuple[float, float]]
        List of (target_sf, scale_factor) pairs.
    """
    p_stress = float(np.percentile(reference_von_mises, stress_percentile))

    if p_stress < 1e-10:
        # Near-zero reference stress — cannot scale meaningfully
        return [(sf, 0.0) for sf in target_safety_factors]

    factors = []
    for sf in target_safety_factors:
        target_stress = sigma_yield / sf
        factor = target_stress / p_stress
        factors.append((sf, factor))

    return factors


def apply_scaling(
    fem_result: FEMResult,
    scale_factor: float,
    target_sf: float,
    characteristic_length: float,
    peak_stress_cap: float,
    linearity_config: dict[str, Any],
    family_name: str,
    governing_thickness: Optional[float] = None,
) -> ScaledSample:
    """Apply linear scaling to a reference FEM result.

    Multiplies displacement, strain, stress, and von Mises by
    ``scale_factor``. Then applies acceptance gates.

    Parameters
    ----------
    fem_result : FEMResult
        Reference solve result.
    scale_factor : float
        Multiplier for all field quantities.
    target_sf : float
        Target safety factor.
    characteristic_length : float
        Object characteristic length [m].
    peak_stress_cap : float
        Maximum acceptable peak von Mises [Pa] (0.9 × UTS).
    linearity_config : dict[str, Any]
        The ``linearity_gate`` config block.
    family_name : str
        Geometry family name for gate selection.
    governing_thickness : float or None
        For bending families: minimum cross-section dimension [m].

    Returns
    -------
    ScaledSample
        Scaled variant with acceptance status.
    """
    # Scale nodal fields
    disp = fem_result.displacement * scale_factor
    stress_v = fem_result.stress_voigt * scale_factor
    strain_v = fem_result.strain_voigt * scale_factor
    stress_t = fem_result.stress_tensor * scale_factor
    vm = fem_result.von_mises * scale_factor

    # Scale element-level DG0 fields
    stress_v_elem = fem_result.stress_voigt_elem * scale_factor
    strain_v_elem = fem_result.strain_voigt_elem * scale_factor
    vm_elem = fem_result.von_mises_elem * scale_factor

    peak_vm = float(np.max(vm))
    p95_vm = float(np.percentile(vm, 95))
    max_disp = float(np.max(np.linalg.norm(disp, axis=1)))

    # Gate 1: Peak stress cap
    accepted = True
    rejection_reason = ""

    if peak_vm > peak_stress_cap:
        accepted = False
        rejection_reason = (
            f"Peak VM {peak_vm:.1f} Pa > cap {peak_stress_cap:.1f} Pa"
        )

    # Gate 2: Geometric linearity — mode-aware
    bulk_families = linearity_config.get("bulk_families", [])
    bending_families = linearity_config.get("bending_families", [])
    bulk_threshold = linearity_config.get("bulk_threshold", 0.05)
    bending_length_threshold = linearity_config.get(
        "bending_length_threshold", 0.05,
    )
    bending_thickness_threshold = linearity_config.get(
        "bending_thickness_threshold", 0.5,
    )

    gate_type = "bulk"

    if family_name in bending_families:
        gate_type = "bending"

        # Check 1: δ/L (same as bulk, but with bending-specific threshold)
        delta_L = max_disp / characteristic_length if characteristic_length > 0 else 0
        if delta_L > bending_length_threshold and accepted:
            accepted = False
            rejection_reason = (
                f"Bending δ/L = {delta_L:.4f} > {bending_length_threshold}"
            )

        # Check 2: δ/t (thickness-relative, the critical bending check)
        if governing_thickness is not None and governing_thickness > 0:
            delta_t = max_disp / governing_thickness
            if delta_t > bending_thickness_threshold and accepted:
                accepted = False
                rejection_reason = (
                    f"Bending δ/t = {delta_t:.4f} > {bending_thickness_threshold}"
                )
            gate_type = "both"  # both checks ran

    elif family_name in bulk_families:
        delta_L = max_disp / characteristic_length if characteristic_length > 0 else 0
        if delta_L > bulk_threshold and accepted:
            accepted = False
            rejection_reason = (
                f"Bulk δ/L = {delta_L:.4f} > {bulk_threshold}"
            )
    else:
        # Unknown family — apply bulk threshold as conservative default
        delta_L = max_disp / characteristic_length if characteristic_length > 0 else 0
        if delta_L > bulk_threshold and accepted:
            accepted = False
            rejection_reason = (
                f"Default δ/L = {delta_L:.4f} > {bulk_threshold}"
            )

    return ScaledSample(
        target_sf=target_sf,
        scale_factor=scale_factor,
        displacement=disp,
        stress_voigt=stress_v,
        strain_voigt=strain_v,
        stress_tensor=stress_t,
        von_mises=vm,
        stress_voigt_elem=stress_v_elem,
        strain_voigt_elem=strain_v_elem,
        von_mises_elem=vm_elem,
        peak_von_mises=peak_vm,
        p95_von_mises=p95_vm,
        max_displacement=max_disp,
        accepted=accepted,
        rejection_reason=rejection_reason,
        linearity_gate_type=gate_type,
    )


def generate_load_variants(
    fem_result: FEMResult,
    sigma_yield: float,
    target_safety_factors: list[float],
    characteristic_length: float,
    peak_stress_cap: float,
    linearity_config: dict[str, Any],
    family_name: str,
    governing_thickness: Optional[float] = None,
    stress_percentile: int = 95,
) -> list[ScaledSample]:
    """Generate all load-level variants from one reference solve.

    Parameters
    ----------
    fem_result : FEMResult
        Reference solve result.
    sigma_yield : float
        Yield strength [Pa].
    target_safety_factors : list[float]
        Target safety factors.
    characteristic_length : float
        Object characteristic length [m].
    peak_stress_cap : float
        0.9 × UTS [Pa].
    linearity_config : dict[str, Any]
        Linearity gate config.
    family_name : str
        Geometry family name.
    governing_thickness : float or None
        For bending families.
    stress_percentile : int
        Percentile for scaling denominator.

    Returns
    -------
    list[ScaledSample]
        One ScaledSample per target safety factor.
    """
    sf_factor_pairs = compute_scale_factors(
        fem_result.von_mises,
        sigma_yield,
        target_safety_factors,
        stress_percentile,
    )

    variants = []
    for target_sf, factor in sf_factor_pairs:
        variant = apply_scaling(
            fem_result=fem_result,
            scale_factor=factor,
            target_sf=target_sf,
            characteristic_length=characteristic_length,
            peak_stress_cap=peak_stress_cap,
            linearity_config=linearity_config,
            family_name=family_name,
            governing_thickness=governing_thickness,
        )
        variants.append(variant)

    return variants
