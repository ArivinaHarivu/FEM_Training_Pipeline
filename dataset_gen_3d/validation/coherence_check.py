"""Coherence check — validates solver output via independent B-matrix strain.

Per-sample gate:
1. Compute strain from displacement via B-matrix (b_matrix_tetra.py)
2. Apply 3D Hooke's law (full constitutive matrix, not plane stress/strain)
3. Compute von Mises from the recomputed stress
4. Compare to solver's own von Mises output
5. Pass/fail at configured tolerance

Supports both tet10 (10-node quadratic) and tet4 (4-node linear)
element connectivity:
- tet10: uses the full quadratic B-matrix from b_matrix_tetra.py
- tet4: uses a simplified constant-strain B-matrix (linear tet)

When DG0 element-level solver output is available, comparison is
done element-to-element directly — much cleaner than the previous
CG1 node-averaged comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dataset_gen_3d.validation.b_matrix_tetra import (
    compute_all_element_strains,
)


@dataclass(frozen=True)
class CoherenceResult:
    """Result of a per-sample coherence check.

    Attributes
    ----------
    passed : bool
        Whether the sample passed the coherence check.
    relative_error : float
        Relative L2 error of von Mises (recomputed vs. solver).
    max_element_error : float
        Maximum per-element relative error.
    mean_element_error : float
        Mean per-element relative error.
    """

    passed: bool
    relative_error: float
    max_element_error: float
    mean_element_error: float


def run_coherence_check(
    vertices: np.ndarray,
    elements: np.ndarray,
    displacement: np.ndarray,
    solver_von_mises: np.ndarray,
    E: float,
    nu: float,
    tolerance: float = 0.05,
    solver_stress_voigt_elem: np.ndarray | None = None,
) -> CoherenceResult:
    """Run the per-sample coherence check.

    Dispatches to tet10 or tet4 B-matrix based on element connectivity
    shape. If DG0 element-level solver output is provided, compares
    element-to-element directly (more accurate than node-based).

    Parameters
    ----------
    vertices : np.ndarray
        Node coordinates, shape (N, 3).
    elements : np.ndarray
        Tet connectivity, shape (E, 4) for tet4 or (E, 10) for tet10.
    displacement : np.ndarray
        Solver displacement, shape (N, 3).
    solver_von_mises : np.ndarray
        Solver's von Mises stress, shape (N,) nodal or (E,) element-level.
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio [-].
    tolerance : float
        Maximum acceptable relative L2 error.
    solver_stress_voigt_elem : np.ndarray or None
        If available, DG0 element-level Voigt stress shape (E, 6)
        for direct element-to-element comparison.

    Returns
    -------
    CoherenceResult
        Pass/fail with error metrics.
    """
    nodes_per_elem = elements.shape[1]

    if nodes_per_elem == 10:
        # tet10 path — use quadratic B-matrix
        return _coherence_tet10(
            vertices, elements, displacement, solver_von_mises,
            E, nu, tolerance, solver_stress_voigt_elem,
        )
    elif nodes_per_elem == 4:
        # tet4 path — use linear B-matrix (constant strain)
        return _coherence_tet4(
            vertices, elements, displacement, solver_von_mises,
            E, nu, tolerance, solver_stress_voigt_elem,
        )
    else:
        raise ValueError(
            f"Unsupported element type with {nodes_per_elem} nodes/element. "
            f"Expected 4 (tet4) or 10 (tet10)."
        )


def _coherence_tet10(
    vertices: np.ndarray,
    elements: np.ndarray,
    displacement: np.ndarray,
    solver_von_mises: np.ndarray,
    E: float,
    nu: float,
    tolerance: float,
    solver_stress_voigt_elem: np.ndarray | None,
) -> CoherenceResult:
    """Coherence check using tet10 quadratic B-matrix.

    Uses the 10-node shape functions and 4-point Gauss quadrature
    from b_matrix_tetra.py. Computes volume-averaged strain per element.
    """
    # Compute element-level strain via tet10 B-matrix
    elem_strains = compute_all_element_strains(vertices, elements, displacement)

    # Apply Hooke's law
    C = _constitutive_matrix(E, nu)
    elem_stresses = (C @ elem_strains.T).T  # (E, 6)
    elem_vm = _von_mises_from_voigt(elem_stresses)

    return _compare_results(
        elem_vm, solver_von_mises, tolerance,
        solver_stress_voigt_elem, elem_stresses,
    )


def _coherence_tet4(
    vertices: np.ndarray,
    elements: np.ndarray,
    displacement: np.ndarray,
    solver_von_mises: np.ndarray,
    E: float,
    nu: float,
    tolerance: float,
    solver_stress_voigt_elem: np.ndarray | None,
) -> CoherenceResult:
    """Coherence check using tet4 constant-strain B-matrix.

    For linear tets, strain is constant within each element.
    B-matrix is computed from the 4 corner nodes only.
    """
    n_elements = len(elements)
    elem_strains = np.zeros((n_elements, 6), dtype=np.float64)

    for i in range(n_elements):
        node_ids = elements[i]
        elem_verts = vertices[node_ids]  # (4, 3)
        elem_disp = displacement[node_ids]  # (4, 3)

        try:
            B, _ = _b_matrix_tet4(elem_verts)
            u_flat = elem_disp.flatten()  # (12,)
            elem_strains[i] = B @ u_flat
        except ValueError:
            pass  # degenerate element — leave strain as zero

    # Apply Hooke's law
    C = _constitutive_matrix(E, nu)
    elem_stresses = (C @ elem_strains.T).T  # (E, 6)
    elem_vm = _von_mises_from_voigt(elem_stresses)

    return _compare_results(
        elem_vm, solver_von_mises, tolerance,
        solver_stress_voigt_elem, elem_stresses,
    )


def _b_matrix_tet4(vertices: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute the constant-strain B-matrix for a tet4 element.

    Parameters
    ----------
    vertices : np.ndarray
        Shape (4, 3) — corner node coordinates.

    Returns
    -------
    tuple[np.ndarray, float]
        B-matrix shape (6, 12) and element volume × 6.
    """
    # Form the shape function gradient matrix
    # For tet4, dN/dx = J^{-1} @ dN/d(xi,eta,zeta) is constant.
    v0, v1, v2, v3 = vertices

    # Jacobian: columns = edge vectors from v0
    J = np.column_stack([v1 - v0, v2 - v0, v3 - v0])
    det_J = np.linalg.det(J)

    if abs(det_J) < 1e-30:
        raise ValueError("Degenerate tet4 element (zero Jacobian)")

    J_inv = np.linalg.inv(J)

    # Shape function derivatives in natural coords:
    # N0 = 1 - xi - eta - zeta → dN0 = [-1, -1, -1]
    # N1 = xi                  → dN1 = [1, 0, 0]
    # N2 = eta                 → dN2 = [0, 1, 0]
    # N3 = zeta                → dN3 = [0, 0, 1]
    dN_nat = np.array([
        [-1, 1, 0, 0],
        [-1, 0, 1, 0],
        [-1, 0, 0, 1],
    ], dtype=np.float64)

    # Physical-space derivatives
    dN_phys = J_inv @ dN_nat  # (3, 4)

    # Build B-matrix (6 × 12)
    B = np.zeros((6, 12), dtype=np.float64)

    for i in range(4):
        col = 3 * i
        dNi_dx = dN_phys[0, i]
        dNi_dy = dN_phys[1, i]
        dNi_dz = dN_phys[2, i]

        B[0, col] = dNi_dx      # εxx
        B[1, col + 1] = dNi_dy  # εyy
        B[2, col + 2] = dNi_dz  # εzz
        B[3, col] = dNi_dy      # γxy
        B[3, col + 1] = dNi_dx
        B[4, col + 1] = dNi_dz  # γyz
        B[4, col + 2] = dNi_dy
        B[5, col] = dNi_dz      # γxz
        B[5, col + 2] = dNi_dx

    return B, det_J


def _compare_results(
    recomputed_vm_elem: np.ndarray,
    solver_von_mises: np.ndarray,
    tolerance: float,
    solver_stress_voigt_elem: np.ndarray | None,
    recomputed_stress_elem: np.ndarray,
) -> CoherenceResult:
    """Compare recomputed vs solver von Mises at element level.

    Parameters
    ----------
    recomputed_vm_elem : np.ndarray
        B-matrix-derived von Mises per element, shape (E,).
    solver_von_mises : np.ndarray
        Solver von Mises. Shape (N,) if nodal, (E,) if element-level.
    tolerance : float
        Pass/fail threshold.
    solver_stress_voigt_elem : np.ndarray or None
        DG0 element-level Voigt stress from solver, shape (E, 6).
    recomputed_stress_elem : np.ndarray
        B-matrix-derived Voigt stress per element, shape (E, 6).
    """
    n_elements = len(recomputed_vm_elem)

    # If we have DG0 element-level solver output, compare directly
    if solver_stress_voigt_elem is not None:
        solver_vm_elem = _von_mises_from_voigt(solver_stress_voigt_elem)
    elif len(solver_von_mises) == n_elements:
        # Already element-level
        solver_vm_elem = solver_von_mises
    else:
        # Nodal solver output — cannot do element-level comparison,
        # fall back to L2 norm comparison at nodal level
        solver_norm = np.linalg.norm(solver_von_mises)
        if solver_norm < 1e-10:
            return CoherenceResult(
                passed=True, relative_error=0.0,
                max_element_error=0.0, mean_element_error=0.0,
            )

        # Approximate: compare mean recomputed element VM against
        # mean solver nodal VM as a global sanity check
        recomputed_mean = float(np.mean(recomputed_vm_elem))
        solver_mean = float(np.mean(solver_von_mises))
        rel_err = abs(recomputed_mean - solver_mean) / max(solver_mean, 1e-10)
        return CoherenceResult(
            passed=rel_err <= tolerance,
            relative_error=rel_err,
            max_element_error=rel_err,
            mean_element_error=rel_err,
        )

    # Element-level comparison
    solver_norm = np.linalg.norm(solver_vm_elem)
    if solver_norm < 1e-10:
        return CoherenceResult(
            passed=True, relative_error=0.0,
            max_element_error=0.0, mean_element_error=0.0,
        )

    diff_norm = np.linalg.norm(recomputed_vm_elem - solver_vm_elem)
    relative_error = float(diff_norm / solver_norm)

    # Per-element relative errors
    safe_denom = np.maximum(solver_vm_elem, 1e-10)
    elem_errors = np.abs(recomputed_vm_elem - solver_vm_elem) / safe_denom

    return CoherenceResult(
        passed=relative_error <= tolerance,
        relative_error=relative_error,
        max_element_error=float(np.max(elem_errors)),
        mean_element_error=float(np.mean(elem_errors)),
    )


def _constitutive_matrix(E: float, nu: float) -> np.ndarray:
    """Build the 6×6 isotropic linear-elastic constitutive matrix.

    Voigt order [xx, yy, zz, xy, yz, xz], engineering shear.

    Parameters
    ----------
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.

    Returns
    -------
    np.ndarray
        Shape (6, 6).
    """
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    return np.array([
        [lam + 2*mu, lam,        lam,        0,  0,  0],
        [lam,        lam + 2*mu, lam,        0,  0,  0],
        [lam,        lam,        lam + 2*mu, 0,  0,  0],
        [0,          0,          0,          mu, 0,  0],
        [0,          0,          0,          0,  mu, 0],
        [0,          0,          0,          0,  0,  mu],
    ], dtype=np.float64)


def _von_mises_from_voigt(stress_voigt: np.ndarray) -> np.ndarray:
    """Compute von Mises from Voigt stress.

    Parameters
    ----------
    stress_voigt : np.ndarray
        Shape (E, 6) in order [xx, yy, zz, xy, yz, xz].

    Returns
    -------
    np.ndarray
        Von Mises stress, shape (E,).
    """
    s = stress_voigt
    return np.sqrt(
        0.5 * (
            (s[:, 0] - s[:, 1])**2
            + (s[:, 1] - s[:, 2])**2
            + (s[:, 2] - s[:, 0])**2
            + 6 * (s[:, 3]**2 + s[:, 4]**2 + s[:, 5]**2)
        )
    )
