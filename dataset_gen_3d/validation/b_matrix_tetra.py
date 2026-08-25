"""B-matrix for 10-node quadratic tetrahedra (tet10).

The 10-node tet has 10 shape functions (4 corner + 6 mid-edge nodes).
Unlike the linear tet, strain is NOT constant within the element —
it varies linearly. The B-matrix must be evaluated at specific points
(Gauss quadrature points or corner nodes).

This implementation evaluates at the 4 Gauss points of a 4-point
quadrature rule, then extrapolates to nodes for comparison with
solver output.

Voigt order: [xx, yy, zz, xy, yz, xz] (project-wide convention).
Engineering shear: γ = 2ε_ij for off-diagonal components.
"""

from __future__ import annotations

import numpy as np


# 4-point Gauss quadrature for tetrahedra
# Points in barycentric coordinates (L1, L2, L3, L4)
# Hammer-Stroud rule: exact for degree-2 polynomials
_GAUSS_POINTS_BARY = np.array([
    [0.58541020, 0.13819660, 0.13819660, 0.13819660],
    [0.13819660, 0.58541020, 0.13819660, 0.13819660],
    [0.13819660, 0.13819660, 0.58541020, 0.13819660],
    [0.13819660, 0.13819660, 0.13819660, 0.58541020],
])
# Weights (equal for this rule, sum to 1/6 which is ref tet volume)
_GAUSS_WEIGHTS = np.array([1.0 / 24.0] * 4)

# Natural coordinates (xi, eta, zeta) from barycentric
# L1 = 1 - xi - eta - zeta, L2 = xi, L3 = eta, L4 = zeta
_GAUSS_POINTS_NAT = _GAUSS_POINTS_BARY[:, 1:4]  # (4, 3): [xi, eta, zeta]


def shape_functions_tet10(xi: float, eta: float, zeta: float) -> np.ndarray:
    """Evaluate tet10 shape functions at a natural coordinate point.

    Node ordering follows the Gmsh convention:
        0-3: corner nodes
        4-9: mid-edge nodes
              4 = edge(0,1), 5 = edge(1,2), 6 = edge(0,2)
              7 = edge(0,3), 8 = edge(1,3), 9 = edge(2,3)

    Parameters
    ----------
    xi, eta, zeta : float
        Natural coordinates in [0, 1].

    Returns
    -------
    np.ndarray
        Shape function values, shape (10,).
    """
    L1 = 1.0 - xi - eta - zeta
    L2 = xi
    L3 = eta
    L4 = zeta

    N = np.zeros(10)

    # Corner nodes: Ni = Li(2Li - 1)
    N[0] = L1 * (2 * L1 - 1)
    N[1] = L2 * (2 * L2 - 1)
    N[2] = L3 * (2 * L3 - 1)
    N[3] = L4 * (2 * L4 - 1)

    # Mid-edge nodes: Nij = 4*Li*Lj
    N[4] = 4 * L1 * L2   # edge (0,1)
    N[5] = 4 * L2 * L3   # edge (1,2)
    N[6] = 4 * L1 * L3   # edge (0,2)
    N[7] = 4 * L1 * L4   # edge (0,3)
    N[8] = 4 * L2 * L4   # edge (1,3)
    N[9] = 4 * L3 * L4   # edge (2,3)

    return N


def shape_function_derivatives_tet10(
    xi: float, eta: float, zeta: float,
) -> np.ndarray:
    """Derivatives of tet10 shape functions w.r.t. natural coordinates.

    Parameters
    ----------
    xi, eta, zeta : float
        Natural coordinates.

    Returns
    -------
    np.ndarray
        Shape (3, 10) — [dN/dξ, dN/dη, dN/dζ] for each of 10 nodes.
    """
    L1 = 1.0 - xi - eta - zeta
    L2 = xi
    L3 = eta
    L4 = zeta

    dN = np.zeros((3, 10))

    # dL/d(xi, eta, zeta):
    # dL1/dxi = -1, dL1/deta = -1, dL1/dzeta = -1
    # dL2/dxi =  1, dL2/deta =  0, dL2/dzeta =  0
    # dL3/dxi =  0, dL3/deta =  1, dL3/dzeta =  0
    # dL4/dxi =  0, dL4/deta =  0, dL4/dzeta =  1

    # Corner nodes: dNi/dξj = dLi/dξj × (4Li - 1)
    # Node 0: N0 = L1(2L1-1), dN0/dxi = dL1/dxi × (4L1-1) = -(4L1-1)
    dN[0, 0] = -(4 * L1 - 1)  # dN0/dxi
    dN[1, 0] = -(4 * L1 - 1)  # dN0/deta
    dN[2, 0] = -(4 * L1 - 1)  # dN0/dzeta

    dN[0, 1] = (4 * L2 - 1)   # dN1/dxi
    dN[1, 1] = 0.0
    dN[2, 1] = 0.0

    dN[0, 2] = 0.0
    dN[1, 2] = (4 * L3 - 1)   # dN2/deta
    dN[2, 2] = 0.0

    dN[0, 3] = 0.0
    dN[1, 3] = 0.0
    dN[2, 3] = (4 * L4 - 1)   # dN3/dzeta

    # Mid-edge nodes: N4 = 4*L1*L2, etc.
    # N4 = 4*L1*L2: dN4/dxi = 4*(dL1/dxi*L2 + L1*dL2/dxi) = 4*(-L2 + L1)
    dN[0, 4] = 4 * (L1 - L2)     # dN4/dxi
    dN[1, 4] = 4 * (-L2)         # dN4/deta
    dN[2, 4] = 4 * (-L2)         # dN4/dzeta

    # N5 = 4*L2*L3
    dN[0, 5] = 4 * L3            # dN5/dxi
    dN[1, 5] = 4 * L2            # dN5/deta
    dN[2, 5] = 0.0

    # N6 = 4*L1*L3
    dN[0, 6] = 4 * (-L3)         # dN6/dxi
    dN[1, 6] = 4 * (L1 - L3)     # dN6/deta
    dN[2, 6] = 4 * (-L3)         # dN6/dzeta

    # N7 = 4*L1*L4
    dN[0, 7] = 4 * (-L4)         # dN7/dxi
    dN[1, 7] = 4 * (-L4)         # dN7/deta
    dN[2, 7] = 4 * (L1 - L4)     # dN7/dzeta

    # N8 = 4*L2*L4
    dN[0, 8] = 4 * L4            # dN8/dxi
    dN[1, 8] = 0.0
    dN[2, 8] = 4 * L2            # dN8/dzeta

    # N9 = 4*L3*L4
    dN[0, 9] = 0.0
    dN[1, 9] = 4 * L4            # dN9/deta
    dN[2, 9] = 4 * L3            # dN9/dzeta

    return dN


def compute_b_matrix_tet10(
    vertices: np.ndarray,
    xi: float,
    eta: float,
    zeta: float,
) -> tuple[np.ndarray, float]:
    """Compute B-matrix for a 10-node tet at a specific natural coordinate.

    Parameters
    ----------
    vertices : np.ndarray
        Shape (10, 3) — coordinates of all 10 tet nodes.
    xi, eta, zeta : float
        Natural coordinates of the evaluation point.

    Returns
    -------
    tuple[np.ndarray, float]
        B-matrix shape (6, 30) and the Jacobian determinant.
    """
    dN_nat = shape_function_derivatives_tet10(xi, eta, zeta)  # (3, 10)

    # Jacobian: J = dN_nat @ vertices  →  (3, 3)
    J = dN_nat @ vertices  # (3, 10) @ (10, 3) = (3, 3)
    det_J = np.linalg.det(J)

    if abs(det_J) < 1e-30:
        raise ValueError("Degenerate tet10 element (zero Jacobian)")

    J_inv = np.linalg.inv(J)

    # Physical-space derivatives: dN_phys = J_inv @ dN_nat  →  (3, 10)
    dN_phys = J_inv @ dN_nat

    # Build B-matrix (6 × 30)
    # Each node contributes 3 DOFs (ux, uy, uz)
    B = np.zeros((6, 30), dtype=np.float64)

    for i in range(10):
        col = 3 * i
        dNi_dx = dN_phys[0, i]
        dNi_dy = dN_phys[1, i]
        dNi_dz = dN_phys[2, i]

        # εxx = du/dx
        B[0, col] = dNi_dx
        # εyy = dv/dy
        B[1, col + 1] = dNi_dy
        # εzz = dw/dz
        B[2, col + 2] = dNi_dz
        # γxy = du/dy + dv/dx (engineering shear)
        B[3, col] = dNi_dy
        B[3, col + 1] = dNi_dx
        # γyz = dv/dz + dw/dy
        B[4, col + 1] = dNi_dz
        B[4, col + 2] = dNi_dy
        # γxz = du/dz + dw/dx
        B[5, col] = dNi_dz
        B[5, col + 2] = dNi_dx

    return B, det_J


def compute_element_strain_tet10(
    vertices: np.ndarray,
    displacements: np.ndarray,
) -> np.ndarray:
    """Compute strain at Gauss points for a single tet10 element.

    Parameters
    ----------
    vertices : np.ndarray
        Shape (10, 3) — all 10 node coordinates.
    displacements : np.ndarray
        Shape (10, 3) — nodal displacements.

    Returns
    -------
    np.ndarray
        Strain at each Gauss point, shape (4, 6).
        Voigt order [εxx, εyy, εzz, γxy, γyz, γxz].
    """
    u_flat = displacements.flatten()  # (30,)
    strains = np.zeros((4, 6), dtype=np.float64)

    for gp in range(4):
        xi, eta, zeta = _GAUSS_POINTS_NAT[gp]
        B, _ = compute_b_matrix_tet10(vertices, xi, eta, zeta)
        strains[gp] = B @ u_flat

    return strains


def compute_element_strain_averaged_tet10(
    vertices: np.ndarray,
    displacements: np.ndarray,
) -> np.ndarray:
    """Compute volume-averaged strain for a single tet10 element.

    Uses 4-point Gauss quadrature for the volume-weighted average.

    Parameters
    ----------
    vertices : np.ndarray
        Shape (10, 3).
    displacements : np.ndarray
        Shape (10, 3).

    Returns
    -------
    np.ndarray
        Volume-averaged Voigt strain, shape (6,).
    """
    u_flat = displacements.flatten()
    strain_avg = np.zeros(6, dtype=np.float64)
    total_weight = 0.0

    for gp in range(4):
        xi, eta, zeta = _GAUSS_POINTS_NAT[gp]
        B, det_J = compute_b_matrix_tet10(vertices, xi, eta, zeta)
        w = _GAUSS_WEIGHTS[gp] * abs(det_J)
        strain_avg += w * (B @ u_flat)
        total_weight += w

    if total_weight > 1e-30:
        strain_avg /= total_weight

    return strain_avg


def compute_all_element_strains(
    vertices: np.ndarray,
    elements: np.ndarray,
    displacements: np.ndarray,
) -> np.ndarray:
    """Compute volume-averaged strain for all tet10 elements.

    Parameters
    ----------
    vertices : np.ndarray
        Node coordinates, shape (N, 3).
    elements : np.ndarray
        Tet10 connectivity, shape (E, 10), 0-based.
    displacements : np.ndarray
        Nodal displacements, shape (N, 3).

    Returns
    -------
    np.ndarray
        Per-element volume-averaged Voigt strain, shape (E, 6).
    """
    n_elements = len(elements)
    strains = np.zeros((n_elements, 6), dtype=np.float64)

    for i in range(n_elements):
        node_ids = elements[i]
        elem_verts = vertices[node_ids]
        elem_disp = displacements[node_ids]

        try:
            strains[i] = compute_element_strain_averaged_tet10(
                elem_verts, elem_disp,
            )
        except ValueError:
            # Degenerate element — leave strain as zero
            pass

    return strains


def element_volume_tet10(vertices: np.ndarray) -> float:
    """Compute volume of a tet10 element via Gauss quadrature.

    Parameters
    ----------
    vertices : np.ndarray
        Shape (10, 3).

    Returns
    -------
    float
        Element volume (always positive).
    """
    volume = 0.0
    for gp in range(4):
        xi, eta, zeta = _GAUSS_POINTS_NAT[gp]
        dN_nat = shape_function_derivatives_tet10(xi, eta, zeta)
        J = dN_nat @ vertices
        det_J = np.linalg.det(J)
        volume += _GAUSS_WEIGHTS[gp] * abs(det_J)
    return volume
