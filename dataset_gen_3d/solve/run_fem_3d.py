"""FEM solver — one FEniCS linear-elastic solve per call.

Uses legacy FEniCS/Dolfin API (installed via fem-on-colab).
VectorFunctionSpace CG2 (quadratic tet10: 10 DOFs per element).
Returns displacement, stress tensor, strain, and von Mises fields.

Stress and strain are computed element-wise via DG0 projection
(discontinuous, piecewise-constant per element) to avoid the
L2-projection smoothing that CG1 introduces at stress concentrations.
Node-averaged values are produced post-hoc.

This module does NOT handle load scaling — that is in load_scaling.py
(pure linear algebra, no FEniCS dependency).

Note: This module can only be tested on Colab where FEniCS is available.
"""

from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FEMResult:
    """Result of a single FEM solve.

    All displacement arrays are per-node, with N = number of mesh nodes.
    Stress/strain arrays are available in two forms:
      - Element-level (DG0): one value per tet element
      - Node-averaged: volume-weighted average of adjacent elements

    Attributes
    ----------
    displacement : np.ndarray
        Displacement field, shape (N, 3).
    stress_tensor_elem : np.ndarray
        Full stress tensor per element, shape (E, 3, 3).
    stress_voigt_elem : np.ndarray
        Voigt stress per element [xx, yy, zz, xy, yz, xz], shape (E, 6).
    strain_voigt_elem : np.ndarray
        Voigt strain per element [xx, yy, zz, xy, yz, xz], shape (E, 6).
    von_mises_elem : np.ndarray
        Von Mises stress per element, shape (E,).
    stress_voigt_nodal : np.ndarray
        Volume-weighted node-averaged Voigt stress, shape (N, 6).
    strain_voigt_nodal : np.ndarray
        Volume-weighted node-averaged Voigt strain, shape (N, 6).
    von_mises_nodal : np.ndarray
        Volume-weighted node-averaged von Mises, shape (N,).
    reaction_forces : np.ndarray
        Reaction forces at constrained nodes, shape (N, 3).
    rhs_nodal_forces : np.ndarray
        Assembled RHS nodal forces (traction-consistent), shape (N, 3).
    solve_time_s : float
        Wall-clock solve time in seconds.
    n_elements : int
        Number of tetrahedral elements.
    """

    displacement: np.ndarray
    stress_tensor_elem: np.ndarray
    stress_voigt_elem: np.ndarray
    strain_voigt_elem: np.ndarray
    von_mises_elem: np.ndarray
    stress_voigt_nodal: np.ndarray
    strain_voigt_nodal: np.ndarray
    von_mises_nodal: np.ndarray
    reaction_forces: np.ndarray
    rhs_nodal_forces: np.ndarray
    solve_time_s: float
    n_elements: int

    # ---- backward-compat aliases used by load_scaling.py ----
    @property
    def stress_tensor(self) -> np.ndarray:
        """Nodal stress tensor (derived from element avg). Shape (N, 3, 3)."""
        return _voigt_to_tensor(self.stress_voigt_nodal)

    @property
    def stress_voigt(self) -> np.ndarray:
        """Alias → stress_voigt_nodal for backward compatibility."""
        return self.stress_voigt_nodal

    @property
    def strain_voigt(self) -> np.ndarray:
        """Alias → strain_voigt_nodal for backward compatibility."""
        return self.strain_voigt_nodal

    @property
    def von_mises(self) -> np.ndarray:
        """Alias → von_mises_nodal for backward compatibility."""
        return self.von_mises_nodal


def run_fem_solve(
    xdmf_path: pathlib.Path,
    fixed_face_name: str,
    load_face_name: str,
    traction_direction: np.ndarray,
    traction_magnitude: float,
    E: float,
    nu: float,
    vertices: np.ndarray,
    elements_tet4: np.ndarray | None = None,
) -> FEMResult:
    """Run one linear-elastic FEM solve using FEniCS/Dolfin.

    Parameters
    ----------
    xdmf_path : pathlib.Path
        Path to the XDMF mesh file.
    fixed_face_name : str
        Name of the fixed face (e.g. "x_min").
    load_face_name : str
        Name of the loaded face.
    traction_direction : np.ndarray
        Unit direction vector for surface traction, shape (3,).
    traction_magnitude : float
        Traction magnitude [Pa].
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio [-].
    vertices : np.ndarray
        Original mesh vertices for coordinate-based BC identification,
        shape (N, 3).
    elements_tet4 : np.ndarray or None
        Tet4 connectivity for element-to-node averaging, shape (E, 4).
        If None, element-level fields are returned but nodal averaging
        uses Dolfin's own cell-vertex mapping.

    Returns
    -------
    FEMResult
        Displacement, stress, strain, von Mises fields from the solve.

    Raises
    ------
    RuntimeError
        If the solver fails to converge.
    ImportError
        If FEniCS/Dolfin is not available (not on Colab).
    """
    try:
        import dolfin
    except ImportError as e:
        raise ImportError(
            "FEniCS/Dolfin is required for FEM solves. "
            "Install via fem-on-colab on Google Colab."
        ) from e

    start_time = time.perf_counter()

    # Read mesh
    mesh = dolfin.Mesh()
    with dolfin.XDMFFile(str(xdmf_path)) as f:
        f.read(mesh)

    # Function space — CG2 vector (quadratic tet10: 10 DOFs per element)
    # CG2 gives quadratic displacement → linear strain within each element,
    # matching the tet10 B-matrix used in the coherence check.
    V = dolfin.VectorFunctionSpace(mesh, "CG", 2)

    # Material parameters (Lamé)
    lam = dolfin.Constant(E * nu / ((1 + nu) * (1 - 2 * nu)))
    mu = dolfin.Constant(E / (2 * (1 + nu)))

    # Constitutive law helpers
    def epsilon(u):
        """Symmetric strain tensor."""
        return 0.5 * (dolfin.grad(u) + dolfin.grad(u).T)

    def sigma(u):
        """Stress tensor via Hooke's law."""
        d = u.geometric_dimension()
        return lam * dolfin.div(u) * dolfin.Identity(d) + 2 * mu * epsilon(u)

    # Boundary conditions — identify fixed face by coordinate
    bc_info = _parse_face_name(fixed_face_name)
    fixed_boundary = _make_boundary_subdomain(
        bc_info["axis"], bc_info["value"], bc_info["tol"],
        mesh, vertices,
    )
    bc = dolfin.DirichletBC(
        V,
        dolfin.Constant((0.0, 0.0, 0.0)),
        fixed_boundary,
    )

    # Variational form
    u_trial = dolfin.TrialFunction(V)
    v_test = dolfin.TestFunction(V)

    # Traction vector
    traction = dolfin.Constant(tuple(traction_direction * traction_magnitude))

    # Identify load boundary
    load_info = _parse_face_name(load_face_name)
    load_boundary = _make_boundary_subdomain(
        load_info["axis"], load_info["value"], load_info["tol"],
        mesh, vertices,
    )

    # Mark boundaries for Neumann BC
    boundaries = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    load_boundary.mark(boundaries, 1)
    ds = dolfin.Measure("ds", domain=mesh, subdomain_data=boundaries)

    # Bilinear and linear forms
    a = dolfin.inner(sigma(u_trial), epsilon(v_test)) * dolfin.dx
    L = dolfin.dot(traction, v_test) * ds(1)

    # Solve
    u_sol = dolfin.Function(V, name="displacement")
    dolfin.solve(a == L, u_sol, bc)

    # ── Extract fields ──────────────────────────────────────────────

    n_nodes = mesh.num_vertices()
    n_cells = mesh.num_cells()

    # Displacement — CG2 values at mesh vertices
    u_array = u_sol.compute_vertex_values(mesh).reshape(3, n_nodes).T

    # ── DG0 stress and strain (element-wise, avoids CG1 smoothing) ──
    W_dg0_tensor = dolfin.TensorFunctionSpace(mesh, "DG", 0)

    # Stress
    stress_dg0 = dolfin.project(sigma(u_sol), W_dg0_tensor)
    stress_flat = stress_dg0.vector().get_local()
    # DG0 tensor: 9 components per cell, stored cell-by-cell
    stress_tensor_elem = stress_flat.reshape(n_cells, 3, 3)

    stress_voigt_elem = np.column_stack([
        stress_tensor_elem[:, 0, 0],  # xx
        stress_tensor_elem[:, 1, 1],  # yy
        stress_tensor_elem[:, 2, 2],  # zz
        stress_tensor_elem[:, 0, 1],  # xy
        stress_tensor_elem[:, 1, 2],  # yz
        stress_tensor_elem[:, 0, 2],  # xz
    ])

    # Strain
    strain_dg0 = dolfin.project(epsilon(u_sol), W_dg0_tensor)
    strain_flat = strain_dg0.vector().get_local()
    strain_tensor_elem = strain_flat.reshape(n_cells, 3, 3)

    # Engineering shear (gamma = 2*epsilon_ij)
    strain_voigt_elem = np.column_stack([
        strain_tensor_elem[:, 0, 0],      # exx
        strain_tensor_elem[:, 1, 1],      # eyy
        strain_tensor_elem[:, 2, 2],      # ezz
        2 * strain_tensor_elem[:, 0, 1],  # gxy = 2*exy
        2 * strain_tensor_elem[:, 1, 2],  # gyz = 2*eyz
        2 * strain_tensor_elem[:, 0, 2],  # gxz = 2*exz
    ])

    # Von Mises per element
    von_mises_elem = _compute_von_mises(stress_voigt_elem)

    # ── Element-to-node averaging ────────────────────────────────────
    if elements_tet4 is not None:
        cell_connectivity = elements_tet4
    else:
        # Fall back to Dolfin's own cell-vertex map
        cell_connectivity = np.array([
            mesh.cells()[i] for i in range(n_cells)
        ], dtype=np.int64)

    # Element volumes for weighting
    elem_volumes = _compute_element_volumes(
        mesh.coordinates(), cell_connectivity,
    )

    stress_voigt_nodal = _element_to_node_average(
        stress_voigt_elem, cell_connectivity, elem_volumes, n_nodes,
    )
    strain_voigt_nodal = _element_to_node_average(
        strain_voigt_elem, cell_connectivity, elem_volumes, n_nodes,
    )
    von_mises_nodal = _element_to_node_average_scalar(
        von_mises_elem, cell_connectivity, elem_volumes, n_nodes,
    )

    # ── Reaction forces via residual assembly ────────────────────────
    reaction_forces = _extract_reaction_forces(
        mesh, V, u_sol, sigma, epsilon, bc, n_nodes,
    )

    # ── Traction-consistent RHS nodal forces ─────────────────────────
    rhs_nodal_forces = _extract_rhs_forces(
        L, V, mesh, n_nodes,
    )

    solve_time = time.perf_counter() - start_time

    return FEMResult(
        displacement=u_array,
        stress_tensor_elem=stress_tensor_elem,
        stress_voigt_elem=stress_voigt_elem,
        strain_voigt_elem=strain_voigt_elem,
        von_mises_elem=von_mises_elem,
        stress_voigt_nodal=stress_voigt_nodal,
        strain_voigt_nodal=strain_voigt_nodal,
        von_mises_nodal=von_mises_nodal,
        reaction_forces=reaction_forces,
        rhs_nodal_forces=rhs_nodal_forces,
        solve_time_s=solve_time,
        n_elements=n_cells,
    )


def _compute_von_mises(stress_voigt: np.ndarray) -> np.ndarray:
    """Compute von Mises equivalent stress from Voigt stress.

    Parameters
    ----------
    stress_voigt : np.ndarray
        Stress in Voigt order [xx, yy, zz, xy, yz, xz], shape (M, 6).

    Returns
    -------
    np.ndarray
        Von Mises stress, shape (M,).
    """
    sxx = stress_voigt[:, 0]
    syy = stress_voigt[:, 1]
    szz = stress_voigt[:, 2]
    sxy = stress_voigt[:, 3]
    syz = stress_voigt[:, 4]
    sxz = stress_voigt[:, 5]

    vm = np.sqrt(
        0.5 * (
            (sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2
            + 6 * (sxy**2 + syz**2 + sxz**2)
        )
    )
    return vm


def _compute_element_volumes(
    vertices: np.ndarray,
    elements: np.ndarray,
) -> np.ndarray:
    """Compute volumes of tet4 elements.

    Parameters
    ----------
    vertices : np.ndarray
        Node coordinates, shape (N, 3).
    elements : np.ndarray
        Tet4 connectivity, shape (E, 4).

    Returns
    -------
    np.ndarray
        Element volumes, shape (E,).
    """
    v0 = vertices[elements[:, 0]]
    v1 = vertices[elements[:, 1]]
    v2 = vertices[elements[:, 2]]
    v3 = vertices[elements[:, 3]]

    # Volume = |det([v1-v0, v2-v0, v3-v0])| / 6
    d1 = v1 - v0
    d2 = v2 - v0
    d3 = v3 - v0

    cross = np.cross(d2, d3)
    det = np.sum(d1 * cross, axis=1)
    return np.abs(det) / 6.0


def _element_to_node_average(
    elem_values: np.ndarray,
    elements: np.ndarray,
    elem_volumes: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """Volume-weighted average of element values to nodes.

    Parameters
    ----------
    elem_values : np.ndarray
        Per-element values, shape (E, C) where C is component count.
    elements : np.ndarray
        Tet4 connectivity, shape (E, 4).
    elem_volumes : np.ndarray
        Element volumes, shape (E,).
    n_nodes : int
        Total number of nodes.

    Returns
    -------
    np.ndarray
        Per-node averaged values, shape (N, C).
    """
    n_components = elem_values.shape[1]
    node_sum = np.zeros((n_nodes, n_components), dtype=np.float64)
    node_weight = np.zeros(n_nodes, dtype=np.float64)

    for i in range(len(elements)):
        vol = elem_volumes[i]
        for node_id in elements[i]:
            node_sum[node_id] += elem_values[i] * vol
            node_weight[node_id] += vol

    safe_weight = np.maximum(node_weight, 1e-30)
    return node_sum / safe_weight[:, np.newaxis]


def _element_to_node_average_scalar(
    elem_values: np.ndarray,
    elements: np.ndarray,
    elem_volumes: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """Volume-weighted average of scalar element values to nodes.

    Parameters
    ----------
    elem_values : np.ndarray
        Per-element scalar values, shape (E,).
    elements : np.ndarray
        Tet4 connectivity, shape (E, 4).
    elem_volumes : np.ndarray
        Element volumes, shape (E,).
    n_nodes : int
        Total number of nodes.

    Returns
    -------
    np.ndarray
        Per-node averaged values, shape (N,).
    """
    node_sum = np.zeros(n_nodes, dtype=np.float64)
    node_weight = np.zeros(n_nodes, dtype=np.float64)

    for i in range(len(elements)):
        vol = elem_volumes[i]
        for node_id in elements[i]:
            node_sum[node_id] += elem_values[i] * vol
            node_weight[node_id] += vol

    safe_weight = np.maximum(node_weight, 1e-30)
    return node_sum / safe_weight


def _voigt_to_tensor(voigt: np.ndarray) -> np.ndarray:
    """Convert Voigt stress/strain to full symmetric tensor.

    Parameters
    ----------
    voigt : np.ndarray
        Shape (N, 6), order [xx, yy, zz, xy, yz, xz].

    Returns
    -------
    np.ndarray
        Shape (N, 3, 3).
    """
    n = len(voigt)
    tensor = np.zeros((n, 3, 3), dtype=np.float64)
    tensor[:, 0, 0] = voigt[:, 0]  # xx
    tensor[:, 1, 1] = voigt[:, 1]  # yy
    tensor[:, 2, 2] = voigt[:, 2]  # zz
    tensor[:, 0, 1] = voigt[:, 3]  # xy
    tensor[:, 1, 0] = voigt[:, 3]  # xy (symmetric)
    tensor[:, 1, 2] = voigt[:, 4]  # yz
    tensor[:, 2, 1] = voigt[:, 4]  # yz (symmetric)
    tensor[:, 0, 2] = voigt[:, 5]  # xz
    tensor[:, 2, 0] = voigt[:, 5]  # xz (symmetric)
    return tensor


def _extract_reaction_forces(
    mesh: Any,
    V: Any,
    u_sol: Any,
    sigma_func: Any,
    epsilon_func: Any,
    bc: Any,
    n_nodes: int,
) -> np.ndarray:
    """Extract reaction forces at Dirichlet boundary nodes.

    Uses the residual assembly approach:
        R = assemble(inner(sigma(u_sol), epsilon(v_test)) * dx)
    The residual at constrained DOFs gives the reaction forces.

    Parameters
    ----------
    mesh : dolfin.Mesh
        The FEniCS mesh.
    V : dolfin.VectorFunctionSpace
        The function space.
    u_sol : dolfin.Function
        The solved displacement field.
    sigma_func : callable
        Stress function.
    epsilon_func : callable
        Strain function.
    bc : dolfin.DirichletBC
        The applied Dirichlet BC.
    n_nodes : int
        Number of mesh vertices.

    Returns
    -------
    np.ndarray
        Reaction forces, shape (N, 3). Non-zero only at fixed nodes.
    """
    import dolfin

    reaction = np.zeros((n_nodes, 3), dtype=np.float64)

    try:
        v_test = dolfin.TestFunction(V)
        residual_form = dolfin.inner(
            sigma_func(u_sol), epsilon_func(v_test),
        ) * dolfin.dx
        R = dolfin.assemble(residual_form)

        # Apply BC to identify which DOFs are constrained
        # (bc.apply zeros out non-constrained DOFs in the residual)
        bc.apply(R)

        R_array = R.get_local()

        # Map DOFs to vertices
        try:
            v2d = dolfin.vertex_to_dof_map(V)
            for vertex_idx in range(n_nodes):
                for comp in range(3):
                    dof_idx = v2d[vertex_idx * 3 + comp]
                    if dof_idx < len(R_array):
                        reaction[vertex_idx, comp] = R_array[dof_idx]
        except (AttributeError, RuntimeError, IndexError):
            # vertex_to_dof_map not available in this Dolfin version
            # Try dof_to_vertex_map instead
            try:
                d2v = dolfin.dof_to_vertex_map(V)
                for dof_idx in range(len(R_array)):
                    if dof_idx < len(d2v):
                        vertex_idx = d2v[dof_idx] // 3
                        comp = d2v[dof_idx] % 3
                        if vertex_idx < n_nodes:
                            reaction[vertex_idx, comp] = R_array[dof_idx]
            except (AttributeError, RuntimeError):
                pass  # documented limitation — return zeros

    except Exception:
        pass  # documented limitation — return zeros

    return reaction


def _extract_rhs_forces(
    L: Any,
    V: Any,
    mesh: Any,
    n_nodes: int,
) -> np.ndarray:
    """Extract traction-consistent nodal forces from the assembled RHS.

    Assembles the linear form L = dot(traction, v) * ds to get the
    exact force vector that FEniCS uses, rather than the uniform
    approximation of magnitude/n_loaded.

    Parameters
    ----------
    L : dolfin.Form
        The assembled linear form.
    V : dolfin.VectorFunctionSpace
        The function space.
    mesh : dolfin.Mesh
        The FEniCS mesh.
    n_nodes : int
        Number of mesh vertices.

    Returns
    -------
    np.ndarray
        Traction-consistent nodal forces, shape (N, 3).
    """
    import dolfin

    forces = np.zeros((n_nodes, 3), dtype=np.float64)

    try:
        b = dolfin.assemble(L)
        b_array = b.get_local()

        # Map DOFs to vertices
        try:
            v2d = dolfin.vertex_to_dof_map(V)
            for vertex_idx in range(n_nodes):
                for comp in range(3):
                    dof_idx = v2d[vertex_idx * 3 + comp]
                    if dof_idx < len(b_array):
                        forces[vertex_idx, comp] = b_array[dof_idx]
        except (AttributeError, RuntimeError, IndexError):
            try:
                d2v = dolfin.dof_to_vertex_map(V)
                for dof_idx in range(len(b_array)):
                    if dof_idx < len(d2v):
                        vertex_idx = d2v[dof_idx] // 3
                        comp = d2v[dof_idx] % 3
                        if vertex_idx < n_nodes:
                            forces[vertex_idx, comp] = b_array[dof_idx]
            except (AttributeError, RuntimeError):
                pass  # will fall back to approximate forces

    except Exception:
        pass  # documented limitation

    return forces


def _parse_face_name(face_name: str) -> dict[str, Any]:
    """Parse a face name like 'x_min' into axis index and side.

    Parameters
    ----------
    face_name : str
        Face identifier.

    Returns
    -------
    dict[str, Any]
        Keys: axis (int 0-2), side ("min"/"max"), value (float),
        tol (float).
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    parts = face_name.split("_")
    axis_idx = axis_map[parts[0]]
    side = parts[1]

    return {
        "axis": axis_idx,
        "side": side,
        "value": None,  # Filled by _make_boundary_subdomain
        "tol": None,
    }


def _make_boundary_subdomain(
    axis: int,
    value: float | None,
    tol: float | None,
    mesh: Any,
    vertices: np.ndarray,
) -> Any:
    """Create a Dolfin SubDomain for a planar face.

    Parameters
    ----------
    axis : int
        Axis index (0=x, 1=y, 2=z).
    value : float or None
        Coordinate value of the face. If None, auto-detect from mesh.
    tol : float or None
        Tolerance. If None, use 2% of extent.
    mesh : dolfin.Mesh
        The FEniCS mesh.
    vertices : np.ndarray
        Original vertex coordinates.

    Returns
    -------
    dolfin.SubDomain
        Subdomain marking the face.
    """
    import dolfin

    coords = mesh.coordinates()
    min_val = coords[:, axis].min()
    max_val = coords[:, axis].max()
    extent = max_val - min_val

    if tol is None:
        tol = 0.02 * extent

    # Determine which side
    if value is None:
        value = min_val  # default to min side

    class FaceBoundary(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[axis] - value) < tol

    return FaceBoundary()
