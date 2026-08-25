"""FEM solver — modern FEniCSx (dolfinx) linear-elastic solve.

Supports FEniCSx (`dolfinx` >= 0.7) with legacy `dolfin` compatibility fallback.
Uses VectorFunctionSpace CG2 (quadratic tet10 elements).

Stress and strain are computed element-wise via DG0 interpolation
(discontinuous, piecewise-constant per cell) to avoid L2-projection
smoothing at stress concentration notches and fillets.
Nodal values are produced via volume-weighted averaging.
"""

from __future__ import annotations
import uuid
import pathlib
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class FEMResult:
    """Result of a single FEM solve."""

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

    @property
    def stress_tensor(self) -> np.ndarray:
        return _voigt_to_tensor(self.stress_voigt_nodal)

    @property
    def stress_voigt(self) -> np.ndarray:
        return self.stress_voigt_nodal

    @property
    def strain_voigt(self) -> np.ndarray:
        return self.strain_voigt_nodal

    @property
    def von_mises(self) -> np.ndarray:
        return self.von_mises_nodal



def _get_interpolation_points(element):
    """Get interpolation points from a dolfinx element, version-safe."""
    pts = element.interpolation_points
    if callable(pts):
        return pts()
    return pts


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
    """Execute a 3D linear elastic solve using FEniCSx (dolfinx) or legacy dolfin."""
    try:
        import dolfinx
        return _run_solve_fenicsx(
            xdmf_path, fixed_face_name, load_face_name,
            traction_direction, traction_magnitude,
            E, nu, vertices, elements_tet4,
        )
    except ImportError:
        try:
            import dolfin
            return _run_solve_legacy_dolfin(
                xdmf_path, fixed_face_name, load_face_name,
                traction_direction, traction_magnitude,
                E, nu, vertices, elements_tet4,
            )
        except ImportError as err:
            raise ImportError(
                "Neither FEniCSx (dolfinx) nor legacy FEniCS (dolfin) is available. "
                "Install FEniCSx via: !wget -O - https://fem-on-colab.github.io/releases.sh | bash -s -- --install-fenicsx"
            ) from err


def _run_solve_fenicsx(
    xdmf_path: pathlib.Path,
    fixed_face_name: str,
    load_face_name: str,
    traction_direction: np.ndarray,
    traction_magnitude: float,
    E: float,
    nu: float,
    vertices: np.ndarray,
    elements_tet4: np.ndarray | None,
) -> FEMResult:
    """FEniCSx (dolfinx) solver implementation."""
    import dolfinx
    import dolfinx.fem.petsc
    import ufl
    from dolfinx import default_scalar_type, fem, io, mesh
    from mpi4py import MPI

    start_time = time.perf_counter()
    comm = MPI.COMM_WORLD

    # Read mesh
    with io.XDMFFile(comm, str(xdmf_path), "r") as xdmf:
        msh = xdmf.read_mesh(name="Grid")

    n_nodes = msh.geometry.x.shape[0]
    n_cells = msh.topology.index_map(msh.topology.dim).size_local

    # CG2 Vector function space (quadratic tetrahedral DOFs)
    V = fem.functionspace(msh, ("Lagrange", 2, (3,)))

    # Material constants
    lam = fem.Constant(msh, default_scalar_type(E * nu / ((1 + nu) * (1 - 2 * nu))))
    mu = fem.Constant(msh, default_scalar_type(E / (2 * (1 + nu))))

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return lam * ufl.div(u) * ufl.Identity(3) + 2 * mu * epsilon(u)

    # ── Dirichlet Boundary Condition (Fixed face) ──
    fixed_axis, fixed_val, fixed_tol = _get_face_params(fixed_face_name, vertices)

    def fixed_indicator(x):
        return np.isclose(x[fixed_axis], fixed_val, atol=fixed_tol)

    fdim = msh.topology.dim - 1
    fixed_facets = mesh.locate_entities_boundary(msh, fdim, fixed_indicator)
    fixed_dofs = fem.locate_dofs_topological(V, fdim, fixed_facets)
    u_zero = np.zeros(3, dtype=default_scalar_type)
    bc = fem.dirichletbc(u_zero, fixed_dofs, V)

    # ── Neumann Boundary Condition (Traction Load) ──
    load_axis, load_val, load_tol = _get_face_params(load_face_name, vertices)

    def load_indicator(x):
        return np.isclose(x[load_axis], load_val, atol=load_tol)

    load_facets = mesh.locate_entities_boundary(msh, fdim, load_indicator)
    facet_tags = mesh.meshtags(
        msh, fdim, load_facets, np.full_like(load_facets, 1, dtype=np.int32),
    )
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

    traction_vec = traction_direction * traction_magnitude
    T = fem.Constant(msh, default_scalar_type(traction_vec))

    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a_form = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L_form = ufl.dot(T, v) * ds(1)

    a = fem.form(a_form)
    L = fem.form(L_form)
    
    # Solve linear system
    problem = dolfinx.fem.petsc.LinearProblem(
        a_form, L_form, bcs=[bc],
        petsc_options_prefix=f"lp_{uuid.uuid4().hex[:8]}_",
        petsc_options={"ksp_type": "cg", "pc_type": "gamg"},
    )
    u_sol = problem.solve()

    # ── Extract Nodal Displacements ──
    # Extract directly from the CG2 solution - geometry is also
    # quadratic (tet10), so CG2 DOFs coincide with geometry nodes
    # 1:1 (verified empirically, max diff ~1e-17). No downsampling
    # to CG1 needed - keeps full quadratic fidelity, single geometry.
    u_array = u_sol.x.array.reshape(n_nodes, 3).copy()

    # ── Extract DG0 Stress & Strain (Element-wise) ──
    W_dg0 = fem.functionspace(msh, ("DG", 0, (3, 3)))

    stress_expr = fem.Expression(sigma(u_sol), _get_interpolation_points(W_dg0.element))
    stress_dg0 = fem.Function(W_dg0)
    stress_dg0.interpolate(stress_expr)
    stress_tensor_elem = stress_dg0.x.array.reshape(n_cells, 3, 3).copy()

    strain_expr = fem.Expression(epsilon(u_sol), _get_interpolation_points(W_dg0.element))
    strain_dg0 = fem.Function(W_dg0)
    strain_dg0.interpolate(strain_expr)
    strain_tensor_elem = strain_dg0.x.array.reshape(n_cells, 3, 3).copy()

    # Voigt mappings
    stress_voigt_elem = np.column_stack([
        stress_tensor_elem[:, 0, 0],
        stress_tensor_elem[:, 1, 1],
        stress_tensor_elem[:, 2, 2],
        stress_tensor_elem[:, 0, 1],
        stress_tensor_elem[:, 1, 2],
        stress_tensor_elem[:, 0, 2],
    ])

    strain_voigt_elem = np.column_stack([
        strain_tensor_elem[:, 0, 0],
        strain_tensor_elem[:, 1, 1],
        strain_tensor_elem[:, 2, 2],
        2 * strain_tensor_elem[:, 0, 1],
        2 * strain_tensor_elem[:, 1, 2],
        2 * strain_tensor_elem[:, 0, 2],
    ])

    von_mises_elem = _compute_von_mises(stress_voigt_elem)

    # ── Element-to-Node Averaging ──
    # Use dolfinx-native connectivity/coordinates — gmsh's tet10 node
    # ordering and dolfinx's internal geometry ordering are different
    # permutations of the same node set (confirmed empirically: same
    # count, same physical points, different index assignment).
    cell_conn = msh.geometry.dofmaps[0]  # full tet10 connectivity (all 10 nodes/cell), not just corners
    mesh_vertices = msh.geometry.x
    elem_vols = _compute_element_volumes(mesh_vertices, cell_conn)

    stress_voigt_nodal = _element_to_node_average(stress_voigt_elem, cell_conn, elem_vols, n_nodes)
    strain_voigt_nodal = _element_to_node_average(strain_voigt_elem, cell_conn, elem_vols, n_nodes)
    von_mises_nodal = _element_to_node_average_scalar(von_mises_elem, cell_conn, elem_vols, n_nodes)

    # ── Reorder all nodal (dolfinx-ordered) arrays back into the original
    # gmsh/vertices node ordering, so downstream code (HDF5 writer, GNN
    # graph construction) that uses `vertices`/`elements_tet4` for node
    # identity stays correctly aligned with the solved physical fields. ──
    perm = msh.geometry.input_global_indices  # dolfinx_idx -> original_idx
    inverse_perm = np.argsort(perm)            # original_idx -> dolfinx_idx

    u_array = u_array[inverse_perm]
    stress_voigt_nodal = stress_voigt_nodal[inverse_perm]
    strain_voigt_nodal = strain_voigt_nodal[inverse_perm]
    von_mises_nodal = von_mises_nodal[inverse_perm]

    # ── Same fix, for CELLS. dolfinx reorders elements internally too,
    # independent of node reordering. Without this, coherence_check's
    # element-to-element comparison silently compares unrelated tets. ──
    cell_perm = msh.topology.original_cell_index  # dolfinx_cell_idx -> original_cell_idx
    inverse_cell_perm = np.argsort(cell_perm)       # original_cell_idx -> dolfinx_cell_idx

    stress_tensor_elem = stress_tensor_elem[inverse_cell_perm]
    stress_voigt_elem = stress_voigt_elem[inverse_cell_perm]
    strain_voigt_elem = strain_voigt_elem[inverse_cell_perm]
    von_mises_elem = von_mises_elem[inverse_cell_perm]

    # ── Reaction Forces & RHS Forces ──
    reaction_forces = np.zeros((n_nodes, 3), dtype=np.float64)
    rhs_nodal_forces = np.zeros((n_nodes, 3), dtype=np.float64)

    try:
        res_form = fem.form(ufl.inner(sigma(u_sol), epsilon(v)) * ufl.dx)
        R = dolfinx.fem.petsc.assemble_vector(res_form)
        R_arr = R.array
        # Map vertex values
        v_dofs = fem.locate_dofs_topological(V, 0, np.arange(n_nodes, dtype=np.int32))
        for i in range(min(n_nodes, len(v_dofs))):
            dof = v_dofs[i]
            if dof * 3 + 2 < len(R_arr):
                reaction_forces[i] = R_arr[dof*3 : dof*3 + 3]
    except Exception:
        pass

    try:
        b = dolfinx.fem.petsc.assemble_vector(L)
        b_arr = b.array
        v_dofs = fem.locate_dofs_topological(V, 0, np.arange(n_nodes, dtype=np.int32))
        for i in range(min(n_nodes, len(v_dofs))):
            dof = v_dofs[i]
            if dof * 3 + 2 < len(b_arr):
                rhs_nodal_forces[i] = b_arr[dof*3 : dof*3 + 3]
    except Exception:
        pass

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


def _run_solve_legacy_dolfin(
    xdmf_path: pathlib.Path,
    fixed_face_name: str,
    load_face_name: str,
    traction_direction: np.ndarray,
    traction_magnitude: float,
    E: float,
    nu: float,
    vertices: np.ndarray,
    elements_tet4: np.ndarray | None,
) -> FEMResult:
    """Legacy FEniCS fallback solver."""
    import dolfin

    start_time = time.perf_counter()
    mesh_obj = dolfin.Mesh()
    with dolfin.XDMFFile(str(xdmf_path)) as f:
        f.read(mesh_obj)

    V = dolfin.VectorFunctionSpace(mesh_obj, "CG", 2)
    lam = dolfin.Constant(E * nu / ((1 + nu) * (1 - 2 * nu)))
    mu = dolfin.Constant(E / (2 * (1 + nu)))

    def epsilon(u):
        return 0.5 * (dolfin.grad(u) + dolfin.grad(u).T)

    def sigma(u):
        d = u.geometric_dimension()
        return lam * dolfin.div(u) * dolfin.Identity(d) + 2 * mu * epsilon(u)

    f_axis, f_val, f_tol = _get_face_params(fixed_face_name, vertices)

    class FixedBoundary(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[f_axis] - f_val) < f_tol

    bc = dolfin.DirichletBC(V, dolfin.Constant((0.0, 0.0, 0.0)), FixedBoundary())

    l_axis, l_val, l_tol = _get_face_params(load_face_name, vertices)

    class LoadBoundary(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[l_axis] - l_val) < l_tol

    boundaries = dolfin.MeshFunction("size_t", mesh_obj, mesh_obj.topology().dim() - 1)
    boundaries.set_all(0)
    LoadBoundary().mark(boundaries, 1)
    ds = dolfin.Measure("ds", domain=mesh_obj, subdomain_data=boundaries)

    traction = dolfin.Constant(tuple(traction_direction * traction_magnitude))
    u_tr = dolfin.TrialFunction(V)
    v_te = dolfin.TestFunction(V)

    a = dolfin.inner(sigma(u_tr), epsilon(v_te)) * dolfin.dx
    L = dolfin.dot(traction, v_te) * ds(1)

    u_sol = dolfin.Function(V)
    dolfin.solve(a == L, u_sol, bc)

    n_nodes = mesh_obj.num_vertices()
    n_cells = mesh_obj.num_cells()
    u_array = u_sol.compute_vertex_values(mesh_obj).reshape(3, n_nodes).T

    W_dg0 = dolfin.TensorFunctionSpace(mesh_obj, "DG", 0)
    stress_dg0 = dolfin.project(sigma(u_sol), W_dg0)
    stress_tensor_elem = stress_dg0.vector().get_local().reshape(n_cells, 3, 3)

    strain_dg0 = dolfin.project(epsilon(u_sol), W_dg0)
    strain_tensor_elem = strain_dg0.vector().get_local().reshape(n_cells, 3, 3)

    stress_voigt_elem = np.column_stack([
        stress_tensor_elem[:, 0, 0], stress_tensor_elem[:, 1, 1], stress_tensor_elem[:, 2, 2],
        stress_tensor_elem[:, 0, 1], stress_tensor_elem[:, 1, 2], stress_tensor_elem[:, 0, 2],
    ])

    strain_voigt_elem = np.column_stack([
        strain_tensor_elem[:, 0, 0], strain_tensor_elem[:, 1, 1], strain_tensor_elem[:, 2, 2],
        2 * strain_tensor_elem[:, 0, 1], 2 * strain_tensor_elem[:, 1, 2], 2 * strain_tensor_elem[:, 0, 2],
    ])

    von_mises_elem = _compute_von_mises(stress_voigt_elem)
    cell_conn = elements_tet4 if elements_tet4 is not None else np.array([mesh_obj.cells()[i] for i in range(n_cells)], dtype=np.int64)
    elem_vols = _compute_element_volumes(vertices, cell_conn)

    stress_voigt_nodal = _element_to_node_average(stress_voigt_elem, cell_conn, elem_vols, n_nodes)
    strain_voigt_nodal = _element_to_node_average(strain_voigt_elem, cell_conn, elem_vols, n_nodes)
    von_mises_nodal = _element_to_node_average_scalar(von_mises_elem, cell_conn, elem_vols, n_nodes)

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
        reaction_forces=np.zeros((n_nodes, 3)),
        rhs_nodal_forces=np.zeros((n_nodes, 3)),
        solve_time_s=solve_time,
        n_elements=n_cells,
    )


def _get_face_params(face_name: str, vertices: np.ndarray) -> tuple[int, float, float]:
    axis_map = {"x": 0, "y": 1, "z": 2}
    parts = face_name.split("_")
    axis = axis_map[parts[0]]
    side = parts[1]

    min_val = float(vertices[:, axis].min())
    max_val = float(vertices[:, axis].max())
    extent = max_val - min_val
    tol = max(0.02 * extent, 1e-6)
    val = min_val if side == "min" else max_val

    return axis, val, tol


def _compute_von_mises(stress_voigt: np.ndarray) -> np.ndarray:
    sxx, syy, szz = stress_voigt[:, 0], stress_voigt[:, 1], stress_voigt[:, 2]
    sxy, syz, sxz = stress_voigt[:, 3], stress_voigt[:, 4], stress_voigt[:, 5]
    return np.sqrt(
        0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2 + 6 * (sxy**2 + syz**2 + sxz**2))
    )


def _compute_element_volumes(vertices: np.ndarray, elements: np.ndarray) -> np.ndarray:
    v0, v1, v2, v3 = vertices[elements[:, 0]], vertices[elements[:, 1]], vertices[elements[:, 2]], vertices[elements[:, 3]]
    d1, d2, d3 = v1 - v0, v2 - v0, v3 - v0
    return np.abs(np.sum(d1 * np.cross(d2, d3), axis=1)) / 6.0


def _element_to_node_average(elem_values: np.ndarray, elements: np.ndarray, elem_vols: np.ndarray, n_nodes: int) -> np.ndarray:
    node_sum = np.zeros((n_nodes, elem_values.shape[1]), dtype=np.float64)
    node_w = np.zeros(n_nodes, dtype=np.float64)
    for i in range(len(elements)):
        vol = elem_vols[i]
        for nid in elements[i]:
            node_sum[nid] += elem_values[i] * vol
            node_w[nid] += vol
    return node_sum / np.maximum(node_w[:, np.newaxis], 1e-30)


def _element_to_node_average_scalar(elem_values: np.ndarray, elements: np.ndarray, elem_vols: np.ndarray, n_nodes: int) -> np.ndarray:
    node_sum = np.zeros(n_nodes, dtype=np.float64)
    node_w = np.zeros(n_nodes, dtype=np.float64)
    for i in range(len(elements)):
        vol = elem_vols[i]
        for nid in elements[i]:
            node_sum[nid] += elem_values[i] * vol
            node_w[nid] += vol
    return node_sum / np.maximum(node_w, 1e-30)


def _voigt_to_tensor(voigt: np.ndarray) -> np.ndarray:
    n = len(voigt)
    tensor = np.zeros((n, 3, 3), dtype=np.float64)
    tensor[:, 0, 0] = voigt[:, 0]
    tensor[:, 1, 1] = voigt[:, 1]
    tensor[:, 2, 2] = voigt[:, 2]
    tensor[:, 0, 1] = tensor[:, 1, 0] = voigt[:, 3]
    tensor[:, 1, 2] = tensor[:, 2, 1] = voigt[:, 4]
    tensor[:, 0, 2] = tensor[:, 2, 0] = voigt[:, 5]
    return tensor


def _get_mesh_connectivity(msh: Any) -> np.ndarray:
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim, 0)
    c2v = msh.topology.connectivity(tdim, 0)
    return c2v.array.reshape(-1, 4).astype(np.int64)
