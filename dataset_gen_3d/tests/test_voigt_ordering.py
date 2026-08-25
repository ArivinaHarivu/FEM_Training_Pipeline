"""Voigt ordering validation — uniaxial tension analytical test.

Validates that the Voigt stress tensor mapping matches [xx, yy, zz, xy, yz, xz].
Supports both modern FEniCSx (dolfinx) and legacy dolfin.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_voigt_ordering_uniaxial_tension() -> None:
    """Verify Voigt ordering with a uniaxial tension analytical case."""
    try:
        import dolfinx
        _test_voigt_fenicsx()
    except ImportError:
        try:
            import dolfin
            _test_voigt_legacy_dolfin()
        except ImportError:
            pytest.skip("Neither FEniCSx (dolfinx) nor legacy FEniCS is available.")


def _test_voigt_fenicsx() -> None:
    import dolfinx
    import dolfinx.fem.petsc
    import ufl
    from dolfinx import default_scalar_type, fem, mesh
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_cube(comm, 5, 5, 5, cell_type=mesh.CellType.tetrahedron)

    V = fem.functionspace(msh, ("Lagrange", 2, (3,)))
    E_val = 200e9
    nu_val = 0.3

    lam = fem.Constant(msh, default_scalar_type(E_val * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))))
    mu = fem.Constant(msh, default_scalar_type(E_val / (2 * (1 + nu_val))))

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return lam * ufl.div(u) * ufl.Identity(3) + 2 * mu * epsilon(u)

    fdim = msh.topology.dim - 1

    def fixed_facets_locator(x):
        return np.isclose(x[0], 0.0)

    fixed_facets = mesh.locate_entities_boundary(msh, fdim, fixed_facets_locator)
    fixed_dofs = fem.locate_dofs_topological(V, fdim, fixed_facets)
    u_zero = np.zeros(3, dtype=default_scalar_type)
    bc = fem.dirichletbc(u_zero, fixed_dofs, V)

    def load_facets_locator(x):
        return np.isclose(x[0], 1.0)

    load_facets = mesh.locate_entities_boundary(msh, fdim, load_facets_locator)
    facet_tags = mesh.meshtags(msh, fdim, load_facets, np.full_like(load_facets, 1, dtype=np.int32))
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

    traction_mag = 1e6
    T = fem.Constant(msh, default_scalar_type([traction_mag, 0.0, 0.0]))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = fem.form(ufl.inner(sigma(u), epsilon(v)) * ufl.dx)
    L = fem.form(ufl.dot(T, v) * ds(1))

    problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    u_sol = problem.solve()

    W_dg0 = fem.functionspace(msh, ("DG", 0, (3, 3)))
    stress_expr = fem.Expression(sigma(u_sol), W_dg0.element.interpolation_points)
    stress_dg0 = fem.Function(W_dg0)
    stress_dg0.interpolate(stress_expr)

    n_cells = msh.topology.index_map(msh.topology.dim).size_local
    stress_tensor = stress_dg0.x.array.reshape(n_cells, 3, 3)

    stress_voigt = np.column_stack([
        stress_tensor[:, 0, 0],  # xx
        stress_tensor[:, 1, 1],  # yy
        stress_tensor[:, 2, 2],  # zz
        stress_tensor[:, 0, 1],  # xy
        stress_tensor[:, 1, 2],  # yz
        stress_tensor[:, 0, 2],  # xz
    ])

    mean_stress = np.mean(stress_voigt, axis=0)

    assert abs(mean_stress[0]) > 0.5 * traction_mag, f"σ_xx ({mean_stress[0]}) not dominant."
    assert np.argmax(np.abs(mean_stress)) == 0, f"σ_xx not max component: {mean_stress}"
    assert np.all(np.abs(mean_stress[3:6]) < 0.1 * abs(mean_stress[0])), f"Shear not zero: {mean_stress[3:6]}"
    print(f"✓ FEniCSx Voigt ordering validated! Mean Voigt: {mean_stress}")


def _test_voigt_legacy_dolfin() -> None:
    import dolfin
    mesh_obj = dolfin.UnitCubeMesh(5, 5, 5)
    V = dolfin.VectorFunctionSpace(mesh_obj, "CG", 2)
    E_val, nu_val = 200e9, 0.3
    lam = dolfin.Constant(E_val * nu_val / ((1 + nu_val) * (1 - 2 * nu_val)))
    mu = dolfin.Constant(E_val / (2 * (1 + nu_val)))

    def epsilon(u): return 0.5 * (dolfin.grad(u) + dolfin.grad(u).T)
    def sigma(u): return lam * dolfin.div(u) * dolfin.Identity(3) + 2 * mu * epsilon(u)

    class FixedEnd(dolfin.SubDomain):
        def inside(self, x, on_boundary): return on_boundary and abs(x[0]) < 1e-10

    class LoadedEnd(dolfin.SubDomain):
        def inside(self, x, on_boundary): return on_boundary and abs(x[0] - 1.0) < 1e-10

    bc = dolfin.DirichletBC(V, dolfin.Constant((0.0, 0.0, 0.0)), FixedEnd())
    boundaries = dolfin.MeshFunction("size_t", mesh_obj, mesh_obj.topology().dim() - 1)
    boundaries.set_all(0)
    LoadedEnd().mark(boundaries, 1)
    ds = dolfin.Measure("ds", domain=mesh_obj, subdomain_data=boundaries)

    traction_mag = 1e6
    traction = dolfin.Constant((traction_mag, 0.0, 0.0))
    u_tr, v_te = dolfin.TrialFunction(V), dolfin.TestFunction(V)
    a = dolfin.inner(sigma(u_tr), epsilon(v_te)) * dolfin.dx
    L = dolfin.dot(traction, v_te) * ds(1)

    u_sol = dolfin.Function(V)
    dolfin.solve(a == L, u_sol, bc)

    W_dg0 = dolfin.TensorFunctionSpace(mesh_obj, "DG", 0)
    stress_dg0 = dolfin.project(sigma(u_sol), W_dg0)
    stress_tensor = stress_dg0.vector().get_local().reshape(mesh_obj.num_cells(), 3, 3)

    stress_voigt = np.column_stack([
        stress_tensor[:, 0, 0], stress_tensor[:, 1, 1], stress_tensor[:, 2, 2],
        stress_tensor[:, 0, 1], stress_tensor[:, 1, 2], stress_tensor[:, 0, 2],
    ])
    mean_stress = np.mean(stress_voigt, axis=0)
    assert abs(mean_stress[0]) > 0.5 * traction_mag
    assert np.argmax(np.abs(mean_stress)) == 0
    print(f"✓ Legacy FEniCS Voigt ordering validated! Mean Voigt: {mean_stress}")
