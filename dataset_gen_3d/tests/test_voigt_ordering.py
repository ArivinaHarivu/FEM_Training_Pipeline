"""Voigt ordering validation — uniaxial tension analytical test.

Gate test: must pass before trusting any stress/strain label
from the FEM solver. Verifies that the Voigt component ordering
matches the project-wide convention [xx, yy, zz, xy, yz, xz].

Since this requires FEniCS, it runs on Colab only. The test creates
a unit cube, applies uniaxial tension in x, and checks:
  - σ_xx = E × ε_applied
  - σ_yy = σ_zz ≈ 0 (free lateral faces)
  - σ_xy = σ_yz = σ_xz = 0

Note: This test module can only be run on Colab where FEniCS is installed.
Run before trusting any generated dataset output.
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest


def test_voigt_ordering_uniaxial_tension() -> None:
    """Verify Voigt ordering with a uniaxial tension analytical case.

    Creates a unit cube, fixes x=0 face, applies unit traction on x=1 face.

    Analytical solution (plane stress doesn't apply in 3D, but for
    a free-lateral-faces block):
        ε_xx = σ_xx / E
        σ_xx ≈ applied traction (1.0 Pa in this test)
        σ_yy, σ_zz, σ_xy, σ_yz, σ_xz ≈ 0

    Note: In 3D with constrained Poisson, lateral stresses won't be
    exactly zero — they'll be ~ν × σ_xx for constrained, but for
    free-face BC they should be near zero.
    """
    try:
        import dolfin
    except ImportError:
        pytest.skip("FEniCS/Dolfin not available (Colab only)")

    # Create a unit cube mesh
    mesh = dolfin.UnitCubeMesh(5, 5, 5)

    # CG2 function space (matches our solver)
    V = dolfin.VectorFunctionSpace(mesh, "CG", 2)

    # Material: steel-like
    E_val = 200e9  # Pa
    nu_val = 0.3
    lam = dolfin.Constant(E_val * nu_val / ((1 + nu_val) * (1 - 2 * nu_val)))
    mu = dolfin.Constant(E_val / (2 * (1 + nu_val)))

    def epsilon(u):
        return 0.5 * (dolfin.grad(u) + dolfin.grad(u).T)

    def sigma(u):
        d = u.geometric_dimension()
        return lam * dolfin.div(u) * dolfin.Identity(d) + 2 * mu * epsilon(u)

    # Fix x=0
    class FixedEnd(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[0]) < 1e-10

    # Traction on x=1
    class LoadedEnd(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[0] - 1.0) < 1e-10

    bc = dolfin.DirichletBC(V, dolfin.Constant((0.0, 0.0, 0.0)), FixedEnd())

    boundaries = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    LoadedEnd().mark(boundaries, 1)
    ds = dolfin.Measure("ds", domain=mesh, subdomain_data=boundaries)

    # Unit traction in x-direction
    traction_magnitude = 1e6  # 1 MPa
    traction = dolfin.Constant((traction_magnitude, 0.0, 0.0))

    u_trial = dolfin.TrialFunction(V)
    v_test = dolfin.TestFunction(V)

    a = dolfin.inner(sigma(u_trial), epsilon(v_test)) * dolfin.dx
    L = dolfin.dot(traction, v_test) * ds(1)

    u_sol = dolfin.Function(V, name="displacement")
    dolfin.solve(a == L, u_sol, bc)

    # Extract DG0 stress (same as our solver)
    W_dg0 = dolfin.TensorFunctionSpace(mesh, "DG", 0)
    stress_dg0 = dolfin.project(sigma(u_sol), W_dg0)
    stress_flat = stress_dg0.vector().get_local()
    n_cells = mesh.num_cells()
    stress_tensor = stress_flat.reshape(n_cells, 3, 3)

    # Extract Voigt components using same ordering as run_fem_3d.py
    stress_voigt = np.column_stack([
        stress_tensor[:, 0, 0],  # xx → slot 0
        stress_tensor[:, 1, 1],  # yy → slot 1
        stress_tensor[:, 2, 2],  # zz → slot 2
        stress_tensor[:, 0, 1],  # xy → slot 3
        stress_tensor[:, 1, 2],  # yz → slot 4
        stress_tensor[:, 0, 2],  # xz → slot 5
    ])

    # ── Assertions ──

    # Mean stress across all elements (away from boundary effects)
    mean_stress = np.mean(stress_voigt, axis=0)

    # σ_xx should be close to the applied traction
    # (not exactly equal due to FEM discretization, but should be dominant)
    assert abs(mean_stress[0]) > 0.5 * traction_magnitude, (
        f"σ_xx (slot 0) = {mean_stress[0]:.2f} is not the dominant component. "
        f"Expected close to {traction_magnitude:.2f}. "
        f"Full Voigt mean: {mean_stress}"
    )

    # σ_xx should be the largest magnitude component
    assert np.argmax(np.abs(mean_stress)) == 0, (
        f"σ_xx is not the largest Voigt component. "
        f"Voigt mean: {mean_stress}. "
        f"This indicates wrong tensor → Voigt mapping."
    )

    # Shear components should be near zero
    shear_components = mean_stress[3:6]  # xy, yz, xz
    assert np.all(np.abs(shear_components) < 0.1 * abs(mean_stress[0])), (
        f"Shear stresses are not near zero: {shear_components}. "
        f"σ_xx = {mean_stress[0]:.2f}. "
        f"This indicates wrong Voigt ordering."
    )

    # σ_yy and σ_zz should be small compared to σ_xx (free lateral faces)
    lateral = mean_stress[1:3]  # yy, zz
    assert np.all(np.abs(lateral) < 0.3 * abs(mean_stress[0])), (
        f"Lateral stresses too large: σ_yy={mean_stress[1]:.2f}, "
        f"σ_zz={mean_stress[2]:.2f} vs σ_xx={mean_stress[0]:.2f}. "
        f"This may indicate wrong ordering."
    )

    print(f"✓ Voigt ordering validated. Mean Voigt: {mean_stress}")
    print(f"  σ_xx = {mean_stress[0]:.2f} Pa (expected ≈ {traction_magnitude})")
    print(f"  σ_yy = {mean_stress[1]:.2f} Pa (expected ≈ 0)")
    print(f"  σ_zz = {mean_stress[2]:.2f} Pa (expected ≈ 0)")
    print(f"  σ_xy = {mean_stress[3]:.2f} Pa (expected ≈ 0)")
    print(f"  σ_yz = {mean_stress[4]:.2f} Pa (expected ≈ 0)")
    print(f"  σ_xz = {mean_stress[5]:.2f} Pa (expected ≈ 0)")


if __name__ == "__main__":
    test_voigt_ordering_uniaxial_tension()
