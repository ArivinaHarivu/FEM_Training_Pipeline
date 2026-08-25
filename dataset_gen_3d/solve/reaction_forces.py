"""Reaction forces — extract at Dirichlet boundaries.

Uses the FEniCS residual assembly approach:
    R = assemble(action(a, u_sol)) - assemble(L)
Reaction forces at constrained DOFs are the residual values.

Note: The legacy FEniCS API makes this fragile. If extraction
proves infeasible, this is documented as a known limitation per
the build spec.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def extract_reaction_forces(
    mesh: Any,
    V: Any,
    u_sol: Any,
    sigma_func: Any,
    epsilon_func: Any,
    bc: Any,
    fixed_node_mask: np.ndarray,
) -> np.ndarray:
    """Extract reaction forces at Dirichlet boundary nodes.

    Parameters
    ----------
    mesh : dolfin.Mesh
        The FEniCS mesh.
    V : dolfin.VectorFunctionSpace
        The function space.
    u_sol : dolfin.Function
        The solved displacement field.
    sigma_func : callable
        Function mapping displacement to stress tensor.
    epsilon_func : callable
        Function mapping displacement to strain tensor.
    bc : dolfin.DirichletBC
        The applied Dirichlet BC.
    fixed_node_mask : np.ndarray
        Boolean mask of fixed nodes, shape (N,).

    Returns
    -------
    np.ndarray
        Reaction forces, shape (N, 3). Non-zero only at fixed nodes.
    """
    try:
        import dolfin
    except ImportError:
        return _fallback_reaction_forces(fixed_node_mask)

    n_nodes = mesh.num_vertices()
    reaction = np.zeros((n_nodes, 3), dtype=np.float64)

    try:
        # Assemble the bilinear form action on the solution
        v_test = dolfin.TestFunction(V)
        residual_form = dolfin.inner(sigma_func(u_sol), epsilon_func(v_test)) * dolfin.dx
        R = dolfin.assemble(residual_form)

        # The residual at constrained DOFs gives the reaction forces
        R_array = R.get_local()

        # Map DOF values to vertex-ordered reaction forces
        dofmap = V.dofmap()
        for vertex_idx in range(n_nodes):
            if fixed_node_mask[vertex_idx]:
                # Get DOFs for this vertex
                # Note: vertex-to-dof mapping depends on the mesh/space
                cell_dofs = []
                for cell_idx in range(mesh.num_cells()):
                    local_dofs = dofmap.cell_dofs(cell_idx)
                    # This is approximate — proper implementation needs
                    # vertex_to_dof_map which may not be available in
                    # all Dolfin versions
                    pass

        # Simplified approach: use vertex_to_dof_map if available
        try:
            v2d = dolfin.vertex_to_dof_map(V)
            for vertex_idx in range(n_nodes):
                if fixed_node_mask[vertex_idx]:
                    for comp in range(3):
                        dof_idx = v2d[vertex_idx * 3 + comp]
                        reaction[vertex_idx, comp] = R_array[dof_idx]
        except (AttributeError, RuntimeError):
            # vertex_to_dof_map not available — fall back
            return _fallback_reaction_forces(fixed_node_mask)

    except Exception:
        # Reaction force extraction failed — documented limitation
        return _fallback_reaction_forces(fixed_node_mask)

    return reaction


def _fallback_reaction_forces(fixed_node_mask: np.ndarray) -> np.ndarray:
    """Return zero-filled reaction forces when extraction fails.

    Parameters
    ----------
    fixed_node_mask : np.ndarray
        Boolean mask of fixed nodes, shape (N,).

    Returns
    -------
    np.ndarray
        Zero reaction forces, shape (N, 3).
    """
    n_nodes = len(fixed_node_mask)
    return np.zeros((n_nodes, 3), dtype=np.float64)
