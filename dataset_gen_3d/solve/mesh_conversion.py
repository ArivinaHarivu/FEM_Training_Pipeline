"""Mesh conversion — Gmsh .msh → meshio → Dolfin XDMF.

Handles the mesh format bridge between Gmsh output and the FEniCS solver.
Extracts 3D tetrahedral cells in two forms:
  - tet4 (linear): written to XDMF for Dolfin compatibility
  - tet10 (quadratic): extracted for coherence check and GNN graph builder

Also provides surface node detection from boundary facets.
"""

from __future__ import annotations

import pathlib
from typing import Any

import meshio
import numpy as np


def convert_msh_to_xdmf(
    msh_path: pathlib.Path,
    xdmf_path: pathlib.Path,
) -> dict[str, Any]:
    """Convert a Gmsh .msh file to Dolfin-readable XDMF, preserving full
    tet10 (quadratic) topology when present.

    Modern DOLFINx supports quadratic geometric meshes natively — no need
    to project down to tet4 corners. Preserving tet10 keeps node numbering
    consistent between the solver and the original mesh extraction used
    for HDF5/GNN graph construction, avoiding remap/index mismatches.

    Parameters
    ----------
    msh_path : pathlib.Path
        Input Gmsh .msh file.
    xdmf_path : pathlib.Path
        Output XDMF file path.

    Returns
    -------
    dict[str, Any]
        Metadata: node_count, element_count, has_tets, is_quadratic.

    Raises
    ------
    ValueError
        If no tetrahedral cells are found in the mesh.
    """
    mesh = meshio.read(str(msh_path))

    tet10_cells = [cb for cb in mesh.cells if cb.type == "tetra10"]
    tet4_cells = [cb for cb in mesh.cells if cb.type == "tetra"]

    if tet10_cells:
        write_cells = tet10_cells
        is_quadratic = True
    elif tet4_cells:
        write_cells = tet4_cells
        is_quadratic = False
    else:
        raise ValueError(f"No tetrahedral cells found in {msh_path}")

    out_mesh = meshio.Mesh(points=mesh.points, cells=write_cells)
    meshio.write(str(xdmf_path), out_mesh)

    n_nodes = mesh.points.shape[0]
    n_elements = sum(len(cb.data) for cb in write_cells)

    return {
        "node_count": n_nodes,
        "element_count": n_elements,
        "has_tets": True,
        "is_quadratic": is_quadratic,
    }


def extract_mesh_data(msh_path: pathlib.Path) -> dict[str, np.ndarray]:
    """Extract raw mesh data from a Gmsh .msh file.

    Returns node coordinates and tetrahedral connectivity in both
    tet4 and tet10 forms (when available). The tet10 connectivity
    is used for the coherence check (B-matrix) and GNN Experiment A.
    The tet4 connectivity is used for GNN Experiment B.

    Parameters
    ----------
    msh_path : pathlib.Path
        Input Gmsh .msh file.

    Returns
    -------
    dict[str, np.ndarray]
        Keys:
        - ``vertices``: shape (N, 3) node coordinates
        - ``elements_tet4``: shape (E, 4) tet4 connectivity, 0-based
        - ``elements_tet10``: shape (E, 10) tet10 connectivity, 0-based
          (None if only tet4 is available)
        - ``is_surface_node``: shape (N,) boolean mask of surface nodes
    """
    mesh = meshio.read(str(msh_path))

    # Collect tet4 and tet10 connectivity
    tet4_blocks = [cb.data for cb in mesh.cells if cb.type == "tetra"]
    tet10_blocks = [cb.data for cb in mesh.cells if cb.type == "tetra10"]

    vertices = mesh.points.astype(np.float64)

    if tet10_blocks:
        elements_tet10 = np.vstack(tet10_blocks).astype(np.int64)
        # Derive tet4 from tet10 corners (first 4 nodes)
        elements_tet4 = elements_tet10[:, :4].copy()
    elif tet4_blocks:
        elements_tet4 = np.vstack(tet4_blocks).astype(np.int64)
        elements_tet10 = None
    else:
        raise ValueError(f"No tetrahedra in {msh_path}")

    # Compute surface node mask from boundary facets
    is_surface = compute_surface_node_mask(vertices, elements_tet4)

    result: dict[str, Any] = {
        "vertices": vertices,
        "elements_tet4": elements_tet4,
        "elements_tet10": elements_tet10,
        "is_surface_node": is_surface,
    }

    return result


def compute_surface_node_mask(
    vertices: np.ndarray,
    elements: np.ndarray,
) -> np.ndarray:
    """Identify surface (boundary) nodes from tet mesh connectivity.

    A face is a boundary face if it appears in exactly one tetrahedron.
    Nodes belonging to any boundary face are surface nodes.

    Parameters
    ----------
    vertices : np.ndarray
        Node coordinates, shape (N, 3).
    elements : np.ndarray
        Tet connectivity, shape (E, 4), 0-based.

    Returns
    -------
    np.ndarray
        Boolean mask, shape (N,). True for surface nodes.
    """
    n_nodes = len(vertices)
    is_surface = np.zeros(n_nodes, dtype=bool)

    # Each tet4 has 4 triangular faces
    # Face i is opposite to node i: face 0 = (1,2,3), face 1 = (0,2,3), etc.
    face_node_local = [
        (1, 2, 3),  # face opposite node 0
        (0, 2, 3),  # face opposite node 1
        (0, 1, 3),  # face opposite node 2
        (0, 1, 2),  # face opposite node 3
    ]

    # Collect all faces with sorted node IDs for deduplication
    face_counts: dict[tuple[int, ...], int] = {}

    for elem in elements:
        for local_face in face_node_local:
            face = tuple(sorted(int(elem[i]) for i in local_face))
            face_counts[face] = face_counts.get(face, 0) + 1

    # Boundary faces appear exactly once
    for face, count in face_counts.items():
        if count == 1:
            for node_id in face:
                is_surface[node_id] = True

    return is_surface
