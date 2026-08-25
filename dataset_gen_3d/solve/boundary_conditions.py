"""Boundary conditions — configurable Dirichlet BC face selection.

Selects which face(s) of the geometry to fix (zero displacement).
Supports the multi-hot edge case where a fixed node can also carry load.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BoundaryConditionSpec:
    """Specification for Dirichlet boundary conditions.

    Attributes
    ----------
    fixed_face : str
        Name of the face to fix (e.g. "x_min", "y_max").
    fixed_node_mask : np.ndarray
        Boolean mask of shape (N,) — True for constrained nodes.
    """

    fixed_face: str
    fixed_node_mask: np.ndarray


def select_fixed_face(
    surface_tags: dict[str, int],
    load_face: str,
    rng: np.random.Generator,
) -> str:
    """Select a face to fix, avoiding the load face where possible.

    Only considers axis-aligned faces matching [xyz]_(min|max) (H10).
    If only one valid face is available (excluding load face), uses it.
    If no other face is available, fixes the load face (multi-hot case).

    Parameters
    ----------
    surface_tags : dict[str, int]
        Available face names → Gmsh tags.
    load_face : str
        Name of the face carrying the load.
    rng : np.random.Generator
        Seeded RNG.

    Returns
    -------
    str
        Name of the face to fix.

    Raises
    ------
    ValueError
        If no valid faces are available.
    """
    from dataset_gen_3d.solve.loads import filter_valid_faces

    valid = filter_valid_faces(surface_tags)
    if not valid:
        raise ValueError(
            f"No valid axis-aligned faces found in surface_tags: "
            f"{list(surface_tags.keys())}"
        )
    available = list(valid.keys())

    # Prefer fixing a face different from the load face
    non_load = [f for f in available if f != load_face]

    if non_load:
        return str(rng.choice(non_load))
    # Multi-hot edge case: fix the load face
    return load_face


def compute_fixed_node_mask(
    vertices: np.ndarray,
    face_name: str,
    dims: np.ndarray | None = None,
    tol_fraction: float = 0.02,
) -> np.ndarray:
    """Identify nodes on a face by coordinate position.

    Parameters
    ----------
    vertices : np.ndarray
        Node coordinates, shape (N, 3).
    face_name : str
        Face identifier: "x_min", "x_max", "y_min", "y_max",
        "z_min", "z_max".
    dims : np.ndarray or None
        Bounding-box dimensions [x, y, z]. If None, computed from vertices.
    tol_fraction : float
        Tolerance as fraction of the relevant dimension.

    Returns
    -------
    np.ndarray
        Boolean mask of shape (N,).
    """
    if dims is None:
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)
        dims = maxs - mins
    else:
        mins = vertices.min(axis=0)
        maxs = vertices.max(axis=0)

    axis_map = {"x": 0, "y": 1, "z": 2}
    parts = face_name.split("_")
    axis = axis_map[parts[0]]
    side = parts[1]  # "min" or "max"

    extent = dims[axis]
    tol = tol_fraction * extent

    if side == "min":
        return vertices[:, axis] < (mins[axis] + tol)
    else:
        return vertices[:, axis] > (maxs[axis] - tol)


def apply_boundary_conditions(
    vertices: np.ndarray,
    surface_tags: dict[str, int],
    load_face: str,
    rng: np.random.Generator,
) -> BoundaryConditionSpec:
    """Select and compute boundary conditions for a sample.

    Parameters
    ----------
    vertices : np.ndarray
        Node coordinates, shape (N, 3).
    surface_tags : dict[str, int]
        Available faces.
    load_face : str
        Face carrying the load.
    rng : np.random.Generator
        Seeded RNG.

    Returns
    -------
    BoundaryConditionSpec
        Fixed face name and node mask.
    """
    fixed_face = select_fixed_face(surface_tags, load_face, rng)
    mask = compute_fixed_node_mask(vertices, fixed_face)

    return BoundaryConditionSpec(
        fixed_face=fixed_face,
        fixed_node_mask=mask,
    )
