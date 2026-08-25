"""Loads — surface traction application.

Surface tractions only — point loads are banned (consistent with ADR-005).
Randomises load direction, magnitude, and application face.

Face name validation (H10): only axis-aligned faces matching the
pattern [xyz]_(min|max) are accepted for load/BC application. This
prevents crashes if future geometry families return non-standard
surface tag names (e.g. "fillet_surface", "hole_1").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np

# Only accept face names matching axis-aligned planar faces (H10)
_VALID_FACE_PATTERN = re.compile(r"^[xyz]_(min|max)$")


@dataclass(frozen=True)
class LoadSpec:
    """Specification for applied surface traction.

    Attributes
    ----------
    load_face : str
        Name of the face carrying the traction (e.g. "x_max").
    direction : np.ndarray
        Unit direction vector for the traction, shape (3,).
    magnitude : float
        Traction magnitude [Pa] (force per unit area).
    loaded_node_mask : np.ndarray
        Boolean mask of shape (N,) — True for loaded nodes.
    nodal_forces : np.ndarray
        Per-node force vectors, shape (N, 3). Zero for unloaded nodes.
    """

    load_face: str
    direction: np.ndarray
    magnitude: float
    loaded_node_mask: np.ndarray
    nodal_forces: np.ndarray


def filter_valid_faces(surface_tags: dict[str, int]) -> dict[str, int]:
    """Filter surface tags to only axis-aligned planar faces (H10).

    Prevents crashes from non-standard face names that the
    coordinate-based BC/load detection cannot handle.

    Parameters
    ----------
    surface_tags : dict[str, int]
        All surface tags from geometry generation.

    Returns
    -------
    dict[str, int]
        Only tags matching ``[xyz]_(min|max)``.
    """
    return {
        name: tag for name, tag in surface_tags.items()
        if _VALID_FACE_PATTERN.match(name)
    }


def select_load_face(
    surface_tags: dict[str, int],
    rng: np.random.Generator,
) -> str:
    """Randomly select a face to apply surface traction.

    Only considers axis-aligned faces matching [xyz]_(min|max).

    Parameters
    ----------
    surface_tags : dict[str, int]
        Available face names → Gmsh tags.
    rng : np.random.Generator
        Seeded RNG.

    Returns
    -------
    str
        Selected face name.

    Raises
    ------
    ValueError
        If no valid faces are available.
    """
    valid = filter_valid_faces(surface_tags)
    if not valid:
        raise ValueError(
            f"No valid axis-aligned faces found in surface_tags: "
            f"{list(surface_tags.keys())}"
        )
    available = list(valid.keys())
    return str(rng.choice(available))


def sample_load_direction(rng: np.random.Generator) -> np.ndarray:
    """Sample a random 3D unit vector for the traction direction.

    Uses the uniform-on-sphere method (normalize Gaussian samples).

    Parameters
    ----------
    rng : np.random.Generator
        Seeded RNG.

    Returns
    -------
    np.ndarray
        Unit vector, shape (3,).
    """
    vec = rng.standard_normal(3)
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        # Degenerate case — default to x-direction
        return np.array([1.0, 0.0, 0.0])
    return vec / norm


def compute_loaded_node_mask(
    vertices: np.ndarray,
    face_name: str,
    tol_fraction: float = 0.02,
) -> np.ndarray:
    """Identify nodes on the load face.

    Parameters
    ----------
    vertices : np.ndarray
        Node coordinates, shape (N, 3).
    face_name : str
        Face name (e.g. "x_min", "y_max").
    tol_fraction : float
        Tolerance as fraction of the relevant dimension.

    Returns
    -------
    np.ndarray
        Boolean mask, shape (N,).
    """
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    dims = maxs - mins

    axis_map = {"x": 0, "y": 1, "z": 2}
    parts = face_name.split("_")
    axis = axis_map[parts[0]]
    side = parts[1]

    tol = tol_fraction * dims[axis]

    if side == "min":
        return vertices[:, axis] < (mins[axis] + tol)
    else:
        return vertices[:, axis] > (maxs[axis] - tol)


def compute_nodal_forces(
    vertices: np.ndarray,
    loaded_mask: np.ndarray,
    direction: np.ndarray,
    magnitude: float,
) -> np.ndarray:
    """Distribute surface traction as nodal forces.

    For a surface traction, the total force is magnitude × area.
    We distribute it equally among loaded nodes (consistent lumping).
    This is a simplification; FEniCS applies the traction correctly
    via the variational form. These nodal forces are for node-level
    feature recording only.

    Parameters
    ----------
    vertices : np.ndarray
        Node coordinates, shape (N, 3).
    loaded_mask : np.ndarray
        Boolean mask identifying loaded nodes.
    direction : np.ndarray
        Unit traction direction vector, shape (3,).
    magnitude : float
        Traction magnitude [Pa].

    Returns
    -------
    np.ndarray
        Per-node force vectors, shape (N, 3). Zero for unloaded nodes.
    """
    n_nodes = len(vertices)
    forces = np.zeros((n_nodes, 3), dtype=np.float64)

    n_loaded = int(loaded_mask.sum())
    if n_loaded == 0:
        return forces

    # Force per loaded node (equal distribution)
    force_per_node = direction * magnitude / n_loaded
    forces[loaded_mask] = force_per_node

    return forces


def apply_loads(
    vertices: np.ndarray,
    surface_tags: dict[str, int],
    reference_magnitude: float,
    rng: np.random.Generator,
) -> LoadSpec:
    """Configure and apply loads for a sample.

    Parameters
    ----------
    vertices : np.ndarray
        Node coordinates, shape (N, 3).
    surface_tags : dict[str, int]
        Available face names → Gmsh tags.
    reference_magnitude : float
        Reference traction magnitude [Pa] for the unit solve.
    rng : np.random.Generator
        Seeded RNG.

    Returns
    -------
    LoadSpec
        Complete load specification with nodal forces.
    """
    load_face = select_load_face(surface_tags, rng)
    direction = sample_load_direction(rng)
    loaded_mask = compute_loaded_node_mask(vertices, load_face)
    nodal_forces = compute_nodal_forces(
        vertices, loaded_mask, direction, reference_magnitude,
    )

    return LoadSpec(
        load_face=load_face,
        direction=direction,
        magnitude=reference_magnitude,
        loaded_node_mask=loaded_mask,
        nodal_forces=nodal_forces,
    )
