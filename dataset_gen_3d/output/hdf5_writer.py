"""HDF5 writer — per-sample output matching ADR-007 schema.

Each sample produces one .h5 file with the following datasets:

| Key                | Shape     | Dtype   | Description                         |
|--------------------|-----------|---------|-------------------------------------|
| Vertices           | (N, 3)   | float64 | Node coordinates                    |
| Facets             | (E, 4)   | int64   | Tet4 connectivity (corner nodes)    |
| Facets_tet10       | (E, 10)  | int64   | Tet10 connectivity (all nodes)      |
| IsSurfaceNode      | (N,)     | int64   | 1 = surface/boundary node           |
| IsFixed            | (N,)     | int64   | 1 = Dirichlet constrained           |
| Load_Class         | (1,)     | bytes   | Magnitude class string              |
| u                  | (N, 3)   | float64 | Displacement field                  |
| Strain             | (N, 6)   | float64 | Node-averaged Voigt strain          |
| StrainElem         | (E, 6)   | float64 | Element-wise DG0 Voigt strain       |
| StressVoigt        | (N, 6)   | float64 | Node-averaged Voigt stress          |
| StressVoigtElem    | (E, 6)   | float64 | Element-wise DG0 Voigt stress       |
| StressTensor       | (N, 3,3) | float64 | Node-averaged full stress tensor    |
| VonMises           | (N, 1)   | float64 | Node-averaged von Mises             |
| VonMisesElem       | (E,)     | float64 | Element-wise DG0 von Mises          |
| LoadForces         | (N, 3)   | float64 | Traction-consistent nodal forces    |
| IsLoaded           | (N,)     | int64   | Load flag per node                  |
| ReactionForces     | (N, 3)   | float64 | Reaction forces at fixed nodes      |

Voigt order: [xx, yy, zz, xy, yz, xz] — project-wide convention.
Engineering shear strain for off-diagonal components.
"""

from __future__ import annotations

import pathlib

import h5py
import numpy as np


def write_sample_hdf5(
    path: pathlib.Path,
    vertices: np.ndarray,
    elements_tet4: np.ndarray,
    displacement: np.ndarray,
    stress_voigt_nodal: np.ndarray,
    strain_voigt_nodal: np.ndarray,
    stress_voigt_elem: np.ndarray,
    strain_voigt_elem: np.ndarray,
    stress_tensor: np.ndarray,
    von_mises_nodal: np.ndarray,
    von_mises_elem: np.ndarray,
    is_fixed: np.ndarray,
    is_loaded: np.ndarray,
    is_surface_node: np.ndarray,
    nodal_forces: np.ndarray,
    load_class: str = "",
    reaction_forces: np.ndarray | None = None,
    elements_tet10: np.ndarray | None = None,
) -> None:
    """Write a single sample to HDF5, matching ADR-007 schema.

    Stores both element-level (DG0) and node-averaged stress/strain
    fields. The DG0 fields are ground truth; the node-averaged fields
    are convenience for the GNN which operates on nodal features.

    Parameters
    ----------
    path : pathlib.Path
        Output .h5 file path.
    vertices : np.ndarray
        Node coordinates, shape (N, 3).
    elements_tet4 : np.ndarray
        Tet4 connectivity, shape (E, 4), 0-based.
    displacement : np.ndarray
        Displacement field, shape (N, 3).
    stress_voigt_nodal : np.ndarray
        Node-averaged Voigt stress, shape (N, 6).
    strain_voigt_nodal : np.ndarray
        Node-averaged Voigt strain, shape (N, 6).
    stress_voigt_elem : np.ndarray
        Element-wise DG0 Voigt stress, shape (E, 6).
    strain_voigt_elem : np.ndarray
        Element-wise DG0 Voigt strain, shape (E, 6).
    stress_tensor : np.ndarray
        Node-averaged full stress tensor, shape (N, 3, 3).
    von_mises_nodal : np.ndarray
        Node-averaged von Mises stress, shape (N,).
    von_mises_elem : np.ndarray
        Element-wise DG0 von Mises, shape (E,).
    is_fixed : np.ndarray
        Boolean/int mask of fixed nodes, shape (N,).
    is_loaded : np.ndarray
        Boolean/int mask of loaded nodes, shape (N,).
    is_surface_node : np.ndarray
        Boolean/int mask of surface (boundary) nodes, shape (N,).
    nodal_forces : np.ndarray
        Traction-consistent per-node force vectors, shape (N, 3).
    load_class : str
        Magnitude class label (e.g. "SF_1.5").
    reaction_forces : np.ndarray or None
        Reaction forces at constrained nodes, shape (N, 3).
    elements_tet10 : np.ndarray or None
        Tet10 connectivity, shape (E, 10), 0-based.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(path), "w") as f:
        # ── Geometry ──
        f.create_dataset("Vertices", data=vertices.astype(np.float64))
        f.create_dataset("Facets", data=elements_tet4.astype(np.int64))

        if elements_tet10 is not None:
            f.create_dataset(
                "Facets_tet10", data=elements_tet10.astype(np.int64),
            )

        # ── Node flags (real data, not dummy ones) ──
        f.create_dataset(
            "IsSurfaceNode", data=is_surface_node.astype(np.int64),
        )
        f.create_dataset("IsFixed", data=is_fixed.astype(np.int64))
        f.create_dataset("IsLoaded", data=is_loaded.astype(np.int64))

        # ── Load class ──
        f.create_dataset(
            "Load_Class",
            data=np.array([load_class.encode("utf-8")]),
        )

        # ── Displacement ──
        f.create_dataset("u", data=displacement.astype(np.float64))

        # ── Node-averaged fields (for GNN nodal targets) ──
        f.create_dataset(
            "Strain", data=strain_voigt_nodal.astype(np.float64),
        )
        f.create_dataset(
            "StressVoigt", data=stress_voigt_nodal.astype(np.float64),
        )
        f.create_dataset(
            "StressTensor", data=stress_tensor.astype(np.float64),
        )
        f.create_dataset(
            "VonMises",
            data=von_mises_nodal.reshape(-1, 1).astype(np.float64),
        )

        # ── Element-level DG0 fields (ground truth) ──
        f.create_dataset(
            "StrainElem", data=strain_voigt_elem.astype(np.float64),
        )
        f.create_dataset(
            "StressVoigtElem", data=stress_voigt_elem.astype(np.float64),
        )
        f.create_dataset(
            "VonMisesElem", data=von_mises_elem.astype(np.float64),
        )

        # ── Forces ──
        f.create_dataset("LoadForces", data=nodal_forces.astype(np.float64))

        if reaction_forces is not None:
            f.create_dataset(
                "ReactionForces",
                data=reaction_forces.astype(np.float64),
            )
