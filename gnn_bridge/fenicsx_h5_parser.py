"""FEniCSx H5 parser — reads HDF5 files produced by Training_Pipeline_version_2.

The pipeline produces HDF5 files with a different schema than the legacy
SFEM dataset. This parser handles the new format and passes through all
available fields (boundary conditions, loads, surface flags, displacement,
stress, von Mises) to the downstream Converter.

Expected HDF5 keys
-------------------
Vertices       (N, 3)     float64   Node coordinates
Facets         (E, 4)     int32     Tetrahedral connectivity (tet4, 0-based)
Facets_tet10   (E, 10)    int32     Tet10 connectivity (optional)
IsFixed        (N,)       int8      Per-node fixed boundary flag
IsLoaded       (N,)       int8      Per-node load flag
IsSurfaceNode  (N,)       int8      Per-node surface flag
LoadForces     (N, 3)     float64   Per-node applied force vector
u              (N, 3)     float64   Displacement (target)
StressVoigt    (N, 6)     float64   Voigt stress (target)
VonMises       (N, 1)     float64   Von Mises stress (target)
Load_Class     (1,)       bytes     Load class string (e.g. "SF_2.0")
ReactionForces (N, 3)     float64   Reaction forces (optional)
"""

import numpy as np
import h5py
import meshio

from .base_parser import BaseParser


# Mapping from Load_Class string → approximate total load magnitude (N).
# The pipeline uses safety-factor labels; these are rough magnitudes
# for the global feature. The exact per-node forces are in LoadForces.
_LOAD_CLASS_MAGNITUDES = {
    "SF_1.2": 200.0,
    "SF_1.5": 200.0,
    "SF_2.0": 200.0,
    "SF_3.0": 200.0,
    "SF_5.0": 200.0,
}


class FEniCSxH5Parser(BaseParser):
    """Parser for FEniCSx/Training Pipeline HDF5 mesh files.

    Reads the pipeline's HDF5 layout into a ``meshio.Mesh``, packing
    boundary conditions, loads, and FEA solution fields into
    ``point_data`` so the Converter can pick them up generically.
    """

    def parse(self, file_path: str) -> meshio.Mesh:
        """Parse a Training Pipeline .h5 file.

        Parameters
        ----------
        file_path : str
            Path to the .h5 file.

        Returns
        -------
        meshio.Mesh
            meshio-compatible object with geometry in ``points``/``cells``
            and physics data in ``point_data``/``field_data``.
        """
        with h5py.File(file_path, "r") as f:
            points = np.array(f["Vertices"], dtype=np.float64)
            connectivity = np.array(f["Facets"], dtype=np.int64)

            point_data = {}

            # --- Boundary conditions ---
            # IsFixed is a direct per-node flag (N,) → reshape to (N, 1)
            if "IsFixed" in f:
                is_fixed = np.array(f["IsFixed"], dtype=np.int64)
                point_data["boundary_conditions"] = is_fixed.reshape(-1, 1)

            # --- Load flags and forces ---
            if "IsLoaded" in f:
                is_loaded = np.array(f["IsLoaded"], dtype=np.float64)
                point_data["is_loaded"] = is_loaded

            if "LoadForces" in f:
                load_forces = np.array(f["LoadForces"], dtype=np.float64)
                point_data["loads"] = load_forces

            # --- Surface node flag ---
            if "IsSurfaceNode" in f:
                is_surface = np.array(f["IsSurfaceNode"], dtype=np.float64)
                point_data["is_surface"] = is_surface

            # --- Targets ---
            if "u" in f:
                point_data["displacement"] = np.array(
                    f["u"], dtype=np.float64
                )

            if "StressVoigt" in f:
                point_data["stress"] = np.array(
                    f["StressVoigt"], dtype=np.float64
                )

            if "VonMises" in f:
                point_data["von_mises"] = np.array(
                    f["VonMises"], dtype=np.float64
                ).squeeze()  # (N, 1) → (N,)

            if "ReactionForces" in f:
                point_data["reaction_force"] = np.array(
                    f["ReactionForces"], dtype=np.float64
                )

            # --- Load class → magnitude ---
            field_data = {}
            if "Load_Class" in f:
                raw = f["Load_Class"][0]
                load_class = (
                    raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                )
                field_data["load_class"] = load_class

            # Compute total load magnitude from per-node forces
            if "LoadForces" in f:
                forces = np.array(f["LoadForces"], dtype=np.float64)
                total_mag = float(np.linalg.norm(forces.sum(axis=0)))
                field_data["total_load_magnitude"] = np.array(
                    [total_mag], dtype=np.float64
                )
            elif "Load_Class" in f:
                magnitude = _LOAD_CLASS_MAGNITUDES.get(load_class, 200.0)
                field_data["total_load_magnitude"] = np.array(
                    [magnitude], dtype=np.float64
                )

        cells = [meshio.CellBlock("tetra", connectivity)]

        return meshio.Mesh(
            points=points,
            cells=cells,
            point_data=point_data,
            field_data=field_data,
        )
