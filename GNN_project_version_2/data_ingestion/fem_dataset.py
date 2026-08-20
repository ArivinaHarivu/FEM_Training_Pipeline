"""FEM dataset — PyTorch Geometric Dataset for FEM HDF5 samples.

Reads the manifest CSV to discover accepted samples, then lazily
loads each HDF5 file on demand and converts it to a PyG Data object.

Node features:
  [x, y, z, is_fixed, is_loaded, is_surface, load_fx, load_fy, load_fz]
  → shape (N, 9)

Targets:
  - displacement: (N, 3)
  - stress_voigt: (N, 6)
  - von_mises: (N, 1)

Graph-level attributes:
  - characteristic_length, geometry_family, scale_bucket, safety_factor

Supports both Experiment A (tet10 graph) and Experiment B (tet4 graph)
via the ``graph_mode`` parameter.
"""

from __future__ import annotations

import pathlib
from typing import Any, Literal

import h5py
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset

from GNN_project_version_2.data_ingestion.graph_builder import build_graph
from GNN_project_version_2.data_ingestion.normalization import (
    NormalizationStats,
    normalize_sample,
)


# Map geometry family names to integer codes for the GNN
FAMILY_ENCODING = {
    "block_with_holes": 0,
    "l_bracket": 1,
    "elongated_bar": 2,
    "plate_3d": 3,
    "block_with_fillet": 4,
}

BUCKET_ENCODING = {"small": 0, "medium": 1, "large": 2}


class FEMDataset(Dataset):
    """PyG Dataset wrapping the FEM HDF5 + manifest output.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory containing the manifest CSV and HDF5 dir.
    manifest_path : str or pathlib.Path
        Path to the manifest CSV file.
    hdf5_dir : str or pathlib.Path
        Path to the directory containing per-sample HDF5 files.
    split : str or None
        If set, filter to samples with this split label
        (``"train"``, ``"val"``, ``"test"``).
    graph_mode : str
        ``"tet4"`` for Experiment B (default) or ``"tet10"`` for
        Experiment A.
    norm_stats : NormalizationStats or None
        Pre-computed normalization statistics. If None, raw values
        are used (no normalization).
    accepted_only : bool
        If True (default), skip rejected samples.
    coherence_pass_only : bool
        If True (default), skip samples that failed coherence check.
    """

    def __init__(
        self,
        root: str | pathlib.Path,
        manifest_path: str | pathlib.Path,
        hdf5_dir: str | pathlib.Path,
        split: str | None = None,
        graph_mode: Literal["tet4", "tet10"] = "tet4",
        norm_stats: NormalizationStats | None = None,
        accepted_only: bool = True,
        coherence_pass_only: bool = True,
    ) -> None:
        self._manifest_path = pathlib.Path(manifest_path)
        self._hdf5_dir = pathlib.Path(hdf5_dir)
        self._graph_mode = graph_mode
        self._norm_stats = norm_stats

        # Load and filter manifest
        df = pd.read_csv(self._manifest_path)

        if accepted_only:
            df = df[df["rejection_reason"].isna() | (df["rejection_reason"] == "")]

        if coherence_pass_only and "coherence_pass" in df.columns:
            df = df[df["coherence_pass"] == True]

        if split and "split" in df.columns:
            df = df[df["split"] == split]

        self._manifest = df.reset_index(drop=True)

        super().__init__(str(root))

    def len(self) -> int:
        """Number of samples in this dataset partition."""
        return len(self._manifest)

    def get(self, idx: int) -> Data:
        """Load a single sample as a PyG Data object.

        Parameters
        ----------
        idx : int
            Sample index within the filtered manifest.

        Returns
        -------
        Data
            PyG Data with node features, edge index, and targets.
        """
        row = self._manifest.iloc[idx]
        sample_id = row["sample_id"]
        h5_path = self._hdf5_dir / f"{sample_id}.h5"

        with h5py.File(str(h5_path), "r") as f:
            vertices = f["Vertices"][:].astype(np.float32)
            elements_tet4 = f["Facets"][:].astype(np.int64)

            # Choose connectivity for graph
            if self._graph_mode == "tet10" and "Facets_tet10" in f:
                elements_graph = f["Facets_tet10"][:].astype(np.int64)
            else:
                elements_graph = elements_tet4

            # Node flags
            is_fixed = f["IsFixed"][:].astype(np.float32)
            is_loaded = f["IsLoaded"][:].astype(np.float32)
            is_surface = f["IsSurfaceNode"][:].astype(np.float32)

            # Forces
            load_forces = f["LoadForces"][:].astype(np.float32)

            # Targets
            displacement = f["u"][:].astype(np.float32)
            stress_voigt = f["StressVoigt"][:].astype(np.float32)
            von_mises = f["VonMises"][:].astype(np.float32)

        # Build node features: [x, y, z, is_fixed, is_loaded, is_surface, fx, fy, fz]
        node_features = np.column_stack([
            vertices,                   # (N, 3)
            is_fixed.reshape(-1, 1),    # (N, 1)
            is_loaded.reshape(-1, 1),   # (N, 1)
            is_surface.reshape(-1, 1),  # (N, 1)
            load_forces,                # (N, 3)
        ])  # → (N, 9)

        # Build graph
        edge_index, edge_attr = build_graph(
            vertices, elements_graph, mode=self._graph_mode,
        )

        # Graph-level attributes
        char_length = float(row.get("characteristic_length", 1.0))
        family_code = FAMILY_ENCODING.get(row.get("geometry_family", ""), -1)
        bucket_code = BUCKET_ENCODING.get(row.get("scale_bucket", ""), -1)
        safety_factor = float(row.get("target_safety_factor", 1.0))

        # Construct PyG Data
        data = Data(
            x=torch.from_numpy(node_features),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y_displacement=torch.from_numpy(displacement),
            y_stress_voigt=torch.from_numpy(stress_voigt),
            y_von_mises=torch.from_numpy(von_mises),
            characteristic_length=torch.tensor([char_length], dtype=torch.float32),
            family_code=torch.tensor([family_code], dtype=torch.long),
            bucket_code=torch.tensor([bucket_code], dtype=torch.long),
            safety_factor=torch.tensor([safety_factor], dtype=torch.float32),
            sample_id=sample_id,
        )

        # Normalize if stats provided
        if self._norm_stats is not None:
            data = normalize_sample(data, self._norm_stats)

        return data
