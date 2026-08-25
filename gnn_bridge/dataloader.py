"""FEM DataLoader — loads Training Pipeline HDF5 files into PyG MeshData batches.

This module bridges the FEM pipeline's HDF5 output to the GNN model's
expected input format. It:

1. Discovers HDF5 files from a directory or manifest CSV
2. Parses each file via ``fenicsx_h5_parser.py``
3. Converts to canonical ``Mesh`` via the GNN project's converter
4. Builds PyG ``MeshData`` graphs with 11-dim node features
5. Wraps everything in a PyG ``DataLoader`` for batched training

Node features (11-dim):
    [x, y, z, is_fixed, is_loaded, is_surface, hops_fixed, hops_load, Fx, Fy, Fz]

Edge features (4-dim):
    [dx, dy, dz, length]

Global features (1-dim):
    [total_load_magnitude]

Targets:
    y_displacement (N, 3), y_stress (N, 6), y_von_mises (N,)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
from collections import deque

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import PyG components. On Colab this may need pip install first.
# ---------------------------------------------------------------------------
try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader as PyGDataLoader
except ImportError:
    raise ImportError(
        "torch_geometric is required. Install with:\n"
        "  pip install torch torch_geometric"
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_HOP_CAP = 20  # Clamp BFS hops before normalising to [0, 1]


# ---------------------------------------------------------------------------
# MeshData — custom PyG Data with elem_conn batching
# ---------------------------------------------------------------------------
class MeshData(Data):
    """PyG Data subclass that correctly batches elem_conn."""

    def __inc__(self, key: str, value, *args, **kwargs) -> int:
        if key == "elem_conn":
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value, *args, **kwargs) -> int:
        if key == "elem_conn":
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


# ---------------------------------------------------------------------------
# H5 → MeshData conversion (self-contained, no external parser dependency)
# ---------------------------------------------------------------------------

def _extract_edges_from_tetra(connectivity: np.ndarray) -> np.ndarray:
    """Extract unique bidirectional edge_index from tetrahedral connectivity.

    Parameters
    ----------
    connectivity : np.ndarray
        Shape ``(E_elem, 4)`` — tet4 node indices.

    Returns
    -------
    np.ndarray
        Shape ``(2, num_edges)`` — bidirectional, deduplicated.
    """
    # 6 edges per tetrahedron: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
    edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    edges = set()
    for i, j in edge_pairs:
        for row in connectivity:
            a, b = int(row[i]), int(row[j])
            if a > b:
                a, b = b, a
            edges.add((a, b))

    if not edges:
        return np.zeros((2, 0), dtype=np.int64)

    arr = np.array(sorted(edges), dtype=np.int64)  # (num_unique, 2)
    # Make bidirectional
    bidir = np.concatenate([arr, arr[:, ::-1]], axis=0)  # (2*num_unique, 2)
    return bidir.T  # (2, num_edges)


def _bfs_distances(
    num_nodes: int,
    adj: list[list[int]],
    source_mask: np.ndarray,
) -> np.ndarray:
    """Multi-source BFS returning shortest hop distance from any source."""
    INF = num_nodes
    dist = np.full(num_nodes, INF, dtype=np.int32)
    queue: deque = deque()
    for node in np.where(source_mask)[0]:
        dist[node] = 0
        queue.append(node)
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if dist[v] == INF:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def _compute_hop_features(
    num_nodes: int,
    edge_index: np.ndarray,
    bc_mask: np.ndarray,
    load_mask: np.ndarray,
) -> np.ndarray:
    """Compute normalised BFS hop-distance features.

    Returns shape (num_nodes, 2): [hops_to_fixed, hops_to_load], normalised.
    """
    src, dst = edge_index[0], edge_index[1]
    adj: list[list[int]] = [[] for _ in range(num_nodes)]
    for u, v in zip(src.tolist(), dst.tolist()):
        adj[u].append(v)

    hops_bc = _bfs_distances(num_nodes, adj, bc_mask)
    hops_load = _bfs_distances(num_nodes, adj, load_mask)

    hops_bc_norm = np.clip(hops_bc, 0, _HOP_CAP).astype(np.float32) / _HOP_CAP
    hops_load_norm = np.clip(hops_load, 0, _HOP_CAP).astype(np.float32) / _HOP_CAP
    return np.column_stack([hops_bc_norm, hops_load_norm])


def h5_to_meshdata(file_path: str | Path, dtype: str = "float32") -> MeshData:
    """Convert a single Training Pipeline HDF5 file to a PyG MeshData.

    This is self-contained — no dependency on the GNN project's parsers.

    Parameters
    ----------
    file_path : str or Path
        Path to the ``.h5`` file.
    dtype : str
        Torch float dtype. Default ``"float32"``.

    Returns
    -------
    MeshData
        PyG-compatible graph with 11-dim node features.
    """
    torch_dtype = getattr(torch, dtype)

    with h5py.File(str(file_path), "r") as f:
        vertices = np.array(f["Vertices"], dtype=np.float64)  # (N, 3)
        connectivity = np.array(f["Facets"], dtype=np.int64)  # (E, 4)
        num_nodes = vertices.shape[0]

        # --- Boundary / load / surface flags ---
        is_fixed = (
            np.array(f["IsFixed"], dtype=np.float64)
            if "IsFixed" in f
            else np.zeros(num_nodes, dtype=np.float64)
        )
        is_loaded = (
            np.array(f["IsLoaded"], dtype=np.float64)
            if "IsLoaded" in f
            else np.zeros(num_nodes, dtype=np.float64)
        )
        is_surface = (
            np.array(f["IsSurfaceNode"], dtype=np.float64)
            if "IsSurfaceNode" in f
            else np.zeros(num_nodes, dtype=np.float64)
        )
        load_forces = (
            np.array(f["LoadForces"], dtype=np.float64)
            if "LoadForces" in f
            else np.zeros((num_nodes, 3), dtype=np.float64)
        )

        # --- Targets ---
        displacement = (
            np.array(f["u"], dtype=np.float64)
            if "u" in f
            else None
        )
        stress = (
            np.array(f["StressVoigt"], dtype=np.float64)
            if "StressVoigt" in f
            else None
        )
        von_mises = (
            np.array(f["VonMises"], dtype=np.float64).squeeze()
            if "VonMises" in f
            else None
        )

        # --- Total load magnitude (global feature) ---
        if "LoadForces" in f:
            total_load_mag = float(np.linalg.norm(load_forces.sum(axis=0)))
        else:
            total_load_mag = 0.0

    # --- Edge index ---
    edge_index_np = _extract_edges_from_tetra(connectivity)

    # --- Hop features ---
    bc_mask = is_fixed.astype(bool)
    load_mask = is_loaded.astype(bool)
    hop_features = _compute_hop_features(num_nodes, edge_index_np, bc_mask, load_mask)

    # --- Node features (11-dim) ---
    # [x, y, z, is_fixed, is_loaded, is_surface, hops_fixed, hops_load, Fx, Fy, Fz]
    node_features = np.column_stack([
        vertices,           # (N, 3)
        is_fixed.reshape(-1, 1),   # (N, 1)
        is_loaded.reshape(-1, 1),  # (N, 1)
        is_surface.reshape(-1, 1), # (N, 1)
        hop_features,       # (N, 2)
        load_forces,        # (N, 3)
    ])  # Total: (N, 11)

    # --- Edge features (4-dim): [dx, dy, dz, length] ---
    src_nodes = edge_index_np[0]
    dst_nodes = edge_index_np[1]
    relative_pos = vertices[dst_nodes] - vertices[src_nodes]  # (E, 3)
    lengths = np.linalg.norm(relative_pos, axis=1, keepdims=True)  # (E, 1)
    edge_features = np.hstack([relative_pos, lengths])  # (E, 4)

    # --- Build MeshData ---
    data = MeshData(
        x=torch.as_tensor(node_features, dtype=torch_dtype),
        edge_index=torch.as_tensor(edge_index_np, dtype=torch.long),
        edge_attr=torch.as_tensor(edge_features, dtype=torch_dtype),
        u=torch.tensor([[total_load_mag]], dtype=torch_dtype),  # (1, 1)
        elem_conn=torch.as_tensor(connectivity, dtype=torch.long),
        pos=torch.as_tensor(vertices, dtype=torch_dtype),
    )

    # --- Targets ---
    if displacement is not None:
        data.y_displacement = torch.as_tensor(displacement, dtype=torch_dtype)
    if stress is not None:
        data.y_stress = torch.as_tensor(stress, dtype=torch_dtype)
    if von_mises is not None:
        data.y_von_mises = torch.as_tensor(von_mises, dtype=torch_dtype)

    return data


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class FEMGraphDataset:
    """Lazy-loading dataset of FEM HDF5 files → PyG MeshData.

    Parameters
    ----------
    h5_dir : str or Path
        Directory containing ``.h5`` files.
    manifest_path : str or Path, optional
        If provided, only files listed in the manifest with a matching
        split are loaded. Requires ``split`` column.
    split : str, optional
        One of ``"train"``, ``"val"``, ``"test"``. Used to filter the
        manifest. Ignored if ``manifest_path`` is None.
    file_list : list[str], optional
        Explicit list of file paths. Overrides ``h5_dir`` discovery.
    """

    def __init__(
        self,
        h5_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
        split: str | None = None,
        file_list: list[str] | None = None,
    ):
        if file_list is not None:
            self._files = [Path(f) for f in file_list]
        elif manifest_path is not None and split is not None:
            self._files = self._files_from_manifest(
                Path(manifest_path), split, Path(h5_dir) if h5_dir else None,
            )
        elif h5_dir is not None:
            self._files = sorted(Path(h5_dir).glob("*.h5"))
        else:
            raise ValueError("Provide h5_dir, manifest_path+split, or file_list")

        logger.info("FEMGraphDataset: %d files for split=%s", len(self._files), split)

    @staticmethod
    def _files_from_manifest(
        manifest_path: Path,
        split: str,
        h5_dir: Path | None,
    ) -> list[Path]:
        """Filter manifest by split and resolve file paths."""
        df = pd.read_csv(manifest_path)

        if "split" not in df.columns:
            raise ValueError(
                "Manifest must have a 'split' column. Run split_strategy first."
            )

        split_df = df[df["split"] == split]

        # Each sample_id should have an H5 file
        sample_ids = split_df["sample_id"].unique()
        files = []
        for sid in sample_ids:
            if h5_dir is not None:
                h5_path = h5_dir / f"{sid}.h5"
                if h5_path.exists():
                    files.append(h5_path)
                else:
                    logger.warning("H5 file not found: %s", h5_path)

        return sorted(files)

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> MeshData:
        return h5_to_meshdata(self._files[idx])

    @property
    def file_paths(self) -> list[Path]:
        return list(self._files)


def create_dataloaders(
    h5_dir: str | Path,
    manifest_path: str | Path,
    batch_size: int = 4,
    num_workers: int = 0,
) -> dict[str, PyGDataLoader]:
    """Create train/val/test DataLoaders from manifest + H5 directory.

    Parameters
    ----------
    h5_dir : str or Path
        Directory containing all ``.h5`` files.
    manifest_path : str or Path
        Path to the manifest CSV (must have ``split`` column).
    batch_size : int
        Batch size for DataLoaders.
    num_workers : int
        Number of data-loading workers.

    Returns
    -------
    dict[str, DataLoader]
        Keys: ``"train"``, ``"val"``, ``"test"``.
    """
    loaders = {}
    for split in ("train", "val", "test"):
        ds = FEMGraphDataset(
            h5_dir=h5_dir,
            manifest_path=manifest_path,
            split=split,
        )
        if len(ds) == 0:
            logger.warning("No files for split '%s'", split)
            continue

        loaders[split] = PyGDataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            drop_last=False,
        )

    return loaders


# ---------------------------------------------------------------------------
# Field statistics (for loss normalisation)
# ---------------------------------------------------------------------------

def compute_field_stds(
    dataset: FEMGraphDataset,
    max_samples: int = 500,
) -> dict[str, float]:
    """Compute per-field standard deviations from training data.

    These are used by ``MeshGraphNetLoss`` for normalisation.

    Parameters
    ----------
    dataset : FEMGraphDataset
        Training dataset.
    max_samples : int
        Maximum number of samples to use for stats.

    Returns
    -------
    dict[str, float]
        Keys: ``"u"``, ``"sigma"``, ``"eps"``, ``"vm"``.
    """
    u_vals, sigma_vals, vm_vals = [], [], []

    n = min(len(dataset), max_samples)
    for i in range(n):
        data = dataset[i]
        if hasattr(data, "y_displacement"):
            u_vals.append(data.y_displacement.numpy().ravel())
        if hasattr(data, "y_stress"):
            sigma_vals.append(data.y_stress.numpy().ravel())
        if hasattr(data, "y_von_mises"):
            vm_vals.append(data.y_von_mises.numpy().ravel())

    eps = 1e-8  # floor to prevent division by zero
    stds = {
        "u": max(float(np.std(np.concatenate(u_vals))), eps) if u_vals else 1.0,
        "sigma": max(float(np.std(np.concatenate(sigma_vals))), eps) if sigma_vals else 1.0,
        "eps": 1.0,  # strain not directly in H5; derived by physics bridge
        "vm": max(float(np.std(np.concatenate(vm_vals))), eps) if vm_vals else 1.0,
    }

    logger.info("Field stds: %s", stds)
    return stds
