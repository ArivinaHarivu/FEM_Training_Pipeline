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

``elem_conn`` is now the **tet10** connectivity (E, 10), read from the
H5 file's ``Facets_tet10`` dataset, not the tet4 corner-only ``Facets``.
This is required by ``physics_bridge.py``'s nodal-strain evaluation,
which needs each element's 6 mid-edge nodes as well as its 4 corners —
using tet4 connectivity would silently drop the mid-edge nodes out of
the graph (no edges reaching them) and out of the physics bridge (no
per-node strain estimate at them). Mesh edges are built from the tet10
connectivity too, so every node — corner or mid-edge — has message-
passing edges reaching it.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Iterator, Optional
import random

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler
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
# ResumableBatchSampler — deterministic shuffling with intra-epoch resumption
# ---------------------------------------------------------------------------
class ResumableBatchSampler(Sampler[list[int]]):
    """Batch sampler supporting deterministic epoch shuffling and intra-epoch resumption.

    When resuming training mid-epoch at batch index `start_batch`, this sampler
    slices the batch indices directly so that the underlying DataLoader does not
    need to read or process completed samples from disk.

    Parameters
    ----------
    dataset_len : int
        Total number of samples in the dataset.
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        Whether to shuffle sample indices every epoch. Default True.
    seed : int
        Base seed for deterministic per-epoch permutations. Default 42.
    drop_last : bool
        Whether to drop the final incomplete batch. Default False.
    """

    def __init__(
        self,
        dataset_len: int,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        self.dataset_len = dataset_len
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 1
        self.start_batch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch for deterministic permutation generation."""
        self.epoch = epoch

    def set_start_batch(self, start_batch: int) -> None:
        """Set the batch offset for resuming mid-epoch."""
        self.start_batch = max(0, start_batch)

    def __iter__(self) -> Iterator[list[int]]:
        if self.dataset_len == 0:
            return

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.dataset_len, generator=g).tolist()
        else:
            indices = list(range(self.dataset_len))

        # Build complete batch list
        batches: list[list[int]] = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i : i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)

        # Slice remaining batches if resuming mid-epoch
        if self.start_batch > 0:
            batches = batches[self.start_batch :]

        for b in batches:
            yield b

    def __len__(self) -> int:
        """Return the number of batches remaining in the current epoch."""
        total = self.total_batches()
        if self.start_batch > 0:
            return max(0, total - self.start_batch)
        return total

    def total_batches(self) -> int:
        """Return the total number of batches in a full, un-sliced epoch."""
        if self.dataset_len == 0:
            return 0
        if self.drop_last:
            return self.dataset_len // self.batch_size
        return (self.dataset_len + self.batch_size - 1) // self.batch_size

    def state_dict(self) -> dict[str, Any]:
        """Serialize sampler state for checkpointing."""
        return {
            "epoch": self.epoch,
            "start_batch": self.start_batch,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "batch_size": self.batch_size,
            "drop_last": self.drop_last,
            "dataset_len": self.dataset_len,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore sampler state from checkpoint."""
        self.epoch = state.get("epoch", self.epoch)
        self.start_batch = state.get("start_batch", self.start_batch)
        self.seed = state.get("seed", self.seed)
        self.shuffle = state.get("shuffle", self.shuffle)
        self.batch_size = state.get("batch_size", self.batch_size)
        self.drop_last = state.get("drop_last", self.drop_last)



# ---------------------------------------------------------------------------
# H5 → MeshData conversion (self-contained, no external parser dependency)
# ---------------------------------------------------------------------------

def _extract_edges_from_tetra(connectivity: np.ndarray) -> np.ndarray:
    """[Legacy, tet4-only] Extract bidirectional edge_index from tetrahedral connectivity.

    Not used by ``h5_to_meshdata`` anymore — see ``_extract_edges_from_tet10``.
    Kept only in case some other caller still passes tet4 (corner-only,
    4-node) connectivity.

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


def _extract_edges_from_tet10(connectivity: np.ndarray) -> np.ndarray:
    """Extract unique bidirectional edge_index from tet10 connectivity.

    Node ordering (VTK_QUADRATIC_TETRA / meshio "tetra10", matching
    ``physics_bridge.py``): local indices 0-3 are corners, 4-9 are the
    mid-edge nodes of edges (0,1), (1,2), (0,2), (0,3), (1,3), (2,3)
    respectively.

    Each parent tet edge is represented as *two* graph edges
    (corner -> mid-edge-node, mid-edge-node -> corner), giving 12 undirected
    segments per element (6 parent edges x 2).

    Parameters
    ----------
    connectivity : np.ndarray
        Shape ``(E_elem, 10)`` — tet10 node indices.

    Returns
    -------
    np.ndarray
        Shape ``(2, num_edges)`` — bidirectional, deduplicated.
    """
    if connectivity.shape[0] == 0:
        return np.zeros((2, 0), dtype=np.int64)

    # 12 directed segments per tet
    pairs = np.array([
        [0, 4], [4, 1],
        [1, 5], [5, 2],
        [0, 6], [6, 2],
        [0, 7], [7, 3],
        [1, 8], [8, 3],
        [2, 9], [9, 3],
    ], dtype=np.int64)

    raw_edges = connectivity[:, pairs].reshape(-1, 2)
    sorted_edges = np.sort(raw_edges, axis=1)
    unique_edges = np.unique(sorted_edges, axis=0)
    bidir = np.concatenate([unique_edges, unique_edges[:, ::-1]], axis=0)
    return bidir.T


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


def stress_to_strain(
    stress_voigt: np.ndarray, E: float, nu: float,
) -> np.ndarray:
    """Invert isotropic Hooke's law: ground-truth Voigt stress -> strain.

    Matches the engineering-shear convention in physics_bridge.py's
    ``hookes_law_stress`` (its forward direction: eps -> sigma). This
    is its algebraic inverse, so loss.py's ``L_eps``/``L_eps_corr``
    terms compare against a real target instead of never firing
    because no ``"eps"`` key exists in ``targets``.

    Parameters
    ----------
    stress_voigt : np.ndarray
        ``(N, 6)`` Voigt stress [xx, yy, zz, xy, yz, xz].
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.

    Returns
    -------
    np.ndarray
        ``(N, 6)`` Voigt engineering strain, same component order.
    """
    sxx, syy, szz = stress_voigt[:, 0], stress_voigt[:, 1], stress_voigt[:, 2]
    sxy, syz, sxz = stress_voigt[:, 3], stress_voigt[:, 4], stress_voigt[:, 5]

    exx = (sxx - nu * (syy + szz)) / E
    eyy = (syy - nu * (sxx + szz)) / E
    ezz = (szz - nu * (sxx + syy)) / E

    # sigma_shear = mu * gamma_shear (engineering shear)  =>  gamma = sigma / mu
    mu = E / (2 * (1 + nu))
    gxy = sxy / mu
    gyz = syz / mu
    gxz = sxz / mu

    return np.column_stack([exx, eyy, ezz, gxy, gyz, gxz])


def h5_to_meshdata(
    file_path: str | Path,
    dtype: str = "float32",
    E: float | None = None,
    nu: float | None = None,
) -> MeshData:
    """Convert a single Training Pipeline HDF5 file to a PyG MeshData.

    This is self-contained — no dependency on the GNN project's parsers.

    Parameters
    ----------
    file_path : str or Path
        Path to the ``.h5`` file.
    dtype : str
        Torch float dtype. Default ``"float32"``.
    E : float, optional
        Young's modulus (Pa). If provided along with ``nu``, ground-
        truth Voigt strain is derived from ``StressVoigt`` via inverse
        Hooke's law and attached as ``data.y_strain`` (N, 6) — needed
        for loss.py's ``L_eps``/``L_eps_corr`` terms to activate at
        all (they silently no-op without a ``"eps"`` target). Must
        match the material constants ``PhysicsBridge`` is constructed
        with, or the derived target and the model's internal
        assumption disagree.
    nu : float, optional
        Poisson's ratio. See ``E``.

    Returns
    -------
    MeshData
        PyG-compatible graph with 11-dim node features.
    """
    torch_dtype = getattr(torch, dtype)

    with h5py.File(str(file_path), "r") as f:
        vertices = np.array(f["Vertices"], dtype=np.float64)  # (N, 3)
        num_nodes = vertices.shape[0]

        if "Facets_tet10" in f:
            connectivity = np.array(f["Facets_tet10"], dtype=np.int64)  # (E, 10)
        else:
            connectivity = np.array(f["Facets"], dtype=np.int64)

        if connectivity.ndim != 2 or connectivity.shape[1] not in (4, 10):
            raise ValueError(
                f"{file_path}: 'Facets' has shape "
                f"{connectivity.shape}, expected (E, 10) or (E, 4)."
            )

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

    # --- Edge index (tet10: reaches corner + mid-edge nodes) ---
    if connectivity.shape[1] == 10:
        edge_index_np = _extract_edges_from_tet10(connectivity)
    else:
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
        if E is not None and nu is not None:
            strain = stress_to_strain(stress, E, nu)
            data.y_strain = torch.as_tensor(strain, dtype=torch_dtype)
    if von_mises is not None:
        data.y_von_mises = torch.as_tensor(von_mises, dtype=torch_dtype)

    return data


def sample_generator(seed: int, epoch: int, sample_id: int | str) -> np.random.Generator:
    """Generate a deterministic NumPy RNG generator for a specific sample and epoch.

    Uses NumPy's SeedSequence to prevent statistical correlation across adjacent
    (seed, epoch, sample_id) tuples, guaranteeing worker-independent and
    prefetch-tolerant data augmentation across multi-process DataLoaders.

    Parameters
    ----------
    seed : int
        Global base seed.
    epoch : int
        Current training epoch.
    sample_id : int or str
        Unique sample index or string identifier.

    Returns
    -------
    np.random.Generator
        Isolated deterministic RNG instance for this sample and epoch.
    """
    if isinstance(sample_id, str):
        import zlib
        sample_id_int = zlib.crc32(sample_id.encode("utf-8"))
    else:
        sample_id_int = int(sample_id)

    ss = np.random.SeedSequence([seed, epoch, sample_id_int])
    return np.random.default_rng(ss)


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
    E : float, optional
        Young's modulus override.
    nu : float, optional
        Poisson's ratio override.
    seed : int
        Base seed for deterministic per-sample generator. Default 42.
    """

    def __init__(
        self,
        h5_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
        split: str | None = None,
        file_list: list[str] | None = None,
        E: float | None = None,
        nu: float | None = None,
        seed: int = 42,
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

        self._E = E
        self._nu = nu
        self.seed = seed
        self.current_epoch: int = 1

        logger.info("FEMGraphDataset: %d files for split=%s", len(self._files), split)

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for deterministic per-sample RNG seeding."""
        self.current_epoch = max(1, epoch)

    @staticmethod
    def _files_from_manifest(
        manifest_path: Path,
        split: str,
        h5_dir: Path | None,
    ) -> list[Path]:
        """Filter manifest by split and resolve file paths."""
        df = pd.read_csv(manifest_path)

        if "split" not in df.columns:
            logger.warning(
                "Manifest %s missing 'split' column. Auto-assigning base-sample-safe 80/10/10 split...",
                manifest_path,
            )
            base_ids = df["base_sample_id"].unique() if "base_sample_id" in df.columns else df["sample_id"].unique()
            rng = np.random.default_rng(42)
            rng.shuffle(base_ids)
            n_train = max(1, int(0.8 * len(base_ids)))
            n_val = max(1, int(0.1 * len(base_ids)))
            train_bases = set(base_ids[:n_train])
            val_bases = set(base_ids[n_train:n_train + n_val])

            id_col = "base_sample_id" if "base_sample_id" in df.columns else "sample_id"
            df["split"] = df[id_col].apply(
                lambda b: "train" if b in train_bases else ("val" if b in val_bases else "test")
            )
            try:
                df.to_csv(manifest_path, index=False)
                logger.info("Saved 'split' column to %s: %s", manifest_path, df["split"].value_counts().to_dict())
            except Exception as e:
                logger.warning("Could not write 'split' to manifest file (%s); using in-memory split", e)

        split_df = df[df["split"] == split]

        # Each sample_id should have an H5 file
        sample_ids = split_df["sample_id"].unique()
        if h5_dir is not None:
            # Batch scan directory once instead of thousands of individual FUSE stat() calls
            existing_names = set(p.name for p in Path(h5_dir).glob("*.h5"))
            files = [
                Path(h5_dir) / f"{sid}.h5"
                for sid in sample_ids
                if f"{sid}.h5" in existing_names
            ]
        else:
            files = []

        return sorted(files)

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> MeshData:
        try:
            return h5_to_meshdata(self._files[idx], E=self._E, nu=self._nu)
        except Exception as e:
            logger.warning("Could not load sample %s (%s); loading fallback sample...", self._files[idx], e)
            fallback_idx = (idx + 1) % len(self._files)
            return h5_to_meshdata(self._files[fallback_idx], E=self._E, nu=self._nu)

    @property
    def file_paths(self) -> list[Path]:
        return list(self._files)


def create_dataloaders(
    h5_dir: str | Path,
    manifest_path: str | Path,
    batch_size: int = 4,
    num_workers: int = 0,
    E: float | None = None,
    nu: float | None = None,
    seed: int = 42,
    use_resumable_sampler: bool = True,
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
    E : float, optional
        Young's modulus override.
    nu : float, optional
        Poisson's ratio override.
    seed : int
        Seed for deterministic training batch sampling. Default 42.
    use_resumable_sampler : bool
        Whether to use ResumableBatchSampler for the training split to
        support zero-I/O intra-epoch batch resumption. Default True.

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
            E=E,
            nu=nu,
            seed=seed,
        )
        if len(ds) == 0:
            logger.warning("No files for split '%s'", split)
            continue

        if split == "train" and use_resumable_sampler:
            sampler = ResumableBatchSampler(
                dataset_len=len(ds),
                batch_size=batch_size,
                shuffle=True,
                seed=seed,
                drop_last=False,
            )
            loaders["train"] = PyGDataLoader(
                ds,
                batch_sampler=sampler,
                num_workers=num_workers,
            )
        else:
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
    max_samples: int = 50,
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
    u_vals, sigma_vals, strain_vals, vm_vals = [], [], [], []

    n = min(len(dataset), max_samples)
    for i in range(n):
        data = dataset[i]
        if hasattr(data, "y_displacement"):
            u_vals.append(data.y_displacement.numpy().ravel())
        if hasattr(data, "y_stress"):
            sigma_vals.append(data.y_stress.numpy().ravel())
        if hasattr(data, "y_strain"):
            strain_vals.append(data.y_strain.numpy().ravel())
        if hasattr(data, "y_von_mises"):
            vm_vals.append(data.y_von_mises.numpy().ravel())

    eps = 1e-8  # floor to prevent division by zero
    stds = {
        "u": max(float(np.std(np.concatenate(u_vals))), eps) if u_vals else 1.0,
        "sigma": max(float(np.std(np.concatenate(sigma_vals))), eps) if sigma_vals else 1.0,
        "eps": max(float(np.std(np.concatenate(strain_vals))), eps) if strain_vals else 1.0,
        "vm": max(float(np.std(np.concatenate(vm_vals))), eps) if vm_vals else 1.0,
    }

    logger.info("Field stds: %s", stds)
    return stds
