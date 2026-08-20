"""Split loader — manifest-driven DataLoader creation.

Reads the ``split`` column from the manifest to create train/val/test
DataLoader instances using PyTorch Geometric's native DataLoader,
which handles variable-size graphs via batched adjacency matrices.

Usage:
    loaders = create_split_loaders(
        manifest_path="output/manifest.csv",
        hdf5_dir="output/hdf5",
        batch_size=32,
        graph_mode="tet4",
    )
    for batch in loaders["train"]:
        pred = model(batch)
"""

from __future__ import annotations

import pathlib
from typing import Any, Literal

from torch_geometric.loader import DataLoader

from GNN_project_version_2.data_ingestion.fem_dataset import FEMDataset
from GNN_project_version_2.data_ingestion.normalization import (
    NormalizationStats,
    compute_normalization_stats,
)


def create_split_loaders(
    manifest_path: str | pathlib.Path,
    hdf5_dir: str | pathlib.Path,
    batch_size: int = 32,
    graph_mode: Literal["tet4", "tet10"] = "tet4",
    num_workers: int = 4,
    compute_norm: bool = True,
    norm_stats_path: str | pathlib.Path | None = None,
    norm_stats: NormalizationStats | None = None,
) -> dict[str, Any]:
    """Create train/val/test DataLoaders from a manifest CSV.

    Automatically computes normalization stats from training split
    unless pre-computed stats are provided.

    Parameters
    ----------
    manifest_path : str or Path
        Path to manifest CSV with ``split`` column.
    hdf5_dir : str or Path
        Directory containing per-sample HDF5 files.
    batch_size : int
        Batch size for DataLoader (default 32).
    graph_mode : str
        ``"tet4"`` or ``"tet10"`` for graph connectivity experiment.
    num_workers : int
        Number of DataLoader worker processes.
    compute_norm : bool
        Whether to compute normalization stats from training set.
    norm_stats_path : str or Path or None
        Path to save/load normalization stats JSON.
    norm_stats : NormalizationStats or None
        Pre-computed stats. Overrides compute_norm if provided.

    Returns
    -------
    dict[str, Any]
        Keys: ``"train"``, ``"val"``, ``"test"`` → DataLoader,
        plus ``"norm_stats"`` → NormalizationStats.
    """
    manifest_path = pathlib.Path(manifest_path)
    hdf5_dir = pathlib.Path(hdf5_dir)

    # Compute or load normalization stats
    if norm_stats is None and compute_norm:
        if norm_stats_path and pathlib.Path(norm_stats_path).exists():
            norm_stats = NormalizationStats.load(pathlib.Path(norm_stats_path))
        else:
            norm_stats = compute_normalization_stats(
                manifest_path, hdf5_dir, split="train",
            )
            if norm_stats_path:
                norm_stats.save(pathlib.Path(norm_stats_path))

    root = str(manifest_path.parent)

    loaders = {}
    for split in ("train", "val", "test"):
        dataset = FEMDataset(
            root=root,
            manifest_path=manifest_path,
            hdf5_dir=hdf5_dir,
            split=split,
            graph_mode=graph_mode,
            norm_stats=norm_stats,
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )

    loaders["norm_stats"] = norm_stats
    return loaders
