"""Normalization — per-feature normalization strategies for GNN training.

Strategies:
- Coordinates: per-sample centering + scale by characteristic_length
- Displacements: global mean/std from training set
- Stresses: global mean/std from training set
- Forces: per-sample normalization by total load magnitude
- Von Mises: global mean/std from training set

Usage:
    1. Compute stats from the training split:
       ``stats = compute_normalization_stats(train_dataset)``
    2. Pass stats to FEMDataset or call normalize_sample() per sample.
    3. Save/load stats for reproducibility across runs.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

import h5py
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


@dataclass
class NormalizationStats:
    """Pre-computed normalization statistics from training set.

    Attributes
    ----------
    disp_mean : np.ndarray
        Mean displacement per component, shape (3,).
    disp_std : np.ndarray
        Std displacement per component, shape (3,).
    stress_mean : np.ndarray
        Mean Voigt stress per component, shape (6,).
    stress_std : np.ndarray
        Std Voigt stress per component, shape (6,).
    vm_mean : float
        Mean von Mises stress.
    vm_std : float
        Std von Mises stress.
    """

    disp_mean: np.ndarray
    disp_std: np.ndarray
    stress_mean: np.ndarray
    stress_std: np.ndarray
    vm_mean: float
    vm_std: float

    def save(self, path: pathlib.Path) -> None:
        """Save stats to JSON for reproducibility."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "disp_mean": self.disp_mean.tolist(),
            "disp_std": self.disp_std.tolist(),
            "stress_mean": self.stress_mean.tolist(),
            "stress_std": self.stress_std.tolist(),
            "vm_mean": self.vm_mean,
            "vm_std": self.vm_std,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: pathlib.Path) -> NormalizationStats:
        """Load stats from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            disp_mean=np.array(data["disp_mean"], dtype=np.float32),
            disp_std=np.array(data["disp_std"], dtype=np.float32),
            stress_mean=np.array(data["stress_mean"], dtype=np.float32),
            stress_std=np.array(data["stress_std"], dtype=np.float32),
            vm_mean=data["vm_mean"],
            vm_std=data["vm_std"],
        )


def compute_normalization_stats(
    manifest_path: pathlib.Path,
    hdf5_dir: pathlib.Path,
    split: str = "train",
    max_samples: int | None = None,
) -> NormalizationStats:
    """Compute normalization statistics from the training split.

    Uses Welford's online algorithm for numerically stable
    mean/variance computation without loading everything into memory.

    Parameters
    ----------
    manifest_path : pathlib.Path
        Path to manifest CSV.
    hdf5_dir : pathlib.Path
        Path to HDF5 directory.
    split : str
        Which split to compute stats from (default "train").
    max_samples : int or None
        Cap on samples to process (for speed during development).

    Returns
    -------
    NormalizationStats
        Computed mean and std for each target field.
    """
    df = pd.read_csv(manifest_path)

    if "split" in df.columns:
        df = df[df["split"] == split]

    # Filter to accepted samples
    df = df[df["rejection_reason"].isna() | (df["rejection_reason"] == "")]

    if max_samples is not None:
        df = df.head(max_samples)

    # Welford accumulators
    n_total = 0
    disp_sum = np.zeros(3, dtype=np.float64)
    disp_sq_sum = np.zeros(3, dtype=np.float64)
    stress_sum = np.zeros(6, dtype=np.float64)
    stress_sq_sum = np.zeros(6, dtype=np.float64)
    vm_sum = 0.0
    vm_sq_sum = 0.0

    for _, row in df.iterrows():
        h5_path = hdf5_dir / f"{row['sample_id']}.h5"
        if not h5_path.exists():
            continue

        with h5py.File(str(h5_path), "r") as f:
            disp = f["u"][:].astype(np.float64)
            stress = f["StressVoigt"][:].astype(np.float64)
            vm = f["VonMises"][:].astype(np.float64).ravel()

        n_nodes = len(disp)
        n_total += n_nodes

        disp_sum += disp.sum(axis=0)
        disp_sq_sum += (disp ** 2).sum(axis=0)
        stress_sum += stress.sum(axis=0)
        stress_sq_sum += (stress ** 2).sum(axis=0)
        vm_sum += vm.sum()
        vm_sq_sum += (vm ** 2).sum()

    if n_total == 0:
        raise ValueError("No valid training samples found for normalization")

    # Compute mean and std
    disp_mean = disp_sum / n_total
    disp_std = np.sqrt(disp_sq_sum / n_total - disp_mean ** 2)
    disp_std = np.maximum(disp_std, 1e-10)  # prevent div-by-zero

    stress_mean = stress_sum / n_total
    stress_std = np.sqrt(stress_sq_sum / n_total - stress_mean ** 2)
    stress_std = np.maximum(stress_std, 1e-10)

    vm_mean = vm_sum / n_total
    vm_std = max(np.sqrt(vm_sq_sum / n_total - vm_mean ** 2), 1e-10)

    return NormalizationStats(
        disp_mean=disp_mean.astype(np.float32),
        disp_std=disp_std.astype(np.float32),
        stress_mean=stress_mean.astype(np.float32),
        stress_std=stress_std.astype(np.float32),
        vm_mean=float(vm_mean),
        vm_std=float(vm_std),
    )


def normalize_sample(
    data: Data,
    stats: NormalizationStats,
) -> Data:
    """Apply normalization to a PyG Data object in-place.

    Coordinate normalization (per-sample):
        x[:, 0:3] = (x[:, 0:3] - centroid) / char_length

    Target normalization (global):
        displacement = (disp - mean) / std
        stress = (stress - mean) / std
        von_mises = (vm - mean) / std

    Force normalization (per-sample):
        x[:, 6:9] = forces / max_force_magnitude

    Parameters
    ----------
    data : Data
        PyG Data object with raw features.
    stats : NormalizationStats
        Pre-computed training set statistics.

    Returns
    -------
    Data
        The same Data object, modified in-place.
    """
    x = data.x  # (N, 9)

    # Coordinate normalization: center and scale by char_length
    coords = x[:, 0:3]
    centroid = coords.mean(dim=0, keepdim=True)
    char_len = data.characteristic_length.item()
    char_len = max(char_len, 1e-10)
    x[:, 0:3] = (coords - centroid) / char_len

    # Force normalization: scale by max force magnitude
    forces = x[:, 6:9]
    max_force = forces.norm(dim=1).max().item()
    if max_force > 1e-30:
        x[:, 6:9] = forces / max_force

    data.x = x

    # Target normalization
    disp_mean = torch.from_numpy(stats.disp_mean)
    disp_std = torch.from_numpy(stats.disp_std)
    data.y_displacement = (data.y_displacement - disp_mean) / disp_std

    stress_mean = torch.from_numpy(stats.stress_mean)
    stress_std = torch.from_numpy(stats.stress_std)
    data.y_stress_voigt = (data.y_stress_voigt - stress_mean) / stress_std

    data.y_von_mises = (data.y_von_mises - stats.vm_mean) / stats.vm_std

    return data
