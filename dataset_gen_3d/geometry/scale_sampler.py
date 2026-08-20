"""Scale sampler — samples characteristic length and assigns scale buckets.

Provides a wide, explicitly configured range of object sizes so the model
cannot overfit to one absolute size regime. Mesh size scales proportionally
with object size (not fixed) so relative resolution stays comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ScaleSample:
    """Result of sampling a scale for one base sample.

    Attributes
    ----------
    characteristic_length : float
        Sampled bounding-box characteristic length [m].
    scale_bucket : str
        Discrete bucket name ("small", "medium", "large").
    mesh_size : float
        Computed mesh element size [m] = mesh_size_fraction × characteristic_length.
    """

    characteristic_length: float
    scale_bucket: str
    mesh_size: float


class ScaleSampler:
    """Samples characteristic lengths and assigns scale buckets.

    Uses log-uniform sampling for better spread across orders of magnitude
    (20mm to 2m spans ~2 orders of magnitude).

    Parameters
    ----------
    config : dict[str, Any]
        The ``scale`` block from config.yaml.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._length_range = config["characteristic_length_range"]
        self._mesh_size_fraction = config["mesh_size_fraction"]
        self._bucket_names = config["buckets"]
        self._bucket_boundaries = config["bucket_boundaries"]

        if len(self._bucket_boundaries) != len(self._bucket_names) + 1:
            raise ValueError(
                f"bucket_boundaries must have {len(self._bucket_names) + 1} values, "
                f"got {len(self._bucket_boundaries)}"
            )

    def sample(self, rng: np.random.Generator) -> ScaleSample:
        """Sample a characteristic length and assign its scale bucket.

        Parameters
        ----------
        rng : np.random.Generator
            Seeded RNG for reproducibility.

        Returns
        -------
        ScaleSample
            Contains characteristic_length, scale_bucket, and mesh_size.
        """
        log_min = np.log(self._length_range[0])
        log_max = np.log(self._length_range[1])
        char_length = float(np.exp(rng.uniform(log_min, log_max)))

        bucket = self._assign_bucket(char_length)
        mesh_size = self._mesh_size_fraction * char_length

        return ScaleSample(
            characteristic_length=char_length,
            scale_bucket=bucket,
            mesh_size=mesh_size,
        )

    def sample_for_bucket(self, bucket: str, rng: np.random.Generator) -> ScaleSample:
        """Sample a characteristic length constrained to a specific bucket.

        Used by the stratum-aware sampler to fill under-represented buckets.

        Parameters
        ----------
        bucket : str
            Target bucket name (e.g. "small", "medium", "large").
        rng : np.random.Generator
            Seeded RNG.

        Returns
        -------
        ScaleSample
            A sample whose scale_bucket is guaranteed to match ``bucket``.
        """
        idx = self._bucket_names.index(bucket)
        low = self._bucket_boundaries[idx]
        high = self._bucket_boundaries[idx + 1]

        log_low = np.log(low)
        log_high = np.log(high)
        char_length = float(np.exp(rng.uniform(log_low, log_high)))

        mesh_size = self._mesh_size_fraction * char_length

        return ScaleSample(
            characteristic_length=char_length,
            scale_bucket=bucket,
            mesh_size=mesh_size,
        )

    def _assign_bucket(self, char_length: float) -> str:
        """Assign a characteristic length to its scale bucket.

        Parameters
        ----------
        char_length : float
            Characteristic length [m].

        Returns
        -------
        str
            Bucket name.
        """
        for i, name in enumerate(self._bucket_names):
            if self._bucket_boundaries[i] <= char_length < self._bucket_boundaries[i + 1]:
                return name
        # Edge case: exactly at the upper boundary → last bucket
        return self._bucket_names[-1]
