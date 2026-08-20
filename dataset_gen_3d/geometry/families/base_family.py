"""Abstract base class for geometry families.

Each family encapsulates one parametric 3D solid topology. The contract:
    1. `sample_params(rng)` → random shape parameters within configured ranges
    2. `generate(params, scale, mesh_size, output_path)` → .msh file path
    3. `governing_thickness(params)` → thickness for bending-mode linearity gate
       (returns None for bulk families)

Pattern: Strategy (GoF). The pipeline orchestrator selects a family via
FAMILY_REGISTRY and calls the same interface regardless of which concrete
family is chosen.
"""

from __future__ import annotations

import abc
import pathlib
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class GeometryResult:
    """Output of a successful geometry generation.

    Attributes
    ----------
    msh_path : pathlib.Path
        Path to the generated Gmsh .msh file.
    params : dict[str, Any]
        Shape parameters used (for manifest logging).
    node_count : int
        Number of nodes in the generated mesh.
    element_count : int
        Number of tetrahedral elements.
    characteristic_length : float
        Bounding-box characteristic length [m].
    governing_thickness : float or None
        For bending families: minimum cross-section dimension [m].
        For bulk families: None.
    surface_tags : dict[str, int]
        Gmsh physical surface tags for BC/load application.
        Keys: face identifiers (e.g. "x_min", "x_max", "y_min", ...).
    """

    msh_path: pathlib.Path
    params: dict[str, Any]
    node_count: int
    element_count: int
    characteristic_length: float
    governing_thickness: Optional[float]
    surface_tags: dict[str, int]


class GeometryFamily(abc.ABC):
    """Abstract base for geometry families.

    Parameters
    ----------
    config : dict[str, Any]
        Family-specific configuration block from config.yaml.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config

    @property
    @abc.abstractmethod
    def family_name(self) -> str:
        """Return the canonical family name (matches config.yaml key)."""

    @property
    def is_bending_family(self) -> bool:
        """Whether this family uses the thickness-relative linearity gate."""
        return False

    @abc.abstractmethod
    def sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        """Sample random shape parameters from the configured ranges.

        Parameters
        ----------
        rng : np.random.Generator
            Seeded random number generator for reproducibility.

        Returns
        -------
        dict[str, Any]
            Shape parameters sufficient to fully define the geometry.
        """

    @abc.abstractmethod
    def generate(
        self,
        params: dict[str, Any],
        scale: float,
        mesh_size: float,
        output_path: pathlib.Path,
    ) -> GeometryResult:
        """Generate the meshed 3D solid.

        Parameters
        ----------
        params : dict[str, Any]
            Shape parameters from ``sample_params``.
        scale : float
            Characteristic length [m] to scale the geometry to.
        mesh_size : float
            Target mesh element size [m].
        output_path : pathlib.Path
            Where to write the .msh file.

        Returns
        -------
        GeometryResult
            Result containing mesh path, metadata, and surface tags.

        Raises
        ------
        GeometryGenerationError
            If mesh generation fails (degenerate geometry, self-intersection, etc.).
        """

    def governing_thickness(self, params: dict[str, Any], scale: float) -> Optional[float]:
        """Compute the governing thickness for the linearity gate.

        For bulk families, returns None (the length-relative gate is used).
        Bending families must override this to return the actual cross-section
        or plate thickness.

        Parameters
        ----------
        params : dict[str, Any]
            Shape parameters.
        scale : float
            Characteristic length [m].

        Returns
        -------
        float or None
            Governing thickness in meters, or None for bulk families.
        """
        return None


class GeometryGenerationError(Exception):
    """Raised when geometry generation fails for a sample."""
