"""Sample specification — defines a base sample configuration.

A SampleSpec represents ONE base configuration: one geometry family +
shape params + scale + BC config + load direction/face. The load-level
sweep (producing 5 safety-factor variants) is NOT part of the spec —
it's a post-solve scaling step in generate_dataset.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SampleSpec:
    """Specification for a single base sample (one FEM solve).

    Attributes
    ----------
    sample_id : str
        Unique identifier for this base sample.
    family_name : str
        Geometry family name (e.g. "l_bracket").
    shape_params : dict[str, Any]
        Family-specific shape parameters.
    characteristic_length : float
        Sampled characteristic length [m].
    scale_bucket : str
        Scale bucket assignment ("small", "medium", "large").
    mesh_size : float
        Computed mesh element size [m].
    fixed_face : str
        Name of the fixed (Dirichlet BC) face.
    load_face : str
        Name of the loaded (Neumann BC) face.
    load_direction : list[float]
        Unit direction vector for surface traction [3].
    governing_thickness : float or None
        For bending families: minimum cross-section dimension [m].
    """

    sample_id: str
    family_name: str
    shape_params: dict[str, Any]
    characteristic_length: float
    scale_bucket: str
    mesh_size: float
    fixed_face: str = ""
    load_face: str = ""
    load_direction: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    governing_thickness: Optional[float] = None

    def to_manifest_row(self) -> dict[str, Any]:
        """Convert to a flat dict for manifest CSV.

        Returns
        -------
        dict[str, Any]
            Flat key-value pairs suitable for a CSV row.
        """
        row: dict[str, Any] = {
            "base_sample_id": self.sample_id,
            "geometry_family": self.family_name,
            "characteristic_length": self.characteristic_length,
            "scale_bucket": self.scale_bucket,
            "mesh_size": self.mesh_size,
            "fixed_face": self.fixed_face,
            "load_face": self.load_face,
            "load_dir_x": self.load_direction[0],
            "load_dir_y": self.load_direction[1],
            "load_dir_z": self.load_direction[2],
        }

        if self.governing_thickness is not None:
            row["governing_thickness"] = self.governing_thickness

        # L-bracket specific params
        if self.family_name == "l_bracket":
            row["radius_ratio"] = self.shape_params.get("radius_ratio", None)
            row["fillet_radius"] = self.shape_params.get("fillet_radius", None)

        return row
