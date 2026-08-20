"""Elongated bar — high aspect-ratio bar for receptive field testing.

Aspect ratio length:cross-section >= 8:1 (configurable up to 15:1).
Optionally includes cylindrical through-holes.
This is a BENDING family — uses thickness-relative linearity gate.
Mandatory family: guarantees high graph-diameter samples.
"""

from __future__ import annotations

import pathlib
from typing import Any, Optional

import gmsh
import numpy as np

from dataset_gen_3d.geometry.families.base_family import (
    GeometryFamily,
    GeometryGenerationError,
    GeometryResult,
)


class ElongatedBar(GeometryFamily):
    """High aspect-ratio bar with optional holes."""

    @property
    def family_name(self) -> str:
        return "elongated_bar"

    @property
    def is_bending_family(self) -> bool:
        return True

    def sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        """Sample elongated bar parameters.

        Parameters
        ----------
        rng : np.random.Generator
            Seeded RNG.

        Returns
        -------
        dict[str, Any]
            Keys: aspect_ratio, cross_section_aspect, has_holes,
            hole_count, holes (if present).
        """
        cfg = self._config
        ar_min = cfg["min_aspect_ratio"]
        ar_max = cfg["max_aspect_ratio"]
        cs_range = cfg["cross_section_aspect"]
        hole_prob = cfg["hole_probability"]

        aspect_ratio = float(rng.uniform(ar_min, ar_max))
        cross_section_aspect = float(rng.uniform(cs_range[0], cs_range[1]))
        has_holes = bool(rng.random() < hole_prob)

        params: dict[str, Any] = {
            "aspect_ratio": aspect_ratio,
            "cross_section_aspect": cross_section_aspect,
            "has_holes": has_holes,
        }

        if has_holes:
            hole_count = int(rng.integers(1, 4))
            holes = []
            for i in range(hole_count):
                # Holes along the bar length (x-axis), through height (y-axis)
                position_frac = float(rng.uniform(0.15, 0.85))
                radius_frac = float(rng.uniform(0.08, 0.25))
                holes.append({
                    "position_frac": position_frac,
                    "radius_frac": radius_frac,
                })
            params["hole_count"] = hole_count
            params["holes"] = holes
        else:
            params["hole_count"] = 0
            params["holes"] = []

        return params

    def generate(
        self,
        params: dict[str, Any],
        scale: float,
        mesh_size: float,
        output_path: pathlib.Path,
    ) -> GeometryResult:
        """Generate elongated bar using Gmsh OCC kernel.

        The bar is oriented along the x-axis (length direction).
        Cross-section is in the y-z plane.

        Parameters
        ----------
        params : dict[str, Any]
            Shape parameters.
        scale : float
            Characteristic length [m] — used as the cross-section reference.
        mesh_size : float
            Target element size [m].
        output_path : pathlib.Path
            Output .msh path.

        Returns
        -------
        GeometryResult
            Mesh with surface tags and governing_thickness.
        """
        ar = params["aspect_ratio"]
        cs_aspect = params["cross_section_aspect"]

        # Cross-section dimensions from scale
        # scale is characteristic_length — for elongated bar, cross-section
        # is the reference, and length = ar × cross-section
        cs_height = scale / ar  # cross-section height
        cs_width = cs_height * cs_aspect
        length = scale  # total bar length = characteristic_length

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("elongated_bar")

        try:
            occ = gmsh.model.occ

            # Create bar
            block = occ.addBox(0, 0, 0, length, cs_height, cs_width)

            # Subtract holes if present
            if params["has_holes"]:
                hole_tags = []
                for hole in params["holes"]:
                    pos_frac = hole["position_frac"]
                    rad_frac = hole["radius_frac"]

                    cx = pos_frac * length
                    cy = cs_height / 2
                    radius = rad_frac * min(cs_height, cs_width)

                    # Hole through z-direction
                    cyl = occ.addCylinder(
                        cx, cy, -mesh_size,
                        0, 0, cs_width + 2 * mesh_size,
                        radius,
                    )
                    hole_tags.append((3, cyl))

                if hole_tags:
                    result, _ = occ.cut([(3, block)], hole_tags)
                    if not result:
                        raise GeometryGenerationError(
                            "Hole cut produced empty geometry"
                        )

            occ.synchronize()

            # Tag surfaces
            surfaces = gmsh.model.getEntities(dim=2)
            surface_tags = self._tag_surfaces(
                length, cs_height, cs_width, surfaces,
            )

            # Mesh
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.3)
            gmsh.model.mesh.generate(3)
            gmsh.model.mesh.setOrder(2)  # tet4 → tet10 (quadratic)

            node_tags, _, _ = gmsh.model.mesh.getNodes()
            elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=3)
            node_count = len(node_tags)
            element_count = sum(len(et) for et in elem_tags)

            if element_count == 0:
                raise GeometryGenerationError("Zero elements generated")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            gmsh.write(str(output_path))

            # Governing thickness: minimum net cross-section dimension
            gov_thickness = self._compute_governing_thickness(
                cs_height, cs_width, params,
            )

            return GeometryResult(
                msh_path=output_path,
                params=params,
                node_count=node_count,
                element_count=element_count,
                characteristic_length=scale,
                governing_thickness=gov_thickness,
                surface_tags=surface_tags,
            )

        except Exception as e:
            if not isinstance(e, GeometryGenerationError):
                raise GeometryGenerationError(f"Gmsh error: {e}") from e
            raise
        finally:
            gmsh.finalize()

    def governing_thickness(
        self, params: dict[str, Any], scale: float,
    ) -> Optional[float]:
        """Return the minimum net cross-section dimension.

        Parameters
        ----------
        params : dict[str, Any]
            Shape parameters.
        scale : float
            Characteristic length [m].

        Returns
        -------
        float
            Governing thickness for the linearity gate.
        """
        ar = params["aspect_ratio"]
        cs_aspect = params["cross_section_aspect"]
        cs_height = scale / ar
        cs_width = cs_height * cs_aspect
        return self._compute_governing_thickness(cs_height, cs_width, params)

    def _compute_governing_thickness(
        self,
        cs_height: float,
        cs_width: float,
        params: dict[str, Any],
    ) -> float:
        """Compute minimum net cross-section accounting for holes.

        Parameters
        ----------
        cs_height : float
            Cross-section height [m].
        cs_width : float
            Cross-section width [m].
        params : dict[str, Any]
            Shape parameters (may contain holes).

        Returns
        -------
        float
            Net minimum cross-section dimension [m].
        """
        min_dim = min(cs_height, cs_width)

        if params["has_holes"]:
            # Largest hole reduces the effective cross-section
            max_hole_radius = max(
                h["radius_frac"] * min(cs_height, cs_width)
                for h in params["holes"]
            )
            net_height = cs_height - 2 * max_hole_radius
            min_dim = min(min_dim, max(net_height, min_dim * 0.1))

        return min_dim

    def _tag_surfaces(
        self,
        length: float,
        height: float,
        width: float,
        surfaces: list[tuple[int, int]],
    ) -> dict[str, int]:
        """Identify bar faces by position."""
        face_map: dict[str, int] = {}
        tol = min(length, height, width) * 0.01

        for dim, tag in surfaces:
            bbox = gmsh.model.getBoundingBox(dim, tag)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox

            if abs(xmin) < tol and abs(xmax) < tol:
                face_map["x_min"] = tag  # left end
            elif abs(xmin - length) < tol and abs(xmax - length) < tol:
                face_map["x_max"] = tag  # right end
            elif abs(ymin) < tol and abs(ymax) < tol:
                face_map["y_min"] = tag
            elif abs(ymin - height) < tol and abs(ymax - height) < tol:
                face_map["y_max"] = tag
            elif abs(zmin) < tol and abs(zmax) < tol:
                face_map["z_min"] = tag
            elif abs(zmin - width) < tol and abs(zmax - width) < tol:
                face_map["z_max"] = tag

        return face_map
