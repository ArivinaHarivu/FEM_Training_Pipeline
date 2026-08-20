"""Thin plate-like block — low height:width:length ratio.

Height << width and length, optionally with holes. Tests thin-section
behavior relevant to sheet-metal-like components.
This is a BENDING family — uses thickness-relative linearity gate.
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


class Plate3D(GeometryFamily):
    """Thin plate-like block with optional holes."""

    @property
    def family_name(self) -> str:
        return "plate_3d"

    @property
    def is_bending_family(self) -> bool:
        return True

    def sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        """Sample thin plate parameters.

        Parameters
        ----------
        rng : np.random.Generator
            Seeded RNG.

        Returns
        -------
        dict[str, Any]
            Keys: thickness_fraction, width_length_ratio, has_holes,
            holes (if present).
        """
        cfg = self._config
        t_frac_range = cfg["thickness_fraction"]
        wl_range = cfg["width_length_ratio"]
        hole_prob = cfg["hole_probability"]

        thickness_frac = float(rng.uniform(t_frac_range[0], t_frac_range[1]))
        width_length_ratio = float(rng.uniform(wl_range[0], wl_range[1]))
        has_holes = bool(rng.random() < hole_prob)

        params: dict[str, Any] = {
            "thickness_fraction": thickness_frac,
            "width_length_ratio": width_length_ratio,
            "has_holes": has_holes,
        }

        if has_holes:
            hole_count = int(rng.integers(1, 4))
            holes = []
            for _ in range(hole_count):
                holes.append({
                    "x_frac": float(rng.uniform(0.15, 0.85)),
                    "y_frac": float(rng.uniform(0.15, 0.85)),
                    "radius_frac": float(rng.uniform(0.03, 0.12)),
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
        """Generate thin plate using Gmsh OCC kernel.

        Plate oriented with length along x, width along y, thickness along z.

        Parameters
        ----------
        params : dict[str, Any]
            Shape parameters.
        scale : float
            Characteristic length [m].
        mesh_size : float
            Target element size [m].
        output_path : pathlib.Path
            Output .msh path.

        Returns
        -------
        GeometryResult
            Mesh with surface tags and governing_thickness.
        """
        t_frac = params["thickness_fraction"]
        wl_ratio = params["width_length_ratio"]

        length = scale
        width = scale * wl_ratio
        thickness = scale * t_frac  # height << length, width

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("plate_3d")

        try:
            occ = gmsh.model.occ

            block = occ.addBox(0, 0, 0, length, width, thickness)

            # Subtract holes (through thickness, z-direction)
            if params["has_holes"]:
                hole_tags = []
                for hole in params["holes"]:
                    cx = hole["x_frac"] * length
                    cy = hole["y_frac"] * width
                    radius = hole["radius_frac"] * min(length, width)

                    cyl = occ.addCylinder(
                        cx, cy, -mesh_size,
                        0, 0, thickness + 2 * mesh_size,
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

            # Ensure at least 2-3 elements through thickness
            thickness_mesh = min(mesh_size, thickness / 3)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", thickness_mesh * 0.5)

            # Tag surfaces
            surfaces = gmsh.model.getEntities(dim=2)
            surface_tags = self._tag_surfaces(
                length, width, thickness, surfaces,
            )

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

            return GeometryResult(
                msh_path=output_path,
                params=params,
                node_count=node_count,
                element_count=element_count,
                characteristic_length=scale,
                governing_thickness=thickness,
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
        """Return plate thickness — the governing dimension for bending.

        Parameters
        ----------
        params : dict[str, Any]
            Shape parameters.
        scale : float
            Characteristic length [m].

        Returns
        -------
        float
            Plate thickness in meters.
        """
        return scale * params["thickness_fraction"]

    def _tag_surfaces(
        self,
        length: float,
        width: float,
        thickness: float,
        surfaces: list[tuple[int, int]],
    ) -> dict[str, int]:
        """Identify plate faces by position."""
        face_map: dict[str, int] = {}
        tol = min(length, width, thickness) * 0.01

        for dim, tag in surfaces:
            bbox = gmsh.model.getBoundingBox(dim, tag)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox

            if abs(xmin) < tol and abs(xmax) < tol:
                face_map["x_min"] = tag
            elif abs(xmin - length) < tol and abs(xmax - length) < tol:
                face_map["x_max"] = tag
            elif abs(ymin) < tol and abs(ymax) < tol:
                face_map["y_min"] = tag
            elif abs(ymin - width) < tol and abs(ymax - width) < tol:
                face_map["y_max"] = tag
            elif abs(zmin) < tol and abs(zmax) < tol:
                face_map["z_min"] = tag  # bottom face
            elif abs(zmin - thickness) < tol and abs(zmax - thickness) < tol:
                face_map["z_max"] = tag  # top face

        return face_map
