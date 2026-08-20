"""Block with holes — rectangular box + 1–4 cylindrical through-holes.

Randomises hole count, position, radius, and axis orientation.
Holes need not all pass through the same face.
This is a bulk family (uses length-relative linearity gate).
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


class BlockWithHoles(GeometryFamily):
    """Rectangular block with cylindrical through-holes."""

    @property
    def family_name(self) -> str:
        return "block_with_holes"

    def sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        """Sample random shape parameters.

        Parameters
        ----------
        rng : np.random.Generator
            Seeded RNG.

        Returns
        -------
        dict[str, Any]
            Keys: base_dims (3,), hole_count, holes (list of dicts with
            position, radius, axis).
        """
        cfg = self._config
        dims_range = cfg["base_dims_range"]
        hole_count_range = cfg["hole_count_range"]
        radius_frac = cfg["hole_radius_fraction"]

        # Base block dimensions as multipliers on unit cube
        dims = rng.uniform(dims_range[0], dims_range[1], size=3)

        hole_count = int(rng.integers(hole_count_range[0], hole_count_range[1] + 1))
        min_face = float(np.min(dims[:2]))  # smallest face dimension

        holes = []
        for _ in range(hole_count):
            # Hole axis: 0=x, 1=y, 2=z
            axis = int(rng.integers(0, 3))

            # Radius as fraction of smallest face dimension
            radius = float(rng.uniform(radius_frac[0], radius_frac[1])) * min_face

            # Position of hole center on the face perpendicular to axis
            # Keep hole center away from edges by at least 1.5× radius
            perp_axes = [a for a in range(3) if a != axis]
            center = {}
            for pa in perp_axes:
                margin = 1.5 * radius / dims[pa]
                margin = min(margin, 0.4)  # don't let margin exceed 40% of dim
                center[pa] = float(rng.uniform(margin, 1.0 - margin))

            holes.append({
                "axis": axis,
                "radius_norm": radius / min_face,
                "center_fractions": center,
            })

        return {
            "base_dims": dims.tolist(),
            "hole_count": hole_count,
            "holes": holes,
        }

    def generate(
        self,
        params: dict[str, Any],
        scale: float,
        mesh_size: float,
        output_path: pathlib.Path,
    ) -> GeometryResult:
        """Generate block with holes using Gmsh OCC kernel.

        Parameters
        ----------
        params : dict[str, Any]
            Shape parameters from ``sample_params``.
        scale : float
            Characteristic length [m].
        mesh_size : float
            Target mesh element size [m].
        output_path : pathlib.Path
            Where to write the .msh file.

        Returns
        -------
        GeometryResult
            Mesh result with surface tags for BC/load application.
        """
        dims = np.array(params["base_dims"]) * scale
        holes = params["holes"]

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("block_with_holes")

        try:
            occ = gmsh.model.occ

            # Create base block
            block = occ.addBox(0, 0, 0, dims[0], dims[1], dims[2])

            # Subtract holes
            hole_tags = []
            for hole in holes:
                axis = hole["axis"]
                radius = hole["radius_norm"] * float(np.min(dims[:2]))
                center_fracs = hole["center_fractions"]

                # Build hole center coordinates
                center = [0.0, 0.0, 0.0]
                perp_axes = [a for a in range(3) if a != axis]
                for pa in perp_axes:
                    center[pa] = center_fracs[pa] * dims[pa]

                # Cylinder along the hole axis, extending through the block
                length = dims[axis] + 2 * mesh_size  # extend past block faces
                start = list(center)
                start[axis] = -mesh_size

                # Direction vector
                dx, dy, dz = 0.0, 0.0, 0.0
                if axis == 0:
                    dx = length
                elif axis == 1:
                    dy = length
                else:
                    dz = length

                cyl = occ.addCylinder(
                    start[0], start[1], start[2],
                    dx, dy, dz,
                    radius,
                )
                hole_tags.append((3, cyl))

            # Boolean cut: block - holes
            if hole_tags:
                result, _ = occ.cut([(3, block)], hole_tags)
                if not result:
                    raise GeometryGenerationError(
                        "Boolean cut produced empty geometry"
                    )
            occ.synchronize()

            # Tag boundary surfaces for BC/load application
            surfaces = gmsh.model.getEntities(dim=2)
            surface_tags = self._tag_surfaces(dims, surfaces)

            # Set mesh size
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.3)

            # Generate 3D tetrahedral mesh
            gmsh.model.mesh.generate(3)
            gmsh.model.mesh.setOrder(2)  # tet4 → tet10 (quadratic)

            # Get mesh statistics
            node_tags, _, _ = gmsh.model.mesh.getNodes()
            elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=3)
            node_count = len(node_tags)
            element_count = sum(len(et) for et in elem_tags)

            if element_count == 0:
                raise GeometryGenerationError("Mesh generation produced zero elements")

            # Write mesh
            output_path.parent.mkdir(parents=True, exist_ok=True)
            gmsh.write(str(output_path))

            return GeometryResult(
                msh_path=output_path,
                params=params,
                node_count=node_count,
                element_count=element_count,
                characteristic_length=scale,
                governing_thickness=None,
                surface_tags=surface_tags,
            )

        except Exception as e:
            if not isinstance(e, GeometryGenerationError):
                raise GeometryGenerationError(f"Gmsh error: {e}") from e
            raise
        finally:
            gmsh.finalize()

    def _tag_surfaces(
        self,
        dims: np.ndarray,
        surfaces: list[tuple[int, int]],
    ) -> dict[str, int]:
        """Identify block faces by their bounding-box center position.

        Parameters
        ----------
        dims : np.ndarray
            Block dimensions [x, y, z] in meters.
        surfaces : list[tuple[int, int]]
            Gmsh surface entities.

        Returns
        -------
        dict[str, int]
            Mapping of face names to Gmsh surface tags.
        """
        face_map: dict[str, int] = {}
        tol = float(np.min(dims)) * 0.01

        for dim, tag in surfaces:
            bbox = gmsh.model.getBoundingBox(dim, tag)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            cz = (zmin + zmax) / 2

            # Identify planar block faces by checking if one coord is near 0 or dims
            if abs(xmin - 0) < tol and abs(xmax - 0) < tol:
                face_map["x_min"] = tag
            elif abs(xmin - dims[0]) < tol and abs(xmax - dims[0]) < tol:
                face_map["x_max"] = tag
            elif abs(ymin - 0) < tol and abs(ymax - 0) < tol:
                face_map["y_min"] = tag
            elif abs(ymin - dims[1]) < tol and abs(ymax - dims[1]) < tol:
                face_map["y_max"] = tag
            elif abs(zmin - 0) < tol and abs(zmax - 0) < tol:
                face_map["z_min"] = tag
            elif abs(zmin - dims[1]) < tol and abs(zmax - dims[2]) < tol:
                face_map["z_max"] = tag

        return face_map
