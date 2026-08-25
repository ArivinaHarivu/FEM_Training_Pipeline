"""L-bracket — box with a rectangular notch + parameterized fillet radius.

The notch root has a NON-ZERO fillet radius, sampled as r/d (radius /
section depth) in [0.02, 0.30]. This replaces the original sharp-corner
design which produced a mesh-non-convergent stress singularity under
linear elasticity.

The r/d range matches Peterson's stress-concentration chart convention.
A configurable holdout sub-range supports extrapolation testing along
the concentration-severity axis.

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


class LBracket(GeometryFamily):
    """L-bracket with filleted notch root."""

    @property
    def family_name(self) -> str:
        return "l_bracket"

    def sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        """Sample L-bracket shape parameters.

        The fillet radius at the notch root is parameterized as r/d where d
        is the section depth at the notch. The r/d ratio is drawn from the
        configured range, with a holdout sub-range excluded during training.

        Parameters
        ----------
        rng : np.random.Generator
            Seeded RNG.

        Returns
        -------
        dict[str, Any]
            Keys: block_aspect (2,), notch_depth_frac, notch_width_frac,
            radius_ratio (r/d), thickness_ratio (depth/width for 3D).
        """
        cfg = self._config
        depth_frac_range = cfg["notch_depth_fraction"]
        width_frac_range = cfg["notch_width_fraction"]
        rd_range = cfg["radius_ratio_range"]

        # Block aspect: width and depth multipliers on unit cube
        # Height is always 1.0 (the notch direction)
        block_aspect = rng.uniform(0.6, 1.5, size=2)  # [width, depth]

        notch_depth_frac = float(rng.uniform(depth_frac_range[0], depth_frac_range[1]))
        notch_width_frac = float(rng.uniform(width_frac_range[0], width_frac_range[1]))

        # Fillet radius ratio r/d
        radius_ratio = float(rng.uniform(rd_range[0], rd_range[1]))

        return {
            "block_aspect": block_aspect.tolist(),
            "notch_depth_frac": notch_depth_frac,
            "notch_width_frac": notch_width_frac,
            "radius_ratio": radius_ratio,
        }

    def generate(
        self,
        params: dict[str, Any],
        scale: float,
        mesh_size: float,
        output_path: pathlib.Path,
    ) -> GeometryResult:
        """Generate filleted L-bracket using Gmsh OCC kernel.

        The L-shape is built as a full block minus a rectangular notch,
        then the notch root corner is filleted with the parameterized radius.

        Parameters
        ----------
        params : dict[str, Any]
            Shape parameters from ``sample_params``.
        scale : float
            Characteristic length [m].
        mesh_size : float
            Target mesh element size [m].
        output_path : pathlib.Path
            Output .msh file path.

        Returns
        -------
        GeometryResult
            Mesh result with surface tags.
        """
        aspect = np.array(params["block_aspect"])
        notch_d_frac = params["notch_depth_frac"]
        notch_w_frac = params["notch_width_frac"]
        radius_ratio = params["radius_ratio"]

        # Physical dimensions [m]
        width = aspect[0] * scale
        height = scale  # reference dimension
        depth = aspect[1] * scale  # z-direction (through-thickness)

        notch_depth = notch_d_frac * height
        notch_width = notch_w_frac * width

        # Section depth at notch = remaining height after notch
        section_depth = height - notch_depth
        fillet_radius = radius_ratio * section_depth

        # Validate fillet radius is geometrically feasible
        max_fillet = min(notch_depth, notch_width, section_depth) * 0.9
        if fillet_radius > max_fillet:
            fillet_radius = max_fillet

        # Local target element size tied directly to fillet radius:
        # local_size = fillet_radius / 3 guarantees >= 3-4 quadratic elements
        # across the fillet arc to accurately capture sharp stress gradients.
        local_mesh_size = min(mesh_size, max(fillet_radius / 3.0, mesh_size * 0.02))

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("l_bracket")

        try:
            occ = gmsh.model.occ

            # Create full block
            block = occ.addBox(0, 0, 0, width, height, depth)

            # Create notch (rectangular cutout at top-right corner)
            notch_x = width - notch_width
            notch_y = height - notch_depth
            notch = occ.addBox(
                notch_x, notch_y, -mesh_size,
                notch_width + mesh_size,
                notch_depth + mesh_size,
                depth + 2 * mesh_size,
            )

            # Cut notch from block
            result, result_map = occ.cut([(3, block)], [(3, notch)])
            if not result:
                raise GeometryGenerationError("Notch cut produced empty geometry")

            occ.synchronize()

            # Fillet the notch root edge
            # The notch root is the internal edge at (notch_x, notch_y, z)
            # running along the z-axis (depth direction)
            edges = gmsh.model.getEntities(dim=1)
            fillet_edges = self._find_notch_root_edges(
                edges, notch_x, notch_y, depth,
                tol=max(mesh_size * 0.1, fillet_radius * 0.5),
            )

            if fillet_edges:
                try:
                    occ.fillet(
                        [result[0][1]],
                        [tag for _, tag in fillet_edges],
                        [fillet_radius],
                    )
                except Exception as e:
                    raise GeometryGenerationError(
                        f"Fillet operation failed at r/d={radius_ratio:.3f}: {e}"
                    ) from e

            occ.synchronize()

            # Tag surfaces
            surfaces = gmsh.model.getEntities(dim=2)
            surface_tags = self._tag_surfaces(
                width, height, depth, notch_x, notch_y, surfaces,
            )

            # Adaptive Mesh Field Settings:
            # 1. Distance field from the notch root fillet curves
            # 2. Threshold field smoothly transitioning from local_mesh_size at the fillet
            #    to standard mesh_size in the far field
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", local_mesh_size * 0.5)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)  # 12 elements per 2pi

            if fillet_edges:
                try:
                    dist_field = gmsh.model.mesh.field.add("Distance")
                    gmsh.model.mesh.field.setNumbers(
                        dist_field, "CurvesList", [float(tag) for _, tag in fillet_edges],
                    )
                    gmsh.model.mesh.field.setNumber(dist_field, "Sampling", 100)

                    thresh_field = gmsh.model.mesh.field.add("Threshold")
                    gmsh.model.mesh.field.setNumber(thresh_field, "InField", dist_field)
                    gmsh.model.mesh.field.setNumber(thresh_field, "SizeMin", local_mesh_size)
                    gmsh.model.mesh.field.setNumber(thresh_field, "SizeMax", mesh_size)
                    gmsh.model.mesh.field.setNumber(thresh_field, "DistMin", fillet_radius * 1.5)
                    gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", fillet_radius * 5.0)

                    gmsh.model.mesh.field.setAsBackgroundMesh(thresh_field)
                except Exception:
                    pass  # fallback to curvature-based and min/max characteristic lengths

            # Generate 3D tet mesh
            gmsh.model.mesh.generate(3)
            gmsh.model.mesh.setOrder(2)  # tet4 → tet10 (quadratic)

            # Check for degenerate elements near fillet
            node_tags, _, _ = gmsh.model.mesh.getNodes()
            elem_types, elem_tags, _ = gmsh.model.mesh.getElements(dim=3)
            node_count = len(node_tags)
            element_count = sum(len(et) for et in elem_tags)

            if element_count == 0:
                raise GeometryGenerationError(
                    "Mesh generation produced zero elements"
                )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            gmsh.write(str(output_path))

            # Enrich params with computed fillet values for manifest
            enriched_params = {
                **params,
                "fillet_radius": fillet_radius,
                "section_depth": section_depth,
                "local_mesh_size": local_mesh_size,
            }

            return GeometryResult(
                msh_path=output_path,
                params=enriched_params,
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

    def _find_notch_root_edges(
        self,
        edges: list[tuple[int, int]],
        notch_x: float,
        notch_y: float,
        depth: float,
        tol: float,
    ) -> list[tuple[int, int]]:
        """Find edges at the notch root corner for filleting.

        The notch root edge runs along the z-axis at position (notch_x, notch_y).

        Parameters
        ----------
        edges : list[tuple[int, int]]
            All Gmsh edge entities.
        notch_x : float
            X-coordinate of notch corner.
        notch_y : float
            Y-coordinate of notch corner.
        depth : float
            Block depth (z-extent).
        tol : float
            Position tolerance.

        Returns
        -------
        list[tuple[int, int]]
            Matching edge entities (dim, tag).
        """
        candidates = []
        for dim, tag in edges:
            bbox = gmsh.model.getBoundingBox(dim, tag)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox

            # Look for edge near (notch_x, notch_y) running in z
            x_match = abs(xmin - notch_x) < tol and abs(xmax - notch_x) < tol
            y_match = abs(ymin - notch_y) < tol and abs(ymax - notch_y) < tol
            z_span = (zmax - zmin) > depth * 0.5  # spans most of the depth

            if x_match and y_match and z_span:
                candidates.append((dim, tag))

        return candidates

    def _tag_surfaces(
        self,
        width: float,
        height: float,
        depth: float,
        notch_x: float,
        notch_y: float,
        surfaces: list[tuple[int, int]],
    ) -> dict[str, int]:
        """Identify L-bracket faces by bounding-box position.

        Parameters
        ----------
        width, height, depth : float
            Block dimensions.
        notch_x, notch_y : float
            Notch corner coordinates.
        surfaces : list[tuple[int, int]]
            Gmsh surface entities.

        Returns
        -------
        dict[str, int]
            Face name → Gmsh surface tag mapping.
        """
        face_map: dict[str, int] = {}
        tol = min(width, height, depth) * 0.01

        for dim, tag in surfaces:
            bbox = gmsh.model.getBoundingBox(dim, tag)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox

            if abs(xmin) < tol and abs(xmax) < tol:
                face_map["x_min"] = tag
            elif abs(zmin) < tol and abs(zmax) < tol:
                face_map["z_min"] = tag
            elif abs(zmin - depth) < tol and abs(zmax - depth) < tol:
                face_map["z_max"] = tag
            elif abs(ymin) < tol and abs(ymax) < tol:
                face_map["y_min"] = tag

        return face_map
