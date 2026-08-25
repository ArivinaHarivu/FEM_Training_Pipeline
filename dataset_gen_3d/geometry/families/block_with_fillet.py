"""Block with fillet — box with filleted/rounded edges.

Tests generalization to curved-boundary geometry (smooth curvature
instead of sharp corners). The fillet radius and number of filleted
edges are randomized.
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


class BlockWithFillet(GeometryFamily):
    """Block with filleted/rounded edges or corners."""

    @property
    def family_name(self) -> str:
        return "block_with_fillet"

    def sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        """Sample block-with-fillet parameters.

        Parameters
        ----------
        rng : np.random.Generator
            Seeded RNG.

        Returns
        -------
        dict[str, Any]
            Keys: block_aspect (3,), fillet_radius_frac, num_fillets,
            fillet_edge_indices.
        """
        cfg = self._config
        radius_range = cfg["fillet_radius_fraction"]
        n_fillet_range = cfg["num_fillets_range"]

        block_aspect = rng.uniform(0.5, 1.5, size=3)
        fillet_radius_frac = float(rng.uniform(radius_range[0], radius_range[1]))
        num_fillets = int(rng.integers(n_fillet_range[0], n_fillet_range[1] + 1))

        # A box has 12 edges; randomly select which ones to fillet
        all_edge_indices = list(range(12))
        rng.shuffle(all_edge_indices)
        fillet_edge_indices = sorted(all_edge_indices[:num_fillets])

        return {
            "block_aspect": block_aspect.tolist(),
            "fillet_radius_frac": fillet_radius_frac,
            "num_fillets": num_fillets,
            "fillet_edge_indices": fillet_edge_indices,
        }

    def generate(
        self,
        params: dict[str, Any],
        scale: float,
        mesh_size: float,
        output_path: pathlib.Path,
    ) -> GeometryResult:
        """Generate filleted block using Gmsh OCC kernel.

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
            Mesh with surface tags.
        """
        aspect = np.array(params["block_aspect"])
        dims = aspect * scale
        fillet_radius = params["fillet_radius_frac"] * float(np.min(dims))
        num_fillets = params["num_fillets"]
        edge_indices = params["fillet_edge_indices"]

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("block_with_fillet")

        try:
            occ = gmsh.model.occ

            block = occ.addBox(0, 0, 0, dims[0], dims[1], dims[2])
            occ.synchronize()

            # Get all edges of the block
            edges = gmsh.model.getEntities(dim=1)

            if edges and num_fillets > 0:
                # Select edges to fillet (clamp indices to available edges)
                available = len(edges)
                selected_edges = [
                    edges[idx % available][1]
                    for idx in edge_indices
                ]

                # Clamp fillet radius to avoid self-intersection
                max_radius = 0.45 * float(np.min(dims))
                actual_radius = min(fillet_radius, max_radius)

                # Local target element size tied directly to fillet radius:
                local_mesh_size = min(mesh_size, max(actual_radius / 3.0, mesh_size * 0.02))

                try:
                    occ.fillet(
                        [(3, block)],
                        selected_edges,
                        [actual_radius],
                    )
                except Exception:
                    # If fillet fails on some edges, try one at a time
                    for edge_tag in selected_edges:
                        try:
                            volumes = gmsh.model.getEntities(dim=3)
                            if volumes:
                                occ.fillet(
                                    [volumes[0]],
                                    [edge_tag],
                                    [actual_radius],
                                )
                        except Exception:
                            continue  # skip unfillettable edges

            occ.synchronize()

            # Tag surfaces
            surfaces = gmsh.model.getEntities(dim=2)
            surface_tags = self._tag_surfaces(dims, surfaces)

            # Adaptive Mesh Field Settings:
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", local_mesh_size * 0.5)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)

            if selected_edges:
                try:
                    dist_field = gmsh.model.mesh.field.add("Distance")
                    gmsh.model.mesh.field.setNumbers(
                        dist_field, "CurvesList", [float(t) for t in selected_edges],
                    )
                    gmsh.model.mesh.field.setNumber(dist_field, "Sampling", 100)

                    thresh_field = gmsh.model.mesh.field.add("Threshold")
                    gmsh.model.mesh.field.setNumber(thresh_field, "InField", dist_field)
                    gmsh.model.mesh.field.setNumber(thresh_field, "SizeMin", local_mesh_size)
                    gmsh.model.mesh.field.setNumber(thresh_field, "SizeMax", mesh_size)
                    gmsh.model.mesh.field.setNumber(thresh_field, "DistMin", actual_radius * 1.5)
                    gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", actual_radius * 5.0)

                    gmsh.model.mesh.field.setAsBackgroundMesh(thresh_field)
                except Exception:
                    pass

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
        """Identify block faces by bounding-box position.

        After filleting, some original planar faces may be subdivided.
        We identify the largest face near each expected position.

        Parameters
        ----------
        dims : np.ndarray
            Block dimensions [x, y, z].
        surfaces : list[tuple[int, int]]
            Gmsh surface entities.

        Returns
        -------
        dict[str, int]
            Face name → Gmsh surface tag mapping.
        """
        face_map: dict[str, int] = {}
        tol = float(np.min(dims)) * 0.05

        # Candidate assignments: face_name -> (tag, area)
        candidates: dict[str, list[tuple[int, float]]] = {
            "x_min": [], "x_max": [],
            "y_min": [], "y_max": [],
            "z_min": [], "z_max": [],
        }

        for dim, tag in surfaces:
            bbox = gmsh.model.getBoundingBox(dim, tag)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox
            area = (xmax - xmin) * (ymax - ymin) + \
                   (xmax - xmin) * (zmax - zmin) + \
                   (ymax - ymin) * (zmax - zmin)

            if abs(xmin) < tol and abs(xmax) < tol:
                candidates["x_min"].append((tag, area))
            elif abs(xmin - dims[0]) < tol and abs(xmax - dims[0]) < tol:
                candidates["x_max"].append((tag, area))
            elif abs(ymin) < tol and abs(ymax) < tol:
                candidates["y_min"].append((tag, area))
            elif abs(ymin - dims[1]) < tol and abs(ymax - dims[1]) < tol:
                candidates["y_max"].append((tag, area))
            elif abs(zmin) < tol and abs(zmax) < tol:
                candidates["z_min"].append((tag, area))
            elif abs(zmin - dims[2]) < tol and abs(zmax - dims[2]) < tol:
                candidates["z_max"].append((tag, area))

        # Pick largest candidate per face
        for name, cands in candidates.items():
            if cands:
                face_map[name] = max(cands, key=lambda x: x[1])[0]

        return face_map
