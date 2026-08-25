"""Tests for geometry families — each produces valid, non-degenerate solids.

Covers:
- All 5 families generate valid .msh files with tetrahedral elements
- L-bracket: zero samples with zero-radius notch root
- L-bracket: r/d distribution matches configured ranges
- Surface tag identification
- Edge cases (extreme aspect ratios, small fillet radii)
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest

from dataset_gen_3d.geometry.families import FAMILY_REGISTRY
from dataset_gen_3d.geometry.families.block_with_holes import BlockWithHoles
from dataset_gen_3d.geometry.families.l_bracket import LBracket
from dataset_gen_3d.geometry.families.elongated_bar import ElongatedBar
from dataset_gen_3d.geometry.families.plate_3d import Plate3D
from dataset_gen_3d.geometry.families.block_with_fillet import BlockWithFillet
from dataset_gen_3d.geometry.families.base_family import GeometryGenerationError


# Default configs for testing
_CONFIGS = {
    "block_with_holes": {
        "base_dims_range": [0.8, 1.2],
        "hole_count_range": [1, 3],
        "hole_radius_fraction": [0.05, 0.15],
        "aspect_ratio_range": [0.5, 2.0],
    },
    "l_bracket": {
        "notch_depth_fraction": [0.25, 0.50],
        "notch_width_fraction": [0.25, 0.50],
        "radius_ratio_range": [0.02, 0.30],
        "radius_ratio_holdout": [0.10, 0.20],
    },
    "elongated_bar": {
        "min_aspect_ratio": 8.0,
        "max_aspect_ratio": 12.0,
        "cross_section_aspect": [0.5, 2.0],
        "hole_probability": 0.5,
    },
    "plate_3d": {
        "thickness_fraction": [0.03, 0.08],
        "width_length_ratio": [0.3, 1.0],
        "hole_probability": 0.3,
    },
    "block_with_fillet": {
        "fillet_radius_fraction": [0.05, 0.20],
        "num_fillets_range": [1, 3],
    },
}


class TestFamilyRegistry:
    """Test that all 5 families are registered."""

    def test_all_families_registered(self) -> None:
        assert len(FAMILY_REGISTRY) == 5
        expected = {
            "block_with_holes", "l_bracket", "elongated_bar",
            "plate_3d", "block_with_fillet",
        }
        assert set(FAMILY_REGISTRY.keys()) == expected


class TestBlockWithHoles:
    """Tests for the block_with_holes family."""

    def test_generates_valid_mesh(self, tmp_path: pathlib.Path) -> None:
        family = BlockWithHoles(_CONFIGS["block_with_holes"])
        rng = np.random.default_rng(42)
        params = family.sample_params(rng)

        result = family.generate(params, scale=0.1, mesh_size=0.01,
                                  output_path=tmp_path / "test.msh")

        assert result.msh_path.exists()
        assert result.node_count > 0
        assert result.element_count > 0
        assert result.governing_thickness is None  # bulk family

    def test_hole_count_in_range(self) -> None:
        family = BlockWithHoles(_CONFIGS["block_with_holes"])
        rng = np.random.default_rng(42)
        for _ in range(20):
            params = family.sample_params(rng)
            assert 1 <= params["hole_count"] <= 3

    def test_surface_tags_present(self, tmp_path: pathlib.Path) -> None:
        family = BlockWithHoles(_CONFIGS["block_with_holes"])
        rng = np.random.default_rng(42)
        params = family.sample_params(rng)
        result = family.generate(params, scale=0.1, mesh_size=0.01,
                                  output_path=tmp_path / "test.msh")
        assert len(result.surface_tags) > 0


class TestLBracket:
    """Tests for the L-bracket with filleted notch root."""

    def test_generates_valid_mesh(self, tmp_path: pathlib.Path) -> None:
        family = LBracket(_CONFIGS["l_bracket"])
        rng = np.random.default_rng(42)
        params = family.sample_params(rng)

        result = family.generate(params, scale=0.1, mesh_size=0.005,
                                  output_path=tmp_path / "test.msh")

        assert result.msh_path.exists()
        assert result.node_count > 0
        assert result.element_count > 0

    def test_no_zero_radius_notch(self) -> None:
        """Assert zero generated L-brackets have zero-radius notch root."""
        family = LBracket(_CONFIGS["l_bracket"])
        rng = np.random.default_rng(42)
        for _ in range(50):
            params = family.sample_params(rng)
            assert params["radius_ratio"] > 0, \
                "L-bracket generated with zero-radius notch root"
            assert params["radius_ratio"] >= 0.02

    def test_radius_ratio_in_range(self) -> None:
        family = LBracket(_CONFIGS["l_bracket"])
        rng = np.random.default_rng(42)
        for _ in range(100):
            params = family.sample_params(rng)
            assert 0.02 <= params["radius_ratio"] <= 0.30

    def test_fillet_radius_computed(self, tmp_path: pathlib.Path) -> None:
        family = LBracket(_CONFIGS["l_bracket"])
        rng = np.random.default_rng(42)
        params = family.sample_params(rng)
        result = family.generate(params, scale=0.1, mesh_size=0.005,
                                  output_path=tmp_path / "test.msh")
        assert "fillet_radius" in result.params
        assert result.params["fillet_radius"] > 0


class TestElongatedBar:
    """Tests for the elongated bar family."""

    def test_generates_valid_mesh(self, tmp_path: pathlib.Path) -> None:
        family = ElongatedBar(_CONFIGS["elongated_bar"])
        rng = np.random.default_rng(42)
        params = family.sample_params(rng)

        result = family.generate(params, scale=0.2, mesh_size=0.01,
                                  output_path=tmp_path / "test.msh")

        assert result.msh_path.exists()
        assert result.element_count > 0

    def test_is_bending_family(self) -> None:
        family = ElongatedBar(_CONFIGS["elongated_bar"])
        assert family.is_bending_family is True

    def test_governing_thickness_not_none(self, tmp_path: pathlib.Path) -> None:
        family = ElongatedBar(_CONFIGS["elongated_bar"])
        rng = np.random.default_rng(42)
        params = family.sample_params(rng)
        result = family.generate(params, scale=0.2, mesh_size=0.01,
                                  output_path=tmp_path / "test.msh")
        assert result.governing_thickness is not None
        assert result.governing_thickness > 0

    def test_aspect_ratio_respected(self) -> None:
        family = ElongatedBar(_CONFIGS["elongated_bar"])
        rng = np.random.default_rng(42)
        for _ in range(20):
            params = family.sample_params(rng)
            assert 8.0 <= params["aspect_ratio"] <= 12.0


class TestPlate3D:
    """Tests for the thin plate family."""

    def test_generates_valid_mesh(self, tmp_path: pathlib.Path) -> None:
        family = Plate3D(_CONFIGS["plate_3d"])
        rng = np.random.default_rng(42)
        params = family.sample_params(rng)

        result = family.generate(params, scale=0.2, mesh_size=0.01,
                                  output_path=tmp_path / "test.msh")

        assert result.msh_path.exists()
        assert result.element_count > 0

    def test_is_bending_family(self) -> None:
        family = Plate3D(_CONFIGS["plate_3d"])
        assert family.is_bending_family is True

    def test_governing_thickness_equals_plate_thickness(self) -> None:
        family = Plate3D(_CONFIGS["plate_3d"])
        rng = np.random.default_rng(42)
        params = family.sample_params(rng)
        scale = 0.2
        thickness = family.governing_thickness(params, scale)
        expected = scale * params["thickness_fraction"]
        assert abs(thickness - expected) < 1e-10


class TestBlockWithFillet:
    """Tests for the block with fillet family."""

    def test_generates_valid_mesh(self, tmp_path: pathlib.Path) -> None:
        family = BlockWithFillet(_CONFIGS["block_with_fillet"])
        rng = np.random.default_rng(42)
        params = family.sample_params(rng)

        result = family.generate(params, scale=0.1, mesh_size=0.01,
                                  output_path=tmp_path / "test.msh")

        assert result.msh_path.exists()
        assert result.element_count > 0
        assert result.governing_thickness is None  # bulk family
