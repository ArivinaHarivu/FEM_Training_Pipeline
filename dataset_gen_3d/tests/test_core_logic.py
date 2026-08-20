"""Tests for B-matrix (tet10), load scaling, coherence check, split strategy.

These tests run locally without FEniCS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dataset_gen_3d.validation.b_matrix_tetra import (
    compute_b_matrix_tet10,
    shape_functions_tet10,
    element_volume_tet10,
    compute_element_strain_averaged_tet10,
)
from dataset_gen_3d.solve.load_scaling import (
    compute_scale_factors,
    apply_scaling,
)
from dataset_gen_3d.solve.run_fem_3d import FEMResult
from dataset_gen_3d.pipeline.split_strategy import (
    assign_splits,
    SplitLeakageError,
)


class TestBMatrixTet10:
    """Tests for the 10-node tet B-matrix."""

    def _make_reference_tet10(self) -> np.ndarray:
        """Create a simple tet10 element with known geometry.

        Corner nodes form a unit tet, mid-edge nodes at edge midpoints.
        """
        corners = np.array([
            [0.0, 0.0, 0.0],  # node 0
            [1.0, 0.0, 0.0],  # node 1
            [0.0, 1.0, 0.0],  # node 2
            [0.0, 0.0, 1.0],  # node 3
        ])
        # Mid-edge nodes
        mid_edges = np.array([
            (corners[0] + corners[1]) / 2,  # edge 0-1 → node 4
            (corners[1] + corners[2]) / 2,  # edge 1-2 → node 5
            (corners[0] + corners[2]) / 2,  # edge 0-2 → node 6
            (corners[0] + corners[3]) / 2,  # edge 0-3 → node 7
            (corners[1] + corners[3]) / 2,  # edge 1-3 → node 8
            (corners[2] + corners[3]) / 2,  # edge 2-3 → node 9
        ])
        return np.vstack([corners, mid_edges])

    def test_shape_functions_sum_to_one(self) -> None:
        """Partition of unity: shape functions sum to 1 at any point."""
        for xi, eta, zeta in [(0.25, 0.25, 0.25), (0.1, 0.2, 0.3),
                               (0.0, 0.0, 0.0), (0.5, 0.0, 0.0)]:
            N = shape_functions_tet10(xi, eta, zeta)
            assert abs(np.sum(N) - 1.0) < 1e-12, \
                f"Partition of unity violated at ({xi}, {eta}, {zeta})"

    def test_b_matrix_shape(self) -> None:
        """B-matrix must be (6, 30) for a 10-node tet."""
        vertices = self._make_reference_tet10()
        B, det_J = compute_b_matrix_tet10(vertices, 0.25, 0.25, 0.25)
        assert B.shape == (6, 30)
        assert det_J != 0

    def test_rigid_body_produces_zero_strain(self) -> None:
        """Pure translation → zero strain everywhere."""
        vertices = self._make_reference_tet10()
        # Uniform translation
        disp = np.tile([0.5, -0.3, 0.7], (10, 1))
        strain = compute_element_strain_averaged_tet10(vertices, disp)
        assert np.allclose(strain, 0.0, atol=1e-12)

    def test_element_volume(self) -> None:
        """Unit tet10 volume should be 1/6."""
        vertices = self._make_reference_tet10()
        vol = element_volume_tet10(vertices)
        assert abs(vol - 1.0 / 6.0) < 1e-10


class TestLoadScaling:
    """Tests for load scaling — pure linear algebra."""

    def _make_mock_fem_result(self) -> FEMResult:
        """Create a synthetic FEM result."""
        n_nodes = 100
        rng = np.random.default_rng(42)
        disp = rng.standard_normal((n_nodes, 3)) * 1e-4
        stress_v = rng.standard_normal((n_nodes, 6)) * 1e6
        strain_v = rng.standard_normal((n_nodes, 6)) * 1e-4
        stress_t = rng.standard_normal((n_nodes, 3, 3)) * 1e6
        vm = np.abs(rng.standard_normal(n_nodes)) * 1e6

        return FEMResult(
            displacement=disp,
            stress_tensor=stress_t,
            stress_voigt=stress_v,
            strain_voigt=strain_v,
            von_mises=vm,
            solve_time_s=1.0,
        )

    def test_scale_factors_decrease_with_sf(self) -> None:
        """Higher safety factor → lower scale factor."""
        vm = np.abs(np.random.default_rng(42).standard_normal(100)) * 1e6
        factors = compute_scale_factors(vm, 420e6, [1.2, 2.0, 5.0])
        # factors[i] = (sf, factor); factor should decrease with sf
        assert factors[0][1] > factors[1][1] > factors[2][1]

    def test_linear_scaling_preserves_ratios(self) -> None:
        """Scaling by 2x should double all field values."""
        fem = self._make_mock_fem_result()
        linearity_config = {
            "bulk_families": ["block_with_holes"],
            "bending_families": [],
            "bulk_threshold": 1.0,  # permissive
        }
        result = apply_scaling(
            fem, scale_factor=2.0, target_sf=1.5,
            characteristic_length=1.0, peak_stress_cap=1e12,
            linearity_config=linearity_config,
            family_name="block_with_holes",
        )
        np.testing.assert_allclose(result.displacement, fem.displacement * 2.0)
        np.testing.assert_allclose(result.stress_voigt, fem.stress_voigt * 2.0)
        np.testing.assert_allclose(result.von_mises, fem.von_mises * 2.0)

    def test_peak_stress_cap_rejection(self) -> None:
        """Samples exceeding peak stress cap are rejected."""
        fem = self._make_mock_fem_result()
        linearity_config = {"bulk_families": ["test"], "bending_families": [],
                            "bulk_threshold": 1.0}
        result = apply_scaling(
            fem, scale_factor=1e6, target_sf=1.0,
            characteristic_length=1.0, peak_stress_cap=1.0,  # tiny cap
            linearity_config=linearity_config,
            family_name="test",
        )
        assert not result.accepted
        assert "cap" in result.rejection_reason.lower()


class TestSplitStrategy:
    """Tests for train/val/test splitting."""

    def _make_mock_manifest(self, n_base: int = 100) -> pd.DataFrame:
        """Create a mock manifest DataFrame."""
        rng = np.random.default_rng(42)
        families = ["block_with_holes", "l_bracket", "elongated_bar",
                     "plate_3d", "block_with_fillet"]
        buckets = ["small", "medium", "large"]

        rows = []
        for i in range(n_base):
            base_id = f"base_{i:06d}"
            family = rng.choice(families)
            bucket = rng.choice(buckets)
            for sf in [1.2, 1.5, 2.0, 3.0, 5.0]:
                row = {
                    "base_sample_id": base_id,
                    "sample_id": f"{base_id}_sf{sf}",
                    "geometry_family": family,
                    "scale_bucket": bucket,
                }
                if family == "l_bracket":
                    row["radius_ratio"] = float(rng.uniform(0.02, 0.30))
                rows.append(row)

        return pd.DataFrame(rows)

    def test_no_leakage(self) -> None:
        """All variants of one base sample must be in the same split."""
        df = self._make_mock_manifest()
        config = {
            "splitting": {"ratios": [0.8, 0.1, 0.1],
                           "large_scale_holdout_fraction": 0.5},
            "l_bracket": {"radius_ratio_holdout": [0.10, 0.20]},
            "validation": {"min_multihot_samples": 0},
            "sampling": {"random_seed": 42},
        }
        result = assign_splits(df, config)
        for base_id, group in result.groupby("base_sample_id"):
            assert group["split"].nunique() == 1, \
                f"Leakage detected for {base_id}"

    def test_all_splits_populated(self) -> None:
        """All three splits should have samples."""
        df = self._make_mock_manifest()
        config = {
            "splitting": {"ratios": [0.8, 0.1, 0.1],
                           "large_scale_holdout_fraction": 0.5},
            "l_bracket": {"radius_ratio_holdout": [0.10, 0.20]},
            "validation": {"min_multihot_samples": 0},
            "sampling": {"random_seed": 42},
        }
        result = assign_splits(df, config)
        splits = result["split"].unique()
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
