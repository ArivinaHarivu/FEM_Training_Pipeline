"""Main orchestrator — generates the full dataset.

Coordinates: geometry generation → mesh conversion → FEM solve →
load scaling → validation → HDF5 output → manifest writing.

Supports:
- Stratum-aware sampling (fills under-represented family × scale_bucket)
- Incremental manifest writing (Colab-disconnect-safe)
- Resumability (skips completed sample IDs)
- Per-sample exception handling (no single failure halts the run)
- Calibration-only mode (runs pilot batch)
- Zero-field rejection gate (H7)
- Post-loop stratum floor verification (H8)

Usage:
    python -m pipeline.generate_dataset --config config.yaml
    python -m pipeline.generate_dataset --config config.yaml --calibration-only
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time
from typing import Any

import numpy as np
import yaml

from dataset_gen_3d.geometry.families import FAMILY_REGISTRY
from dataset_gen_3d.geometry.families.base_family import (
    GeometryFamily,
    GeometryGenerationError,
)
from dataset_gen_3d.geometry.scale_sampler import ScaleSampler
from dataset_gen_3d.output.hdf5_writer import write_sample_hdf5
from dataset_gen_3d.output.manifest_writer import ManifestWriter
from dataset_gen_3d.pipeline.resume import get_completed_sample_ids
from dataset_gen_3d.pipeline.sample_spec import SampleSpec
from dataset_gen_3d.solve.boundary_conditions import apply_boundary_conditions
from dataset_gen_3d.solve.load_scaling import generate_load_variants
from dataset_gen_3d.solve.loads import apply_loads
from dataset_gen_3d.solve.material import Material
from dataset_gen_3d.solve.mesh_conversion import convert_msh_to_xdmf, extract_mesh_data
from dataset_gen_3d.validation.coherence_check import run_coherence_check
from dataset_gen_3d.validation.receptive_field_check import compute_graph_diameter

logger = logging.getLogger(__name__)

# Minimum reference-solve P95 von Mises to accept a sample (H7).
# Below this threshold, the geometry/BC configuration produced
# effectively zero stress, meaning all scaled variants would be
# zero-field samples — useless training data.
_MIN_REFERENCE_P95_VM = 1.0  # Pa


def load_config(config_path: str) -> dict[str, Any]:
    """Load and validate the pipeline config.

    Parameters
    ----------
    config_path : str
        Path to config.yaml.

    Returns
    -------
    dict[str, Any]
        Parsed configuration.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


class DatasetGenerator:
    """Main dataset generation orchestrator.

    Parameters
    ----------
    config : dict[str, Any]
        Full pipeline configuration.
    output_dir : pathlib.Path
        Root output directory.
    calibration_only : bool
        If True, run only the calibration pilot batch.
    """

    def __init__(
        self,
        config: dict[str, Any],
        output_dir: pathlib.Path,
        calibration_only: bool = False,
    ) -> None:
        self._config = config
        self._output_dir = output_dir
        self._calibration_only = calibration_only

        self._material = Material.from_config(config["material"])
        self._scale_sampler = ScaleSampler(config["scale"])

        # Build geometry family instances
        self._families: dict[str, GeometryFamily] = {}
        for family_name in config["geometry"]["families"]:
            family_config = config.get(family_name, {})
            family_cls = FAMILY_REGISTRY[family_name]
            self._families[family_name] = family_cls(family_config)

        self._family_weights = np.array(
            config["geometry"]["family_weights"], dtype=np.float64,
        )
        self._family_weights /= self._family_weights.sum()
        self._family_names = config["geometry"]["families"]

        # Sampling config
        sampling = config.get("sampling", {})
        self._seed = sampling.get("random_seed", 42)
        self._min_per_stratum = sampling.get("min_base_samples_per_stratum", 250)
        self._total_target = sampling.get("total_target_base_samples", 5000)
        if self._calibration_only:
            self._total_target = config.get("calibration", {}).get(
                "pilot_samples", 30,
            )

        # Load scaling config
        self._sf_targets = config["load_scaling"]["safety_factor_targets"]
        self._stress_percentile = config["load_scaling"]["stress_percentile"]
        self._peak_stress_cap = config["load_scaling"]["peak_stress_cap"]
        self._linearity_config = config.get("linearity_gate", {})

        # Output paths
        self._hdf5_dir = output_dir / config["output"]["hdf5_dir"]
        self._geom_dir = output_dir / config["output"]["geometry_dir"]
        self._manifest_path = output_dir / config["output"]["manifest_path"]
        self._log_dir = output_dir / config["output"]["log_dir"]
        self._progress_interval = config["output"].get("progress_interval", 50)

        # Solver config
        self._ref_load_mag = config["solver"]["reference_load_magnitude"]

        # Validation config
        self._coherence_tol = config["validation"]["coherence_tolerance"]

        # RNG
        self._rng = np.random.default_rng(self._seed)

    def run(self) -> None:
        """Execute the dataset generation pipeline."""
        # Setup directories
        self._hdf5_dir.mkdir(parents=True, exist_ok=True)
        self._geom_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)

        # Setup logging
        log_file = self._log_dir / (
            "calibration.log" if self._calibration_only else "generation.log"
        )
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(str(log_file)),
                logging.StreamHandler(),
            ],
        )

        logger.info(
            "Starting %s: target=%d base samples, seed=%d",
            "calibration" if self._calibration_only else "generation",
            self._total_target,
            self._seed,
        )

        # Resume: skip completed samples
        completed = get_completed_sample_ids(self._manifest_path)
        if completed:
            logger.info("Resuming: %d base samples already completed", len(completed))

        # Manifest writer
        manifest_writer = ManifestWriter(self._manifest_path)

        # Stratum tracking
        strata_counts: dict[tuple[str, str], int] = {}
        for family in self._family_names:
            for bucket in self._config["scale"]["buckets"]:
                strata_counts[(family, bucket)] = 0

        # Count existing completed samples per stratum
        if completed and self._manifest_path.exists():
            import pandas as pd
            existing_df = pd.read_csv(self._manifest_path)
            for (f, b), group in existing_df.groupby(
                ["geometry_family", "scale_bucket"],
            ):
                strata_counts[(f, b)] = group["base_sample_id"].nunique()

        # Generation loop
        n_generated = len(completed)
        n_success = 0
        n_failed = 0
        n_coherence_pass = 0
        n_zero_field_rejected = 0
        n_multihot = 0
        start_time = time.perf_counter()

        while n_generated < self._total_target:
            sample_idx = n_generated
            sample_id = f"base_{sample_idx:06d}"

            if sample_id in completed:
                n_generated += 1
                continue

            try:
                # Select family and scale (stratum-aware)
                family_name, scale_sample = self._select_stratum(strata_counts)
                family = self._families[family_name]

                # Sample shape params
                params = family.sample_params(self._rng)

                # Generate geometry
                msh_path = self._geom_dir / f"{sample_id}.msh"
                geom_result = family.generate(
                    params, scale_sample.characteristic_length,
                    scale_sample.mesh_size, msh_path,
                )

                # Extract mesh data (tet4 + tet10 + surface nodes)
                mesh_data = extract_mesh_data(msh_path)
                vertices = mesh_data["vertices"]
                elements_tet4 = mesh_data["elements_tet4"]
                elements_tet10 = mesh_data.get("elements_tet10")
                is_surface_node = mesh_data["is_surface_node"]

                # Apply loads and BCs (with face name validation — H10)
                load_spec = apply_loads(
                    vertices, geom_result.surface_tags,
                    self._ref_load_mag, self._rng,
                )
                bc_spec = apply_boundary_conditions(
                    vertices, geom_result.surface_tags,
                    load_spec.load_face, self._rng,
                )

                # Build sample spec
                spec = SampleSpec(
                    sample_id=sample_id,
                    family_name=family_name,
                    shape_params=geom_result.params,
                    characteristic_length=scale_sample.characteristic_length,
                    scale_bucket=scale_sample.scale_bucket,
                    mesh_size=scale_sample.mesh_size,
                    fixed_face=bc_spec.fixed_face,
                    load_face=load_spec.load_face,
                    load_direction=load_spec.direction.tolist(),
                    governing_thickness=geom_result.governing_thickness,
                )

                # Convert mesh for FEniCS (tet4 XDMF)
                xdmf_path = self._geom_dir / f"{sample_id}.xdmf"
                convert_msh_to_xdmf(msh_path, xdmf_path)

                # FEM solve
                from dataset_gen_3d.solve.run_fem_3d import run_fem_solve

                fem_result = run_fem_solve(
                    xdmf_path=xdmf_path,
                    fixed_face_name=bc_spec.fixed_face,
                    load_face_name=load_spec.load_face,
                    traction_direction=load_spec.direction,
                    traction_magnitude=load_spec.magnitude,
                    E=self._material.E,
                    nu=self._material.nu,
                    vertices=vertices,
                    elements_tet4=elements_tet4,
                )

                # ── Zero-field rejection gate (H7) ──────────────────
                ref_p95_vm = float(
                    np.percentile(fem_result.von_mises, self._stress_percentile)
                )
                if ref_p95_vm < _MIN_REFERENCE_P95_VM:
                    n_zero_field_rejected += 1
                    logger.warning(
                        "Sample %s rejected: reference P95 von Mises "
                        "%.4e Pa < %.1f Pa threshold (zero-field sample)",
                        sample_id, ref_p95_vm, _MIN_REFERENCE_P95_VM,
                    )
                    n_generated += 1
                    continue

                # Compute graph diameter (uses tet4 connectivity)
                graph_diameter = compute_graph_diameter(
                    elements_tet4, len(vertices),
                )

                # Generate load-level variants
                variants = generate_load_variants(
                    fem_result=fem_result,
                    sigma_yield=self._material.sigma_yield,
                    target_safety_factors=self._sf_targets,
                    characteristic_length=scale_sample.characteristic_length,
                    peak_stress_cap=self._peak_stress_cap,
                    linearity_config=self._linearity_config,
                    family_name=family_name,
                    governing_thickness=geom_result.governing_thickness,
                    stress_percentile=self._stress_percentile,
                )

                # Check for multi-hot nodes
                has_multihot = bool(
                    np.any(bc_spec.fixed_node_mask & load_spec.loaded_node_mask)
                )
                if has_multihot:
                    n_multihot += 1

                # Use traction-consistent RHS forces from FEM solve
                # instead of uniform approximation (H3 fix)
                rhs_forces = fem_result.rhs_nodal_forces
                rhs_has_data = np.any(np.abs(rhs_forces) > 1e-30)
                if not rhs_has_data:
                    # Fallback to approximate forces if RHS extraction failed
                    rhs_forces = load_spec.nodal_forces

                # Write each accepted variant
                for variant in variants:
                    variant_id = f"{sample_id}_sf{variant.target_sf:.1f}"

                    # Coherence check (element-level comparison)
                    coherence = run_coherence_check(
                        vertices=vertices,
                        elements=elements_tet10 if elements_tet10 is not None else elements_tet4,
                        displacement=variant.displacement,
                        solver_von_mises=variant.von_mises,
                        E=self._material.E,
                        nu=self._material.nu,
                        tolerance=self._coherence_tol,
                    )

                    if coherence.passed:
                        n_coherence_pass += 1

                    # Write HDF5 (even if rejected, for debugging)
                    if variant.accepted:
                        hdf5_path = self._hdf5_dir / f"{variant_id}.h5"
                        write_sample_hdf5(
                            path=hdf5_path,
                            vertices=vertices,
                            elements_tet4=elements_tet4,
                            displacement=variant.displacement,
                            stress_voigt_nodal=variant.stress_voigt,
                            strain_voigt_nodal=variant.strain_voigt,
                            stress_voigt_elem=variant.stress_voigt_elem,
                            strain_voigt_elem=variant.strain_voigt_elem,
                            stress_tensor=variant.stress_tensor,
                            von_mises_nodal=variant.von_mises,
                            von_mises_elem=variant.von_mises_elem,
                            is_fixed=bc_spec.fixed_node_mask,
                            is_loaded=load_spec.loaded_node_mask,
                            is_surface_node=is_surface_node,
                            nodal_forces=rhs_forces * variant.scale_factor,
                            load_class=f"SF_{variant.target_sf:.1f}",
                            reaction_forces=fem_result.reaction_forces * variant.scale_factor,
                            elements_tet10=elements_tet10,
                        )

                    # Write manifest row
                    manifest_writer.write_row({
                        **spec.to_manifest_row(),
                        "sample_id": variant_id,
                        "target_safety_factor": variant.target_sf,
                        "scale_factor": variant.scale_factor,
                        "node_count": geom_result.node_count,
                        "element_count": geom_result.element_count,
                        "graph_diameter": graph_diameter,
                        "peak_von_mises": variant.peak_von_mises,
                        "p95_von_mises": variant.p95_von_mises,
                        "max_displacement": variant.max_displacement,
                        "coherence_pass": coherence.passed,
                        "coherence_error": coherence.relative_error,
                        "linearity_gate_pass": variant.accepted,
                        "linearity_gate_type": variant.linearity_gate_type,
                        "rejection_reason": variant.rejection_reason,
                        "solve_time_s": fem_result.solve_time_s,
                        "has_multihot_nodes": has_multihot,
                        "has_tet10": elements_tet10 is not None,
                    })

                # Update stratum count
                strata_counts[(family_name, scale_sample.scale_bucket)] += 1
                n_success += 1

            except (GeometryGenerationError, Exception) as e:
                n_failed += 1
                logger.warning(
                    "Sample %s failed: %s", sample_id, str(e),
                )

            n_generated += 1

            # Progress logging
            if n_generated % self._progress_interval == 0:
                elapsed = time.perf_counter() - start_time
                rate = n_generated / elapsed if elapsed > 0 else 0
                remaining = (self._total_target - n_generated) / rate if rate > 0 else 0
                logger.info(
                    "Progress: %d/%d (%.1f%%) | success=%d fail=%d "
                    "coherence_pass=%d zero_rejected=%d multihot=%d | "
                    "%.2f samples/s | ETA %.0fs",
                    n_generated, self._total_target,
                    100 * n_generated / self._total_target,
                    n_success, n_failed, n_coherence_pass,
                    n_zero_field_rejected, n_multihot,
                    rate, remaining,
                )

        # ── Post-loop stratum floor check (H8) ──────────────────────
        _verify_strata_floors(
            strata_counts, self._min_per_stratum,
        )

        # ── Multi-hot count summary ─────────────────────────────────
        min_multihot = self._config.get("validation", {}).get(
            "min_multihot_samples", 50,
        )
        if n_multihot < min_multihot:
            logger.warning(
                "Multi-hot sample count %d < minimum %d",
                n_multihot, min_multihot,
            )
        else:
            logger.info(
                "Multi-hot samples: %d (min=%d) ✓", n_multihot, min_multihot,
            )

        logger.info(
            "Generation complete: %d base samples, %d successes, "
            "%d failures, %d zero-field rejected",
            n_generated, n_success, n_failed, n_zero_field_rejected,
        )

    def _select_stratum(
        self,
        strata_counts: dict[tuple[str, str], int],
    ) -> tuple[str, Any]:
        """Select a (family, scale_bucket) stratum, biased toward under-filled.

        If any stratum is below the minimum floor, samples from that stratum.
        Otherwise, samples randomly with configured family weights.

        Parameters
        ----------
        strata_counts : dict[tuple[str, str], int]
            Current per-stratum base sample counts.

        Returns
        -------
        tuple[str, ScaleSample]
            Selected family name and scale sample.
        """
        # Find under-filled strata
        under_filled = [
            (f, b) for (f, b), count in strata_counts.items()
            if count < self._min_per_stratum
        ]

        if under_filled:
            # Pick randomly from under-filled strata
            idx = self._rng.integers(0, len(under_filled))
            family_name, bucket = under_filled[idx]
            scale_sample = self._scale_sampler.sample_for_bucket(bucket, self._rng)
        else:
            # All strata at floor — sample randomly
            family_name = self._rng.choice(
                self._family_names, p=self._family_weights,
            )
            scale_sample = self._scale_sampler.sample(self._rng)

        return family_name, scale_sample


def _verify_strata_floors(
    strata_counts: dict[tuple[str, str], int],
    min_per_stratum: int,
) -> None:
    """Check that all strata met their minimum floor (H8).

    Logs a WARNING for each under-filled stratum — does not raise,
    since the data is still usable, just imbalanced.

    Parameters
    ----------
    strata_counts : dict[tuple[str, str], int]
        Achieved per-stratum counts.
    min_per_stratum : int
        Required floor.
    """
    under_filled = {
        k: v for k, v in strata_counts.items()
        if v < min_per_stratum
    }

    if under_filled:
        logger.warning(
            "⚠ %d strata below floor (%d):",
            len(under_filled), min_per_stratum,
        )
        for (family, bucket), count in sorted(under_filled.items()):
            logger.warning(
                "  (%s, %s): %d / %d (shortfall: %d)",
                family, bucket, count, min_per_stratum,
                min_per_stratum - count,
            )
    else:
        logger.info(
            "All %d strata met minimum floor of %d ✓",
            len(strata_counts), min_per_stratum,
        )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate 3D FEM dataset for GNN training.",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--calibration-only", action="store_true",
        help="Run only the calibration pilot batch",
    )
    parser.add_argument(
        "--output-dir", type=str, default=".",
        help="Root output directory",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    generator = DatasetGenerator(
        config=config,
        output_dir=pathlib.Path(args.output_dir),
        calibration_only=args.calibration_only,
    )
    generator.run()


if __name__ == "__main__":
    main()
