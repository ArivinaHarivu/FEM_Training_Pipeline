"""Deduplication diagnostic — analyze a manifest for geometry duplicates.

Reads a manifest CSV and reports:
- Per-family duplicate characteristic_length counts
- Per-stratum (family × scale_bucket) duplicate rates
- Full geometry-key duplicate analysis

Usage:
    python -m dataset_gen_3d.pipeline.dedup_diagnostic output/manifest.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _round_col(series: pd.Series, decimals: int = 8) -> pd.Series:
    """Round a numeric series for dedup comparison."""
    return series.round(decimals)


def analyze_manifest(manifest_path: str | Path) -> None:
    """Print a duplicate-geometry diagnostic report.

    Parameters
    ----------
    manifest_path : str or Path
        Path to manifest CSV.
    """
    df = pd.read_csv(manifest_path)
    print(f"Manifest: {manifest_path}")
    print(f"Total rows: {len(df)}")

    # Work at base_sample level
    base_df = df.drop_duplicates(subset="base_sample_id").copy()
    n_base = len(base_df)
    print(f"Unique base samples: {n_base}")

    families = base_df["geometry_family"].unique()
    print(f"Families: {list(families)}")
    print()

    # ── Per-family characteristic_length duplicates ──────────────
    print("=" * 60)
    print("CHARACTERISTIC LENGTH DUPLICATES (per family)")
    print("=" * 60)
    base_df["cl_rounded"] = _round_col(base_df["characteristic_length"])

    for family in sorted(families):
        fam_df = base_df[base_df["geometry_family"] == family]
        n = len(fam_df)
        n_unique_cl = fam_df["cl_rounded"].nunique()
        n_involved = n - n_unique_cl  # rows that are duplicates of something
        pct = 100 * (1 - n_unique_cl / n) if n > 0 else 0

        print(f"\n  {family}:")
        print(f"    base samples:          {n}")
        print(f"    unique char_lengths:   {n_unique_cl}")
        print(f"    duplicate rate:        {pct:.1f}%")

        # Show duplicate groups
        dup_groups = (
            fam_df.groupby("cl_rounded")
            .filter(lambda g: len(g) > 1)
            .groupby("cl_rounded")
            .size()
        )
        if len(dup_groups) > 0:
            print(f"    duplicate groups:      {len(dup_groups)}")
            print(f"    largest group size:    {dup_groups.max()}")

    # ── Per-stratum analysis ────────────────────────────────────
    if "scale_bucket" in base_df.columns:
        print()
        print("=" * 60)
        print("PER-STRATUM DUPLICATE RATES (family × scale_bucket)")
        print("=" * 60)

        for (family, bucket), group in base_df.groupby(
            ["geometry_family", "scale_bucket"]
        ):
            n = len(group)
            n_unique = group["cl_rounded"].nunique()
            pct = 100 * (1 - n_unique / n) if n > 0 else 0
            status = "⚠" if pct > 10 else "✓"
            print(
                f"  ({family:20s}, {bucket:8s}): "
                f"{n_unique:3d} unique / {n:3d} total  "
                f"({pct:5.1f}% dup) {status}"
            )

    # ── Full geometry-key analysis ──────────────────────────────
    print()
    print("=" * 60)
    print("FULL GEOMETRY KEY DUPLICATES")
    print("(family + char_length + radius_ratio + fillet_radius)")
    print("=" * 60)

    key_cols = ["geometry_family", "cl_rounded"]
    if "radius_ratio" in base_df.columns:
        base_df["rr_rounded"] = _round_col(
            base_df["radius_ratio"].fillna(-999)
        )
        key_cols.append("rr_rounded")
    if "fillet_radius" in base_df.columns:
        base_df["fr_rounded"] = _round_col(
            base_df["fillet_radius"].fillna(-999)
        )
        key_cols.append("fr_rounded")

    n_unique_full = base_df.drop_duplicates(subset=key_cols).shape[0]
    pct_full = 100 * (1 - n_unique_full / n_base) if n_base > 0 else 0
    print(f"\n  Total base samples:            {n_base}")
    print(f"  Unique geometry keys:          {n_unique_full}")
    print(f"  Overall duplicate rate:        {pct_full:.1f}%")
    print(f"  Effective distinct geometries: {n_unique_full}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze manifest CSV for geometry duplicates.",
    )
    parser.add_argument("manifest", help="Path to manifest.csv")
    args = parser.parse_args()
    analyze_manifest(args.manifest)


if __name__ == "__main__":
    main()
