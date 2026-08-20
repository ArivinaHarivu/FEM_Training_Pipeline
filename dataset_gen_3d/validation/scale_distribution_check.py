"""Scale distribution check — post-generation diagnostic.

Reports:
1. Distribution of characteristic length and node count across dataset
2. Family × scale_bucket count table (proves balance was achieved)
3. Per-split base_sample_id counts
4. Both linearity gate check types reported separately per family
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def run_scale_distribution_check(
    manifest_path: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Run the post-generation scale distribution check.

    Parameters
    ----------
    manifest_path : str
        Path to the generated manifest CSV.
    config : dict[str, Any]
        Full pipeline config.

    Returns
    -------
    dict[str, Any]
        Report containing:
        - ``stratum_table``: family × scale_bucket counts (DataFrame)
        - ``split_counts``: per-split base_sample_id counts
        - ``length_stats``: characteristic length distribution stats
        - ``node_count_stats``: node count distribution stats
        - ``linearity_gate_stats``: per-family, per-gate-type counts
        - ``warnings``: list of any balance or coverage issues
    """
    df = pd.read_csv(manifest_path)
    report: dict[str, Any] = {"warnings": []}

    # 1. Stratum table: family × scale_bucket
    stratum_table = pd.crosstab(
        df["geometry_family"],
        df["scale_bucket"],
        margins=True,
    )
    report["stratum_table"] = stratum_table

    # Check stratum balance
    min_per_stratum = config.get("sampling", {}).get(
        "min_base_samples_per_stratum", 250,
    )
    # Count unique base_sample_ids per stratum
    if "base_sample_id" in df.columns:
        base_counts = df.groupby(
            ["geometry_family", "scale_bucket"],
        )["base_sample_id"].nunique()
        for (family, bucket), count in base_counts.items():
            if count < min_per_stratum:
                report["warnings"].append(
                    f"Stratum ({family}, {bucket}) has {count} base samples "
                    f"< minimum {min_per_stratum}"
                )
    report["base_stratum_counts"] = (
        base_counts.to_dict() if "base_sample_id" in df.columns else {}
    )

    # 2. Per-split counts
    if "split" in df.columns:
        split_counts = df.groupby("split").agg(
            total_samples=("sample_id", "count"),
            base_samples=(
                "base_sample_id",
                lambda x: x.nunique() if "base_sample_id" in df.columns else len(x),
            ),
        )
        report["split_counts"] = split_counts
    else:
        report["split_counts"] = None

    # 3. Characteristic length distribution
    if "characteristic_length" in df.columns:
        lengths = df["characteristic_length"]
        report["length_stats"] = {
            "min": float(lengths.min()),
            "max": float(lengths.max()),
            "mean": float(lengths.mean()),
            "median": float(lengths.median()),
            "std": float(lengths.std()),
            "p5": float(lengths.quantile(0.05)),
            "p95": float(lengths.quantile(0.95)),
        }

        # Check range coverage
        configured_range = config.get("scale", {}).get(
            "characteristic_length_range", [0.02, 2.0],
        )
        if lengths.min() > configured_range[0] * 1.5:
            report["warnings"].append(
                f"Smallest length {lengths.min():.4f} is far from "
                f"configured min {configured_range[0]}"
            )
        if lengths.max() < configured_range[1] * 0.7:
            report["warnings"].append(
                f"Largest length {lengths.max():.4f} is far from "
                f"configured max {configured_range[1]}"
            )

    # 4. Node count distribution
    if "node_count" in df.columns:
        nodes = df["node_count"]
        report["node_count_stats"] = {
            "min": int(nodes.min()),
            "max": int(nodes.max()),
            "mean": float(nodes.mean()),
            "median": float(nodes.median()),
        }

    # 5. Linearity gate stats — per family, per gate type
    if "linearity_gate_type" in df.columns:
        gate_stats = df.groupby(
            ["geometry_family", "linearity_gate_type"],
        ).agg(
            total=("sample_id", "count"),
            passed=("linearity_gate_pass", "sum"),
        )
        gate_stats["failed"] = gate_stats["total"] - gate_stats["passed"]
        report["linearity_gate_stats"] = gate_stats

    return report


def print_report(report: dict[str, Any]) -> str:
    """Format the distribution check report as a readable string.

    Parameters
    ----------
    report : dict[str, Any]
        Output from ``run_scale_distribution_check``.

    Returns
    -------
    str
        Formatted report text.
    """
    lines = ["=" * 60, "SCALE DISTRIBUTION CHECK REPORT", "=" * 60, ""]

    # Stratum table
    if "stratum_table" in report:
        lines.append("Family × Scale Bucket Counts:")
        lines.append(str(report["stratum_table"]))
        lines.append("")

    # Split counts
    if report.get("split_counts") is not None:
        lines.append("Per-Split Counts:")
        lines.append(str(report["split_counts"]))
        lines.append("")

    # Length stats
    if "length_stats" in report:
        lines.append("Characteristic Length Distribution:")
        for k, v in report["length_stats"].items():
            lines.append(f"  {k}: {v:.4f}")
        lines.append("")

    # Node count stats
    if "node_count_stats" in report:
        lines.append("Node Count Distribution:")
        for k, v in report["node_count_stats"].items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    # Linearity gate
    if "linearity_gate_stats" in report:
        lines.append("Linearity Gate Stats (per family × gate type):")
        lines.append(str(report["linearity_gate_stats"]))
        lines.append("")

    # Warnings
    if report.get("warnings"):
        lines.append("⚠ WARNINGS:")
        for w in report["warnings"]:
            lines.append(f"  - {w}")
    else:
        lines.append("✓ No warnings — all checks passed.")

    return "\n".join(lines)
