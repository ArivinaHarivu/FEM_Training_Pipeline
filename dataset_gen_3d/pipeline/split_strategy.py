"""Split strategy — base-sample-aware, leakage-safe train/val/test splitting.

Guarantees:
1. All safety-factor variants of one base_sample_id stay in the same split
   (zero cross-split leakage from linearly-scaled variants).
2. Stratified by (family, scale_bucket) — every split has representation.
3. Large-scale bucket extrapolation holdout — configurable fraction → test.
4. L-bracket r/d holdout — configured sub-range → test only.
5. Multi-hot assertion — fail if < min_multihot_samples exist.

Order of operations:
1. Large-scale holdout assignment
2. L-bracket r/d holdout assignment
3. Fill remaining strata to per-stratum floor
4. Verify no base_sample_id leakage
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class SplitLeakageError(Exception):
    """Raised when base_sample_id variants leak across splits."""


class MultiHotAssertionError(Exception):
    """Raised when too few multi-hot (fixed+loaded) samples exist."""


def assign_splits(
    manifest_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Assign train/val/test splits to the manifest.

    Parameters
    ----------
    manifest_df : pd.DataFrame
        Full manifest with columns: base_sample_id, geometry_family,
        scale_bucket, and optionally radius_ratio.
    config : dict[str, Any]
        Full pipeline config.

    Returns
    -------
    pd.DataFrame
        Manifest with added ``split`` column.

    Raises
    ------
    SplitLeakageError
        If any base_sample_id has variants in more than one split.
    MultiHotAssertionError
        If fewer than min_multihot_samples have both is_fixed=1 and is_loaded=1.
    """
    split_config = config.get("splitting", {})
    ratios = split_config.get("ratios", [0.80, 0.10, 0.10])
    large_holdout_frac = split_config.get("large_scale_holdout_fraction", 0.5)

    l_bracket_config = config.get("l_bracket", {})
    rd_holdout = l_bracket_config.get("radius_ratio_holdout", [0.10, 0.20])

    validation_config = config.get("validation", {})
    min_multihot = validation_config.get("min_multihot_samples", 50)

    seed = config.get("sampling", {}).get("random_seed", 42)
    rng = np.random.default_rng(seed)

    # Work at the base_sample_id level (not individual variant level)
    base_df = manifest_df.drop_duplicates(subset="base_sample_id").copy()

    # ── Step 0: Compute geometry group ID ────────────────────────
    # Ensures duplicate/near-duplicate geometries always land in the
    # same split, preventing data leakage through geometry similarity.
    base_df["geometry_group"] = _compute_geometry_group(base_df)

    # Initialize split assignment
    base_df["split"] = ""

    # Step 1: Large-scale bucket → test holdout
    # Operate on geometry groups, not individual base_sample_ids
    large_mask = base_df["scale_bucket"] == "large"
    large_groups = base_df.loc[large_mask, "geometry_group"].unique()
    if len(large_groups) > 0:
        n_holdout = max(1, int(len(large_groups) * large_holdout_frac))
        rng.shuffle(large_groups)
        test_large_groups = set(large_groups[:n_holdout])
        base_df.loc[
            base_df["geometry_group"].isin(test_large_groups), "split"
        ] = "test"

    # Step 2: L-bracket r/d holdout → test
    if "radius_ratio" in base_df.columns:
        lb_mask = (
            (base_df["geometry_family"] == "l_bracket")
            & (base_df["radius_ratio"].notna())
            & (base_df["radius_ratio"] >= rd_holdout[0])
            & (base_df["radius_ratio"] <= rd_holdout[1])
            & (base_df["split"] == "")  # not already assigned
        )
        # Assign all members of the geometry group
        holdout_groups = base_df.loc[lb_mask, "geometry_group"].unique()
        base_df.loc[
            (base_df["geometry_group"].isin(holdout_groups))
            & (base_df["split"] == ""),
            "split",
        ] = "test"

    # Step 3: Assign remaining by stratified random split
    # Split at geometry_group level to prevent geometry leakage
    unassigned = base_df["split"] == ""
    unassigned_groups = base_df.loc[unassigned, "geometry_group"].unique()

    # Build stratum → group mapping
    group_stratum = (
        base_df.loc[unassigned]
        .drop_duplicates(subset="geometry_group")
        .set_index("geometry_group")[["geometry_family", "scale_bucket"]]
    )

    # Group geometry_groups by stratum
    strata: dict[tuple[str, str], list] = {}
    for gid, row in group_stratum.iterrows():
        key = (row["geometry_family"], row["scale_bucket"])
        strata.setdefault(key, []).append(gid)

    train_groups: set = set()
    val_groups: set = set()
    test_groups_extra: set = set()

    for stratum_key, gids in strata.items():
        rng.shuffle(gids)
        n = len(gids)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])

        train_groups.update(gids[:n_train])
        val_groups.update(gids[n_train:n_train + n_val])
        test_groups_extra.update(gids[n_train + n_val:])

    # Apply assignments via geometry_group
    base_df.loc[base_df["geometry_group"].isin(train_groups), "split"] = "train"
    base_df.loc[base_df["geometry_group"].isin(val_groups), "split"] = "val"
    base_df.loc[
        (base_df["geometry_group"].isin(test_groups_extra))
        & (base_df["split"] == ""),
        "split",
    ] = "test"

    # Any remaining unassigned → train (shouldn't happen, but safety net)
    base_df.loc[base_df["split"] == "", "split"] = "train"

    # Map back to full manifest (all variants of a base sample get same split)
    split_map = dict(zip(base_df["base_sample_id"], base_df["split"]))
    manifest_df["split"] = manifest_df["base_sample_id"].map(split_map)

    # Step 4: Verify no leakage
    _verify_no_leakage(manifest_df)

    # Step 5: Multi-hot assertion
    if "is_fixed" in manifest_df.columns and "is_loaded" in manifest_df.columns:
        _verify_multihot(manifest_df, min_multihot)

    return manifest_df


def _compute_geometry_group(base_df: pd.DataFrame) -> pd.Series:
    """Compute a geometry group ID for each base sample.

    Groups are defined by (family, rounded char_length, rounded
    family-specific params). Samples in the same group represent
    the same or near-identical geometry.

    Parameters
    ----------
    base_df : pd.DataFrame
        Deduplicated manifest (one row per base_sample_id).

    Returns
    -------
    pd.Series
        String group IDs, one per row.
    """
    parts = [
        base_df["geometry_family"].astype(str),
        base_df["characteristic_length"].round(6).astype(str),
    ]
    if "radius_ratio" in base_df.columns:
        parts.append(
            base_df["radius_ratio"].fillna(-1).round(6).astype(str)
        )
    if "fillet_radius" in base_df.columns:
        parts.append(
            base_df["fillet_radius"].fillna(-1).round(6).astype(str)
        )
    return parts[0].str.cat(parts[1:], sep="|")


def _verify_no_leakage(manifest_df: pd.DataFrame) -> None:
    """Assert no base_sample_id has variants in multiple splits.

    Parameters
    ----------
    manifest_df : pd.DataFrame
        Manifest with ``split`` and ``base_sample_id`` columns.

    Raises
    ------
    SplitLeakageError
        If leakage is detected.
    """
    splits_per_base = manifest_df.groupby("base_sample_id")["split"].nunique()
    leaked = splits_per_base[splits_per_base > 1]

    if len(leaked) > 0:
        raise SplitLeakageError(
            f"{len(leaked)} base_sample_ids have variants in multiple splits: "
            f"{leaked.index.tolist()[:10]}..."
        )


def _verify_multihot(
    manifest_df: pd.DataFrame,
    min_count: int,
) -> None:
    """Assert sufficient multi-hot (fixed+loaded) samples exist.

    Parameters
    ----------
    manifest_df : pd.DataFrame
        Manifest with ``is_fixed`` and ``is_loaded`` columns.
        These are sample-level flags indicating whether any node
        in the sample has both is_fixed=1 and is_loaded=1.
    min_count : int
        Minimum required count.

    Raises
    ------
    MultiHotAssertionError
        If count is below minimum.
    """
    if "has_multihot_nodes" in manifest_df.columns:
        count = int(manifest_df["has_multihot_nodes"].sum())
    else:
        # Fallback: check is_fixed and is_loaded columns
        both = (manifest_df["is_fixed"] == 1) & (manifest_df["is_loaded"] == 1)
        count = int(both.sum())

    if count < min_count:
        raise MultiHotAssertionError(
            f"Only {count} samples have multi-hot (fixed+loaded) nodes, "
            f"minimum required is {min_count}"
        )
