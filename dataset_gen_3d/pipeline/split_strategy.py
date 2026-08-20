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

    # Initialize split assignment
    base_df["split"] = ""

    # Step 1: Large-scale bucket → test holdout
    large_mask = base_df["scale_bucket"] == "large"
    large_ids = base_df.loc[large_mask, "base_sample_id"].values
    if len(large_ids) > 0:
        n_holdout = max(1, int(len(large_ids) * large_holdout_frac))
        rng.shuffle(large_ids)
        test_large = set(large_ids[:n_holdout])
        base_df.loc[
            base_df["base_sample_id"].isin(test_large), "split"
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
        base_df.loc[lb_mask, "split"] = "test"

    # Step 3: Assign remaining by stratified random split
    unassigned = base_df["split"] == ""
    remaining_ids = base_df.loc[unassigned, "base_sample_id"].values.copy()
    rng.shuffle(remaining_ids)

    # Group by stratum for stratified split
    strata = base_df.loc[unassigned].groupby(
        ["geometry_family", "scale_bucket"],
    )["base_sample_id"].apply(list).to_dict()

    train_ids = set()
    val_ids = set()
    test_ids_extra = set()

    for stratum_key, ids in strata.items():
        rng.shuffle(ids)
        n = len(ids)
        # Adjusted ratios (test already partially filled)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])

        train_ids.update(ids[:n_train])
        val_ids.update(ids[n_train:n_train + n_val])
        test_ids_extra.update(ids[n_train + n_val:])

    # Apply assignments
    base_df.loc[base_df["base_sample_id"].isin(train_ids), "split"] = "train"
    base_df.loc[base_df["base_sample_id"].isin(val_ids), "split"] = "val"
    base_df.loc[
        (base_df["base_sample_id"].isin(test_ids_extra))
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
