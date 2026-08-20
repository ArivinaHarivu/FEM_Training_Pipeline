"""Resume — skip completed samples on rerun.

Reads the manifest to determine which sample IDs have already been
generated, so the pipeline can resume after a Colab disconnect
losing at most the current in-progress sample.
"""

from __future__ import annotations

import pathlib
from typing import Set

import pandas as pd


def get_completed_sample_ids(manifest_path: pathlib.Path) -> Set[str]:
    """Read the manifest and return IDs of completed samples.

    Parameters
    ----------
    manifest_path : pathlib.Path
        Path to the manifest CSV.

    Returns
    -------
    set[str]
        Set of completed base_sample_id values.
    """
    if not manifest_path.exists():
        return set()

    try:
        df = pd.read_csv(manifest_path)
        if "base_sample_id" in df.columns:
            return set(df["base_sample_id"].unique())
        elif "sample_id" in df.columns:
            return set(df["sample_id"].unique())
        return set()
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return set()


def get_completed_count(manifest_path: pathlib.Path) -> int:
    """Get the count of completed base samples.

    Parameters
    ----------
    manifest_path : pathlib.Path
        Path to the manifest CSV.

    Returns
    -------
    int
        Number of unique base samples completed.
    """
    return len(get_completed_sample_ids(manifest_path))
