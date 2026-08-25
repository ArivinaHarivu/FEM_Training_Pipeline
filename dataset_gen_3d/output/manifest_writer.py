"""Manifest writer — incremental CSV output.

Appends one row per sample (variant) to the manifest CSV.
Uses file locking to be safe across Colab cell re-runs.
Writes headers only if the file doesn't exist yet.
"""

from __future__ import annotations

import csv
import pathlib
from typing import Any


class ManifestWriter:
    """Incrementally writes manifest rows to a CSV file.

    Thread-safe and Colab-disconnect-safe: each row is flushed
    immediately, so a mid-run disconnect loses at most the
    current sample.

    Parameters
    ----------
    path : pathlib.Path
        Output CSV file path.
    """

    def __init__(self, path: pathlib.Path) -> None:
        self._path = path
        self._headers_written = path.exists() and path.stat().st_size > 0
        self._fieldnames: list[str] | None = None

    def write_row(self, row: dict[str, Any]) -> None:
        """Append a single row to the manifest.

        On first call, writes the CSV header. Subsequent calls
        append data rows.

        Parameters
        ----------
        row : dict[str, Any]
            Column name → value mapping for one sample.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)

        if self._fieldnames is None:
            if self._headers_written:
                # Read existing headers to maintain column order
                with open(self._path, "r", newline="") as f:
                    reader = csv.reader(f)
                    self._fieldnames = next(reader)
            else:
                self._fieldnames = list(row.keys())

        # Once the header is written, the schema is locked. A row
        # introducing a new key here would corrupt the CSV (header
        # would no longer match data rows), so fail loudly instead.
        extra_keys = [k for k in row if k not in self._fieldnames]
        if extra_keys and self._headers_written:
            raise ValueError(
                f"Row introduces new column(s) {extra_keys} after header "
                f"was already written. Fix the row schema (e.g. in "
                f"SampleSpec.to_manifest_row) so all rows share the same "
                f"fixed set of keys from the very first row."
            )
        for key in row:
            if key not in self._fieldnames:
                self._fieldnames.append(key)

        mode = "a" if self._headers_written else "w"
        with open(self._path, mode, newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=self._fieldnames, extrasaction="ignore",
            )
            if not self._headers_written:
                writer.writeheader()
                self._headers_written = True
            writer.writerow(row)
            f.flush()  # immediate flush for disconnect safety
