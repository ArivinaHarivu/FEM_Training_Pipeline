"""
check_h5_indexing.py

Samples a few SFEM .h5 files from data/raw/ and determines whether
the Facets connectivity array uses 0-based or 1-based node indexing.

Run:
    python check_h5_indexing.py
"""

import os
import glob
import h5py
import numpy as np


_DATA_ROOT = os.path.join(os.path.dirname(__file__), "data", "raw")
_MAX_FILES = 5  # number of files to sample


def check_file(path: str) -> dict:
    with h5py.File(path, "r") as f:
        n_nodes = f["Vertices"].shape[0]
        connectivity = np.array(f["Facets"], dtype=np.int64)

    idx_min = int(connectivity.min())
    idx_max = int(connectivity.max())

    # 0-indexed:  min should be 0,         max should be n_nodes - 1
    # 1-indexed:  min should be 1,         max should be n_nodes
    if idx_min == 0 and idx_max == n_nodes - 1:
        verdict = "0-BASED (ok)"
    elif idx_min == 1 and idx_max == n_nodes:
        verdict = "1-BASED"
    elif idx_max >= n_nodes:
        verdict = f"OUT-OF-RANGE (max index {idx_max} >= n_nodes {n_nodes})"
    else:
        verdict = f"AMBIGUOUS (min={idx_min}, max={idx_max}, n_nodes={n_nodes})"

    return {
        "file": os.path.basename(path),
        "n_nodes": n_nodes,
        "n_elements": connectivity.shape[0],
        "nodes_per_elem": connectivity.shape[1],
        "idx_min": idx_min,
        "idx_max": idx_max,
        "verdict": verdict,
    }


def main():
    h5_files = glob.glob(os.path.join(_DATA_ROOT, "**", "*.h5"), recursive=True)

    if not h5_files:
        print(f"No .h5 files found under: {_DATA_ROOT}")
        print("Adjust _DATA_ROOT at the top of this script if needed.")
        return

    sample = h5_files[:_MAX_FILES]
    print(f"\nSampling {len(sample)} of {len(h5_files)} .h5 file(s) found under {_DATA_ROOT}\n")
    print(f"{'File':<40} {'Nodes':>7} {'Elems':>7} {'N/E':>4} {'IdxMin':>7} {'IdxMax':>7}  Verdict")
    print("-" * 95)

    verdicts = []
    for path in sample:
        r = check_file(path)
        print(
            f"{r['file']:<40} {r['n_nodes']:>7} {r['n_elements']:>7} "
            f"{r['nodes_per_elem']:>4} {r['idx_min']:>7} {r['idx_max']:>7}  {r['verdict']}"
        )
        verdicts.append(r["verdict"])

    print()
    unique = set(verdicts)
    if unique == {"0-BASED (ok)"}:
        print("CONCLUSION: All sampled files use 0-based indexing. Safe to use directly.")
    elif all("1-BASED" in v for v in unique):
        print("CONCLUSION: All sampled files use 1-based indexing. Parser must subtract 1.")
    else:
        print("CONCLUSION: Mixed or unexpected results — review individual rows above.")


if __name__ == "__main__":
    main()
