"""
inspect_h5.py

Utility script to print the internal structure of a FEniCSx/SFEM .h5 file.
Run this on one of your raw .h5 files to discover the key paths needed
for the SFEMParser.

Usage:
    python inspect_h5.py path/to/your_file.h5
"""

import sys
import h5py
import numpy as np


def print_h5_structure(name, obj):
    indent = "  " * name.count("/")
    if isinstance(obj, h5py.Dataset):
        print(f"{indent}{name}  ->  shape={obj.shape}  dtype={obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"{indent}{name}/")


def inspect(file_path: str) -> None:
    print(f"\n{'='*60}")
    print(f"Inspecting: {file_path}")
    print(f"{'='*60}")
    with h5py.File(file_path, "r") as f:
        print("\n--- Full key tree ---")
        f.visititems(print_h5_structure)

        print("\n--- Top-level keys ---")
        print(list(f.keys()))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_h5.py <path_to_h5_file>")
        sys.exit(1)
    inspect(sys.argv[1])
