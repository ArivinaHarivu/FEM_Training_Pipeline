"""
Ensures the project root is on sys.path so `core`, `mesh_loader`, and
`validators` are importable as top-level packages during test runs,
regardless of the directory pytest is invoked from.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
