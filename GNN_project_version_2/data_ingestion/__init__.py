"""GNN data ingestion pipeline for FEM dataset.

Converts per-sample HDF5 files + manifest CSV into PyTorch Geometric
Data objects for training a MeshGraphNet-based surrogate model.

Supports two graph connectivity experiments:
  - Experiment A: tet10 graph (full 10-node quadratic connectivity)
  - Experiment B: tet4 graph (projected down from tet10 corners)
"""
