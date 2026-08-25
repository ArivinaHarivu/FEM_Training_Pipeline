"""3D FEM Dataset Generation Pipeline.

Generates synthetic linear-elastic FEM datasets for training
a MeshGraphNet-based structural analysis surrogate.

Toolchain: Gmsh (geometry + meshing) + FEniCS/Dolfin (solving).
Output: per-sample HDF5 files + manifest CSV.

Design constraint: No Monte Carlo aggregation. Every sample =
one geometry + one load config + one FEM solve. Load-magnitude
variants via exact linear scaling of a single solve's result.
"""
