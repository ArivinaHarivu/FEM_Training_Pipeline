"""
element_block.py

Defines a single homogeneous group of mesh elements sharing one element
type (e.g. all triangles, or all quads). A Mesh is composed of one or
more ElementBlocks -- see ADR-004 for why elements are stored this way
rather than as a single merged array.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ElementBlock:
    """
    A homogeneous block of mesh elements of a single type.

    Parameters
    ----------
    element_type : str
        The element type name (e.g. "triangle", "quad", "tetra",
        "hexahedron"). Matches meshio's cell type naming.
    connectivity : np.ndarray
        Node indices per element, shape (num_elements, nodes_per_element).
        Row i lists the node indices making up element i.
    material_ids : Optional[np.ndarray]
        Material tag per element, shape (num_elements,). None if the
        source mesh had no material/physical-group tags for this block.
    """
    element_type: str
    connectivity: np.ndarray
    material_ids: Optional[np.ndarray] = None

    @property
    def num_elements(self) -> int:
        """Number of elements in this block."""
        return self.connectivity.shape[0]

    @property
    def nodes_per_element(self) -> int:
        """Number of nodes per element in this block."""
        return self.connectivity.shape[1]
