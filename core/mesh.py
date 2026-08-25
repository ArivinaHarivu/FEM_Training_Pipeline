"""
mesh.py

Defines the canonical internal representation of an engineering mesh.
"""

from typing import List, Optional
import numpy as np

from core.element_block import ElementBlock


class Mesh:
    """
    Canonical internal representation of an engineering mesh.

    This class stores engineering data only.
    It is intentionally independent of any machine learning framework.

    Elements are stored as a list of typed `ElementBlock`s rather than a
    single merged array, since real meshes commonly mix element types
    (e.g. triangles and quads in a shell mesh) which cannot be represented
    as one homogeneous connectivity array. See ADR-004.
    """

    def __init__(self,
                 coordinates: np.ndarray,
                 elements: List[ElementBlock],
                 loads: Optional[np.ndarray] = None,
                 boundary_conditions: Optional[np.ndarray] = None,
                 mesh_metadata: Optional[dict] = None):

        self.coordinates = coordinates
        self.elements = elements
        self.loads = loads
        self.boundary_conditions = boundary_conditions
        self.mesh_metadata = mesh_metadata or {}

    @property
    def num_elements(self) -> int:
        """Total number of elements across all blocks."""
        return sum(block.num_elements for block in self.elements)

    @property
    def element_types(self) -> List[str]:
        """The element type of each block, in block order."""
        return [block.element_type for block in self.elements]
