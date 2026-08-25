"""Geometry families — each produces a distinct 3D solid topology."""

from dataset_gen_3d.geometry.families.block_with_holes import BlockWithHoles
from dataset_gen_3d.geometry.families.l_bracket import LBracket
from dataset_gen_3d.geometry.families.elongated_bar import ElongatedBar
from dataset_gen_3d.geometry.families.plate_3d import Plate3D
from dataset_gen_3d.geometry.families.block_with_fillet import BlockWithFillet

FAMILY_REGISTRY: dict[str, type] = {
    "block_with_holes": BlockWithHoles,
    "l_bracket": LBracket,
    "elongated_bar": ElongatedBar,
    "plate_3d": Plate3D,
    "block_with_fillet": BlockWithFillet,
}

__all__ = [
    "FAMILY_REGISTRY",
    "BlockWithHoles",
    "LBracket",
    "ElongatedBar",
    "Plate3D",
    "BlockWithFillet",
]
