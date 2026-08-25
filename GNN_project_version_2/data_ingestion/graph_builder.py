"""Graph builder — mesh connectivity → PyG edge_index.

Shared utility for both Experiment A (tet10) and Experiment B (tet4).
Converts raw element connectivity arrays into undirected, deduplicated
edge index tensors for PyTorch Geometric.

Also computes edge features (Euclidean distance, relative position).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch


def connectivity_to_edge_index(
    elements: np.ndarray,
) -> torch.Tensor:
    """Convert element connectivity to an undirected edge index.

    Works for any element shape (E, K): extracts all unique K*(K-1)/2
    edges per element, deduplicates, and returns both directions.

    Parameters
    ----------
    elements : np.ndarray
        Element connectivity, shape (E, K), 0-based.

    Returns
    -------
    torch.Tensor
        Edge index, shape (2, M), dtype long. Undirected (both directions).
    """
    k = elements.shape[1]

    # Generate all unique pairs within each element
    edges = set()
    for elem in elements:
        for i in range(k):
            for j in range(i + 1, k):
                a, b = int(elem[i]), int(elem[j])
                edges.add((a, b))
                edges.add((b, a))  # undirected

    edge_array = np.array(sorted(edges), dtype=np.int64)
    return torch.from_numpy(edge_array.T)  # (2, M)


def connectivity_to_edge_index_fast(
    elements: np.ndarray,
) -> torch.Tensor:
    """Vectorized version of connectivity_to_edge_index.

    Uses numpy broadcasting to avoid Python loops. Significantly faster
    for large meshes (>10k elements).

    Parameters
    ----------
    elements : np.ndarray
        Element connectivity, shape (E, K), 0-based.

    Returns
    -------
    torch.Tensor
        Edge index, shape (2, M), dtype long. Undirected.
    """
    k = elements.shape[1]

    # Generate all pairs (i, j) where i < j within an element
    rows, cols = np.triu_indices(k, k=1)

    # Gather node IDs for all pairs across all elements
    src = elements[:, rows].ravel()  # (E * n_pairs,)
    dst = elements[:, cols].ravel()

    # Stack both directions
    all_src = np.concatenate([src, dst])
    all_dst = np.concatenate([dst, src])

    # Deduplicate via unique
    edge_pairs = np.stack([all_src, all_dst], axis=0)  # (2, 2*E*n_pairs)
    edge_pairs = np.unique(edge_pairs, axis=1)

    return torch.from_numpy(edge_pairs.astype(np.int64))


def compute_edge_features(
    vertices: np.ndarray,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Compute edge features: Euclidean distance and relative position.

    Parameters
    ----------
    vertices : np.ndarray
        Node coordinates, shape (N, 3).
    edge_index : torch.Tensor
        Edge index, shape (2, M).

    Returns
    -------
    torch.Tensor
        Edge features, shape (M, 4): [dx, dy, dz, distance].
    """
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()

    delta = vertices[dst] - vertices[src]  # (M, 3)
    dist = np.linalg.norm(delta, axis=1, keepdims=True)  # (M, 1)

    features = np.hstack([delta, dist]).astype(np.float32)
    return torch.from_numpy(features)


def build_graph(
    vertices: np.ndarray,
    elements: np.ndarray,
    mode: Literal["tet4", "tet10"] = "tet4",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build graph topology + edge features from mesh data.

    Parameters
    ----------
    vertices : np.ndarray
        Node coordinates, shape (N, 3).
    elements : np.ndarray
        Element connectivity. Shape (E, 10) for tet10 or (E, 4) for tet4.
    mode : str
        ``"tet10"`` for Experiment A (full 10-node connectivity),
        ``"tet4"`` for Experiment B (corner nodes only).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        (edge_index, edge_attr) — shapes (2, M) and (M, 4).
    """
    if mode == "tet4" and elements.shape[1] == 10:
        # Project tet10 → tet4 by taking first 4 nodes
        elements = elements[:, :4]

    edge_index = connectivity_to_edge_index_fast(elements)
    edge_attr = compute_edge_features(vertices, edge_index)

    return edge_index, edge_attr
