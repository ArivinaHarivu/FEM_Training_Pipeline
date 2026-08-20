"""Receptive field check — graph diameter via BFS.

Computes the graph diameter (longest shortest-path in hops) of the
mesh connectivity graph. Used to verify that elongated bar + large
scale produces high-diameter samples as expected.

The GNN's processor stack has 15 message-passing layers, giving a
nominal receptive field of 15 hops. Structures with diameter >> 15
may have nodes that cannot communicate (mitigated by global broadcast).
"""

from __future__ import annotations

import numpy as np
import networkx as nx


def compute_graph_diameter(
    elements: np.ndarray,
    n_nodes: int | None = None,
) -> int:
    """Compute the graph diameter of a tetrahedral mesh.

    Builds the adjacency graph from element connectivity, then computes
    the diameter via BFS from several sampled starting nodes (exact
    diameter via all-pairs shortest path is too expensive for large meshes).

    Parameters
    ----------
    elements : np.ndarray
        Tet connectivity, shape (E, 4), 0-based.
    n_nodes : int or None
        Total node count. If None, inferred from elements.

    Returns
    -------
    int
        Graph diameter (longest shortest path in hops).
        Returns -1 if the graph is disconnected.
    """
    if n_nodes is None:
        n_nodes = int(elements.max()) + 1

    # Build adjacency graph from tet edges
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    # Each tet has 6 edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    edges = set()
    for elem in elements:
        for i, j in edge_pairs:
            edge = (int(elem[i]), int(elem[j]))
            edges.add(edge)

    G.add_edges_from(edges)

    # Check connectivity
    if not nx.is_connected(G):
        return -1

    # Approximate diameter via double-BFS from sampled starting nodes
    # True diameter is expensive (O(V×E)); this gives exact result
    # for tree-like meshes and a tight approximation for others.
    diameter = _approximate_diameter(G, n_samples=5)

    return diameter


def _approximate_diameter(G: nx.Graph, n_samples: int = 5) -> int:
    """Approximate graph diameter via double-BFS.

    For each sampled start node:
    1. BFS to find the farthest node
    2. BFS from that farthest node
    3. The eccentricity of that farthest node is a lower bound on diameter

    The maximum across samples is a tight approximation.

    Parameters
    ----------
    G : nx.Graph
        The mesh adjacency graph.
    n_samples : int
        Number of starting nodes to sample.

    Returns
    -------
    int
        Approximate graph diameter.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n <= 1:
        return 0

    max_diameter = 0
    rng = np.random.default_rng(42)  # deterministic for reproducibility
    sample_indices = rng.choice(n, size=min(n_samples, n), replace=False)

    for idx in sample_indices:
        start = nodes[idx]

        # First BFS: find farthest node from start
        lengths_1 = nx.single_source_shortest_path_length(G, start)
        farthest_1 = max(lengths_1, key=lengths_1.get)

        # Second BFS: find diameter from farthest node
        lengths_2 = nx.single_source_shortest_path_length(G, farthest_1)
        local_diameter = max(lengths_2.values())

        max_diameter = max(max_diameter, local_diameter)

    return max_diameter


def compute_hop_features(
    elements: np.ndarray,
    fixed_mask: np.ndarray,
    loaded_mask: np.ndarray,
    n_nodes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute BFS hop distances to nearest fixed/loaded nodes.

    These are used as node features in the GNN (columns 6–7).

    Parameters
    ----------
    elements : np.ndarray
        Tet connectivity, shape (E, 4).
    fixed_mask : np.ndarray
        Boolean mask of fixed nodes, shape (N,).
    loaded_mask : np.ndarray
        Boolean mask of loaded nodes, shape (N,).
    n_nodes : int or None
        Total node count.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (hops_to_fixed, hops_to_loaded), each shape (N,).
        Unreachable nodes get value = n_nodes (effective infinity).
    """
    if n_nodes is None:
        n_nodes = int(elements.max()) + 1

    # Build adjacency
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    edge_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    edges = set()
    for elem in elements:
        for i, j in edge_pairs:
            edges.add((int(elem[i]), int(elem[j])))
    G.add_edges_from(edges)

    # Multi-source BFS from fixed nodes
    fixed_sources = set(np.where(fixed_mask)[0])
    hops_to_fixed = _multi_source_bfs(G, fixed_sources, n_nodes)

    # Multi-source BFS from loaded nodes
    loaded_sources = set(np.where(loaded_mask)[0])
    hops_to_loaded = _multi_source_bfs(G, loaded_sources, n_nodes)

    return hops_to_fixed, hops_to_loaded


def _multi_source_bfs(
    G: nx.Graph,
    sources: set[int],
    n_nodes: int,
) -> np.ndarray:
    """Compute shortest distance from any source node to all nodes.

    Parameters
    ----------
    G : nx.Graph
        The graph.
    sources : set[int]
        Source node IDs.
    n_nodes : int
        Total nodes.

    Returns
    -------
    np.ndarray
        Distance to nearest source, shape (N,). Default = n_nodes.
    """
    distances = np.full(n_nodes, n_nodes, dtype=np.int32)

    if not sources:
        return distances

    # Multi-source BFS
    lengths = nx.multi_source_shortest_path_length(G, sources)
    for node, dist in lengths:
        distances[node] = dist

    return distances
