"""Tree analysis utilities for TMAP.

Functions for analyzing properties of a fitted TMAP tree: boundary
detection, confusion matrices, edge gradients, subtree purity, and
path extraction.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray

from tmap.graph.types import Tree


def boundary_edges(tree: Tree, labels: NDArray) -> NDArray[np.intp]:
    """Return indices of edges connecting nodes with different labels.

    Parameters
    ----------
    tree : Tree
        Fitted tree.
    labels : array-like of shape (n_nodes,)
        Per-node labels (categorical strings or integers).

    Returns
    -------
    NDArray[np.intp]
        Edge indices where labels differ across the edge.
    """
    labels = np.asarray(labels)
    if len(labels) < tree.n_nodes:
        raise ValueError(
            f"labels has {len(labels)} elements but tree has {tree.n_nodes} nodes."
        )
    mask = labels[tree.edges[:, 0]] != labels[tree.edges[:, 1]]
    return np.where(mask)[0]


def confusion_matrix_from_tree(
    tree: Tree,
    labels: NDArray,
) -> tuple[NDArray[np.int64], list]:
    """Build a confusion matrix from tree edge topology.

    Counts how many tree edges connect each pair of label values.

    Parameters
    ----------
    tree : Tree
        Fitted tree.
    labels : array-like of shape (n_nodes,)
        Per-node labels.

    Returns
    -------
    cmat : NDArray[np.int64]
        Symmetric matrix of shape (n_classes, n_classes).
    classes : list
        Sorted unique label values (index into cmat rows/columns).
    """
    labels = np.asarray(labels)
    classes = sorted(set(labels.tolist()))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    cmat = np.zeros((n, n), dtype=np.int64)

    for src, tgt in tree.edges:
        i = cls_to_idx[labels[src]]
        j = cls_to_idx[labels[tgt]]
        cmat[i, j] += 1
        cmat[j, i] += 1

    return cmat, classes


def edge_delta(tree: Tree, values: NDArray) -> NDArray[np.float64]:
    """Compute absolute value change across each tree edge.

    Parameters
    ----------
    tree : Tree
        Fitted tree.
    values : array-like of shape (n_nodes,)
        Per-node continuous values.

    Returns
    -------
    NDArray[np.float64]
        Absolute deltas, shape (n_edges,).
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) < tree.n_nodes:
        raise ValueError(
            f"values has {len(values)} elements but tree has {tree.n_nodes} nodes."
        )
    return np.abs(values[tree.edges[:, 0]] - values[tree.edges[:, 1]])


def path_properties(
    tree: Tree,
    from_idx: int,
    to_idx: int,
    values: NDArray,
) -> NDArray:
    """Extract node values along the tree path between two points.

    Parameters
    ----------
    tree : Tree
        Fitted tree.
    from_idx : int
        Source node.
    to_idx : int
        Target node.
    values : array-like of shape (n_nodes,)
        Per-node values.

    Returns
    -------
    NDArray
        Ordered values along the path (inclusive of endpoints).
    """
    values = np.asarray(values)
    node_path = tree.path(from_idx, to_idx)
    return values[node_path]


def node_diversity(
    tree: Tree,
    method: str = "mean",
) -> NDArray[np.float64]:
    """Per-node diversity from MST edge weights.

    Nodes connected to distant neighbors (high edge weight) are diverse —
    they sit at transitions between clusters.  Nodes in tight clusters
    have low diversity.

    Parameters
    ----------
    tree : Tree
        Fitted tree.
    method : str, default ``"mean"``
        Aggregation over incident edge weights: ``"mean"``, ``"max"``,
        or ``"median"``.

    Returns
    -------
    NDArray[np.float64]
        Diversity score per node, shape ``(n_nodes,)``.
    """
    valid = {"mean", "max", "median"}
    if method not in valid:
        raise ValueError(f"method must be one of {sorted(valid)}, got {method!r}")

    diversity = np.zeros(tree.n_nodes, dtype=np.float64)
    for i in range(tree.n_nodes):
        weights = [w for _, w in tree._adjacency[i]]
        if weights:
            if method == "mean":
                diversity[i] = float(np.mean(weights))
            elif method == "max":
                diversity[i] = float(np.max(weights))
            else:  # median
                diversity[i] = float(np.median(weights))

    return diversity


def subtree_purity(
    tree: Tree,
    labels: NDArray,
    min_size: int = 10,
) -> NDArray[np.float64]:
    """Compute label purity for each node's subtree.

    Purity = (count of most common label) / (subtree size).
    Subtrees smaller than *min_size* get NaN.

    Parameters
    ----------
    tree : Tree
        Fitted tree.
    labels : array-like of shape (n_nodes,)
        Per-node labels.
    min_size : int, default 10
        Minimum subtree size to compute purity.

    Returns
    -------
    NDArray[np.float64]
        Purity per node, shape (n_nodes,). NaN for small subtrees.
    """
    labels = np.asarray(labels)
    n = tree.n_nodes

    # Build children list from BFS traversal
    children: list[list[int]] = [[] for _ in range(n)]
    order: list[int] = []
    visited = set()
    queue: deque[int] = deque([tree.root])
    visited.add(tree.root)

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor, _ in tree._adjacency[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                children[node].append(neighbor)
                queue.append(neighbor)

    # Post-order: accumulate label counts bottom-up
    # Use a dict per node for label counts
    label_counts: list[dict] = [{} for _ in range(n)]
    subtree_sizes = np.ones(n, dtype=np.int64)

    for node in reversed(order):
        lbl = labels[node]
        if isinstance(lbl, np.generic):
            lbl = lbl.item()
        label_counts[node][lbl] = label_counts[node].get(lbl, 0) + 1
        for child in children[node]:
            subtree_sizes[node] += subtree_sizes[child]
            for k, v in label_counts[child].items():
                label_counts[node][k] = label_counts[node].get(k, 0) + v

    purity = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if subtree_sizes[i] >= min_size:
            max_count = max(label_counts[i].values())
            purity[i] = max_count / subtree_sizes[i]

    return purity
