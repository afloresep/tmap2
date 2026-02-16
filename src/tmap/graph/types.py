"""
Type definitions for the graph module.
A tree with N nodes has exactly N-1 edges.
We store it as edge list + adjacency for efficient traversal.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


"""idea is to have the tree class to support DFS, BFS etc. probably not worht keeping """

@dataclass(slots=True)
class Tree:
    """
    Tree structure (MST result).

    Attributes:
        n_nodes: Number of nodes
        edges: Array of (source, target) pairs, shape (n_nodes-1, 2)
        weights: Edge weights, shape (n_nodes-1,)
        root: Root node index (usually 0 or node with highest degree)

    The tree is stored edge-list style but also builds adjacency
    for efficient traversal.

    NOTE: use arrays, not Python lists
    - Faster for numerical operations
    - Less memory for large trees
    - Easy to serialize (np.save)
    """

    n_nodes: int
    edges: NDArray[np.int32]  # Shape: (n_edges, 2) where n_edges = n_nodes - 1
    weights: NDArray[np.float32]  # Shape: (n_edges,)
    root: int = 0

    _adjacency: dict[int, list[tuple[int, float]]] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Build adjacency list for traversal."""
        self._build_adjacency()

    def _build_adjacency(self) -> None:
        """
        Build adjacency list from edges.

        Adjacency maps: node -> [(neighbor, weight), ...]
        Undirected: each edge appears in both directions.
        """
        self._adjacency = {i: [] for i in range(self.n_nodes)}
        for i, (src, tgt) in enumerate(self.edges):
            w = self.weights[i]
            self._adjacency[int(src)].append((int(tgt), float(w)))
            self._adjacency[int(tgt)].append((int(src), float(w)))

    def neighbors(self, node: int) -> list[tuple[int, float]]:
        """Get neighbors of a node with their edge weights."""
        return self._adjacency[node]

    def children(self, node: int, parent: int | None = None) -> list[int]:
        """
        Get children of a node (neighbors excluding parent).

        For tree traversal from root downward.
        """
        return [neighbor for neighbor, _ in self._adjacency[node] if neighbor != parent]

    def bfs(self, start: int | None = None) -> Iterator[tuple[int, int | None, int]]:
        """
        Breadth-first traversal from start (default: root).

        Yields: (node, parent, depth)

        Useful for:
        - Level-by-level processing
        - Finding all nodes at depth D
        - Layout algorithms that work top-down
        """
        start = start if start is not None else self.root
        visited: set[int] = {start}
        queue: list[tuple[int, int | None, int]] = [(start, None, 0)]  # (node, parent, depth)

        while queue:
            node, parent, depth = queue.pop(0)
            yield node, parent, depth

            for neighbor, _ in self._adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, node, depth + 1))

    def dfs(self, start: int | None = None) -> Iterator[tuple[int, int | None, int]]:
        """
        Depth-first traversal from start (default: root).

        Yields: (node, parent, depth)

        Useful for:
        - Subtree operations
        - Post-order processing (children before parent)
        """
        start = start if start is not None else self.root
        visited: set[int] = {start}
        stack: list[tuple[int, int | None, int]] = [(start, None, 0)]

        while stack:
            node, parent, depth = stack.pop()
            yield node, parent, depth

            for neighbor, _ in reversed(self._adjacency[node]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append((neighbor, node, depth + 1))

    def subtree_sizes(self) -> NDArray[np.int32]:
        """
        Compute size of subtree rooted at each node.

        Returns array where result[i] = number of nodes in subtree rooted at i.
        Useful for layout algorithms that need to allocate space.
        """
        sizes: NDArray[np.int32] = np.ones(self.n_nodes, dtype=np.int32)

        # Process in reverse BFS order (leaves first)
        traversal = list(self.bfs())
        for node, parent, _ in reversed(traversal):
            if parent is not None:
                sizes[parent] += sizes[node]

        return sizes
