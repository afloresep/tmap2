"""
Type definitions for the index module.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class Edge(NamedTuple):
    """Single edge: (source_node, target_node, weight/distance)."""

    source: int
    target: int
    weight: float


@dataclass(frozen=True, slots=True)
class EdgeList:
    """
    Generic input format - just a list of weighted edges.

    This is how tmap stays generic: users can provide ANY data
    as edges. Molecule similarities, paper citations, image distances, etc.

    frozen=True: Makes instances immutable (hashable, safer)
    slots=True: Memory optimization (Python 3.10+)

    Example usage:
        edges = EdgeList(
            edges=[(0, 1, 0.9), (0, 2, 0.7), (1, 2, 0.5)],
            n_nodes=3,
        )
    """

    edges: list[Edge] | list[tuple[int, int, float]]
    n_nodes: int
    labels: list[str] | None = None  # Optional node labels

    def __post_init__(self) -> None:
        """Validate on construction. Fail fast, fail loud."""
        if self.n_nodes <= 0:
            raise ValueError(f"n_nodes must be positive, got {self.n_nodes}")
        # Convert tuples to Edge namedtuples for consistency
        if self.edges and isinstance(self.edges[0], tuple):
            object.__setattr__(self, "edges", [Edge(*e) for e in self.edges])


@dataclass(slots=True)
class KNNGraph:
    """
    k-Nearest Neighbor graph as sparse arrays.

    This is the OUTPUT of the Index stage and INPUT to Graph stage.

    Storage format (CSR-like, memory efficient):
        indices[i, j] = index of j-th nearest neighbor of node i
        distances[i, j] = distance to that neighbor

    Shape: (n_nodes, k) for both arrays

    WHY NOT scipy.sparse?
    - We always have exactly k neighbors per node (dense in that sense)
    - These arrays are simpler and faster for our use case
    - Easy to convert to scipy.sparse if needed later
    """

    indices: NDArray[np.int32]  # Shape: (n_nodes, k)
    distances: NDArray[np.float32]  # Shape: (n_nodes, k)

    @property
    def n_nodes(self) -> int:
        """Return the number of nodes in the graph."""
        return int(self.indices.shape[0])

    @property
    def k(self) -> int:
        """Return the number of neighbors per node."""
        return int(self.indices.shape[1])

    def to_edge_list(self) -> EdgeList:
        """Convert to EdgeList format (useful for debugging/export)."""
        edges: list[Edge] = []
        for i in range(self.n_nodes):
            for j in range(self.k):
                neighbor = self.indices[i, j]
                dist = self.distances[i, j]
                if neighbor != -1:  # -1 means no neighbor (sparse)
                    edges.append(Edge(i, int(neighbor), float(dist)))
        return EdgeList(edges=edges, n_nodes=self.n_nodes)
