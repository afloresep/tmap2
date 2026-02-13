"""
Type definitions for the index module.
"""

from __future__ import annotations

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

    @classmethod
    def from_arrays(
        cls,
        indices: NDArray[np.int32] | list[list[int]],
        distances: NDArray[np.float32] | list[list[float]],
    ) -> KNNGraph:
        """Build a KNNGraph from raw neighbor index and distance arrays."""
        indices_arr = np.asarray(indices, dtype=np.int32)
        distances_arr = np.asarray(distances, dtype=np.float32)

        if indices_arr.ndim != 2 or distances_arr.ndim != 2:
            raise ValueError(
                "indices and distances must be 2D arrays with shape (n_nodes, k)."
            )
        if indices_arr.shape != distances_arr.shape:
            raise ValueError(
                f"indices and distances must have identical shapes. "
                f"Got {indices_arr.shape} and {distances_arr.shape}."
            )
        if indices_arr.shape[0] < 1:
            raise ValueError("KNN arrays must contain at least one node.")
        if indices_arr.shape[1] < 1:
            raise ValueError("KNN arrays must contain at least one neighbor per node.")

        return cls(indices=indices_arr, distances=distances_arr)

    @classmethod
    def from_distance_matrix(
        cls,
        distance_matrix: NDArray[np.float32] | list[list[float]],
        k: int,
    ) -> KNNGraph:
        """Convert a dense distance matrix to KNNGraph."""
        distances = np.asarray(distance_matrix, dtype=np.float32)

        if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
            raise ValueError(
                "distance_matrix must be square with shape (n_samples, n_samples)."
            )
        n_samples = distances.shape[0]
        if n_samples < 2:
            raise ValueError("distance_matrix must contain at least 2 samples.")
        if not 1 <= k < n_samples:
            raise ValueError(f"k must satisfy 1 <= k < n_samples ({n_samples}), got {k}.")
        if not np.all(np.isfinite(distances)):
            raise ValueError("distance_matrix must contain only finite values.")

        # Exclude self-neighbors by setting diagonal to +inf during ranking.
        rank_matrix = distances.copy()
        np.fill_diagonal(rank_matrix, np.inf)

        indices = np.argsort(rank_matrix, axis=1)[:, :k].astype(np.int32)
        knn_distances = np.take_along_axis(rank_matrix, indices.astype(np.intp), axis=1)
        return cls(indices=indices, distances=knn_distances.astype(np.float32, copy=False))

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
