"""
Type definitions for the index module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class KNNGraph:
    """
    k-Nearest Neighbor graph as sparse arrays.

    Storage format (CSR-like, memory efficient):
        indices[i, j] = index of j-th nearest neighbor of node i
        distances[i, j] = distance to that neighbor

    Shape: (n_nodes, k) for both arrays
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
            raise ValueError("indices and distances must be 2D arrays with shape (n_nodes, k).")
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
            raise ValueError("distance_matrix must be square with shape (n_samples, n_samples).")
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
