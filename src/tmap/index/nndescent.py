"""NNDescent-based approximate nearest neighbor index.

Wraps PyNNDescent (https://github.com/lmcinnes/pynndescent) for cosine/euclidean
kNN search. PyNNDescent is pure Python + Numba and supports 20+ distance metrics.

Requires: ``pip install pynndescent``
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tmap.index.base import Index
from tmap.index.types import KNNGraph

# Default build k — NNDescent builds a graph with this many neighbors eagerly.
# query_all can return up to _BUILD_K neighbors from the cached graph; beyond
# that it falls back to a full re-query.
_BUILD_K_DEFAULT = 30

# Default number of random projection trees.  PyNNDescent's auto default (~5)
# gives ~73% recall; 32 trees consistently yields >95%.
_N_TREES_DEFAULT = 32


class NNDescentIndex(Index):
    """Approximate nearest neighbor index using PyNNDescent.

    Parameters
    ----------
    seed : int or None
        Random seed for reproducibility.
    n_trees : int or None
        Number of random projection trees for initialization.
        Default is 32 (>95% recall). None uses PyNNDescent's auto default.
    n_iters : int or None
        Number of NNDescent iterations. None uses PyNNDescent's default.
    """

    def __init__(
        self,
        seed: int | None = None,
        n_trees: int | None = _N_TREES_DEFAULT,
        n_iters: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self._n_trees = n_trees
        self._n_iters = n_iters
        self._vectors: NDArray[np.float32] | None = None
        self._nnd_index: Any = None
        self._build_k: int = 0

    def _build_from_vectors(self, vectors: NDArray[np.float32], metric: str) -> None:
        from pynndescent import NNDescent

        self._vectors = np.asarray(vectors, dtype=np.float32)
        n = vectors.shape[0]
        self._build_k = min(_BUILD_K_DEFAULT, n - 1)

        kwargs: dict[str, Any] = {
            "n_neighbors": self._build_k + 1,  # +1 because NNDescent includes self
            "metric": metric,
        }
        if self._seed is not None:
            kwargs["random_state"] = self._seed
        if self._n_trees is not None:
            kwargs["n_trees"] = self._n_trees
        if self._n_iters is not None:
            kwargs["n_iters"] = self._n_iters

        self._nnd_index = NNDescent(self._vectors, **kwargs)
        self._nnd_index.prepare()

    def _query_all(self, k: int) -> KNNGraph:
        if k <= self._build_k:
            # Fast path: slice from the pre-built neighbor graph.
            # neighbor_graph includes self at position 0; strip it.
            indices, distances = self._nnd_index.neighbor_graph
            indices = indices[:, 1 : k + 1]
            distances = distances[:, 1 : k + 1]
        else:
            # k exceeds what we built — re-query with k+1 and strip self.
            indices, distances = self._nnd_index.query(self._vectors, k=k + 1)
            # Self is typically at column 0; strip vectorized.
            row_ids = np.arange(indices.shape[0])
            self_at_0 = indices[:, 0] == row_ids
            if self_at_0.all():
                indices = indices[:, 1:]
                distances = distances[:, 1:]
            else:
                # Slow fallback: per-row filter
                n = indices.shape[0]
                out_idx = np.empty((n, k), dtype=indices.dtype)
                out_dist = np.empty((n, k), dtype=distances.dtype)
                for i in range(n):
                    mask = indices[i] != i
                    out_idx[i] = indices[i][mask][:k]
                    out_dist[i] = distances[i][mask][:k]
                indices = out_idx
                distances = out_dist

        return KNNGraph.from_arrays(
            indices.astype(np.int32),
            distances.astype(np.float32),
        )

    def _query_single(
        self,
        point: NDArray[np.float32],
        k: int,
    ) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        indices, distances = self._nnd_index.query(point.reshape(1, -1), k=k)
        return (
            indices[0].astype(np.int32),
            distances[0].astype(np.float32),
        )

    def _save_implementation(self, path: Path) -> None:
        data = {
            "vectors": self._vectors,
            "metric": self._metric,
            "n_trees": self._n_trees,
            "n_iters": self._n_iters,
            "build_k": self._build_k,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def _load_implementation(self, path: Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._vectors = data["vectors"]
        self._metric = data.get("metric")
        self._n_trees = data["n_trees"]
        self._n_iters = data["n_iters"]
        self._build_k = data.get("build_k", _BUILD_K_DEFAULT)

        # Rebuild the NNDescent index so query_point works after load.
        from pynndescent import NNDescent

        kwargs: dict[str, Any] = {
            "n_neighbors": self._build_k + 1,
            "metric": self._metric,
        }
        if self._seed is not None:
            kwargs["random_state"] = self._seed
        if self._n_trees is not None:
            kwargs["n_trees"] = self._n_trees
        if self._n_iters is not None:
            kwargs["n_iters"] = self._n_iters

        self._nnd_index = NNDescent(self._vectors, **kwargs)
        self._nnd_index.prepare()
