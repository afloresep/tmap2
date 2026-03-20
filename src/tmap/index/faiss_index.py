"""FAISS-based nearest neighbor index.

Wraps Facebook AI Similarity Search (https://github.com/facebookresearch/faiss)
for cosine/euclidean kNN search.

Auto mode: flat (exact) for n < 50k, HNSW (approximate) for n >= 50k.

Requires: ``pip install faiss-cpu``
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tmap.index.types import KNNGraph

_HNSW_THRESHOLD = 50_000


class FaissIndex:
    """Nearest neighbor index using FAISS.

    Parameters
    ----------
    seed : int or None
        Random seed.
    mode : str
        ``"auto"`` (default): flat for n < 50k, hnsw for n >= 50k.
        ``"flat"``: Exact brute-force search.
        ``"hnsw"``: Hierarchical Navigable Small World graph.
    hnsw_m : int
        Connections per node in the HNSW graph. Default 32.
    hnsw_ef_construction : int
        Search depth during HNSW build. Default 40.
    hnsw_ef_search : int
        Search depth during HNSW query. Default 64.
    """

    def __init__(
        self,
        seed: int | None = None,
        mode: str = "auto",
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 40,
        hnsw_ef_search: int = 64,
    ) -> None:
        if mode not in {"auto", "flat", "hnsw"}:
            raise ValueError(f"mode must be auto/flat/hnsw, got {mode!r}")
        self._seed = seed
        self._mode = mode
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construction = hnsw_ef_construction
        self._hnsw_ef_search = hnsw_ef_search
        self._effective_mode: str | None = None
        self._vectors: NDArray[np.float32] | None = None
        self._faiss_index = None
        self._is_built = False
        self._n_nodes: int = 0
        self._metric: str | None = None

    @property
    def is_built(self) -> bool:
        """Whether the index has been built and is ready for queries."""
        return self._is_built

    @property
    def n_nodes(self) -> int:
        """Number of points in the index."""
        return self._n_nodes

    @property
    def metric(self) -> str | None:
        """Distance metric used during build, or None if not yet built."""
        return self._metric

    @property
    def effective_mode(self) -> str | None:
        """The actual index mode after building (None if not yet built)."""
        return self._effective_mode

    def build_from_vectors(
        self,
        vectors: NDArray[np.float32],
        metric: str = "euclidean",
    ) -> FaissIndex:
        """Build index from raw vectors.

        Parameters
        ----------
        vectors : ndarray of shape (n_samples, n_features)
            Data to index.
        metric : str
            ``"euclidean"`` or ``"cosine"``.

        Returns
        -------
        self
        """
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2D, got shape {vectors.shape}")
        if vectors.shape[0] < 2:
            raise ValueError("Need at least 2 vectors to build index")

        import faiss

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        n, d = vectors.shape

        if metric == "cosine":
            faiss.normalize_L2(vectors)
            metric_type = faiss.METRIC_INNER_PRODUCT
        elif metric == "euclidean":
            metric_type = faiss.METRIC_L2
        else:
            raise ValueError(f"FaissIndex does not support metric={metric!r}")

        if self._mode == "auto":
            effective = "flat" if n < _HNSW_THRESHOLD else "hnsw"
        else:
            effective = self._mode
        self._effective_mode = effective

        if effective == "flat":
            index = self._build_flat(d, metric_type)
            index.add(vectors)
        else:
            index = self._build_hnsw(vectors, d, metric_type)

        self._faiss_index = index
        self._vectors = vectors
        self._n_nodes = n
        self._metric = metric
        self._is_built = True
        return self

    def query_knn(self, k: int) -> KNNGraph:
        """Get k-nearest neighbors for all points in the index.

        Parameters
        ----------
        k : int
            Number of neighbors per point.

        Returns
        -------
        KNNGraph
        """
        self._check_is_built()
        if k >= self._n_nodes:
            raise ValueError(f"k={k} must be < n_nodes={self._n_nodes}")
        return self._query_all(k)

    def query_point(
        self,
        point: NDArray[np.float32],
        k: int,
    ) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        """Query k-nearest neighbors for a single new point.

        Returns
        -------
        (indices, distances)
            Each of shape ``(k,)``.
        """
        self._check_is_built()
        import faiss

        query = np.ascontiguousarray(point.reshape(1, -1), dtype=np.float32)
        if self._metric == "cosine":
            faiss.normalize_L2(query)

        distances, indices = self._faiss_index.search(query, k)

        dists = distances[0].copy()
        if self._metric == "cosine":
            dists = 1.0 - dists
        elif self._metric == "euclidean":
            np.maximum(dists, 0, out=dists)
            np.sqrt(dists, out=dists)

        return indices[0].astype(np.int32), dists.astype(np.float32)

    def query_batch(
        self,
        points: NDArray[np.float32],
        k: int,
    ) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        """Query k-nearest neighbors for a batch of new points.

        Returns
        -------
        (indices, distances)
            Each of shape ``(m, k)``.
        """
        self._check_is_built()
        import faiss

        queries = np.ascontiguousarray(points, dtype=np.float32)
        if self._metric == "cosine":
            faiss.normalize_L2(queries)

        distances, indices = self._faiss_index.search(queries, k)

        dists = distances.copy()
        if self._metric == "cosine":
            dists = 1.0 - dists
        elif self._metric == "euclidean":
            np.maximum(dists, 0, out=dists)
            np.sqrt(dists, out=dists)

        return indices.astype(np.int32), dists.astype(np.float32)

    def save(self, path: str | Path) -> None:
        """Save index to disk."""
        self._check_is_built()
        import faiss

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._faiss_index, str(path))

        meta = {
            "n_nodes": self._n_nodes,
            "seed": self._seed,
            "metric": self._metric,
            "mode": self._mode,
            "effective_mode": self._effective_mode,
            "hnsw_m": self._hnsw_m,
            "hnsw_ef_construction": self._hnsw_ef_construction,
            "hnsw_ef_search": self._hnsw_ef_search,
        }
        with open(str(path) + ".meta", "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, path: str | Path) -> FaissIndex:
        """Load a previously saved index from disk."""
        import faiss

        path = Path(path)
        with open(str(path) + ".meta", "rb") as f:
            meta = pickle.load(f)

        instance = cls(
            seed=meta.get("seed"),
            mode=meta.get("mode", "auto"),
            hnsw_m=meta.get("hnsw_m", 32),
            hnsw_ef_construction=meta.get("hnsw_ef_construction", 40),
            hnsw_ef_search=meta.get("hnsw_ef_search", 64),
        )
        instance._faiss_index = faiss.read_index(str(path))
        instance._n_nodes = meta.get("n_nodes", 0)
        instance._metric = meta.get("metric")
        instance._effective_mode = meta.get("effective_mode")
        instance._is_built = True
        instance._vectors = None
        return instance

    # -- internals --

    def _check_is_built(self) -> None:
        if not self._is_built:
            raise RuntimeError("Index not built. Call build_from_vectors() first.")

    def _build_flat(self, d: int, metric_type: int) -> Any:
        import faiss

        if metric_type == faiss.METRIC_INNER_PRODUCT:
            return faiss.IndexFlatIP(d)
        return faiss.IndexFlatL2(d)

    def _build_hnsw(
        self,
        vectors: NDArray[np.float32],
        d: int,
        metric_type: int,
    ) -> Any:
        import faiss

        index = faiss.IndexHNSWFlat(d, self._hnsw_m, metric_type)
        index.hnsw.efConstruction = self._hnsw_ef_construction
        index.hnsw.efSearch = self._hnsw_ef_search
        index.add(vectors)
        return index

    def _query_all(self, k: int) -> KNNGraph:
        if self._vectors is None:
            raise RuntimeError(
                "Cannot call query_knn() on a loaded index: the original "
                "vectors were not saved. Rebuild the index with "
                "build_from_vectors() or use query_point()/query_batch() instead."
            )
        n = self._vectors.shape[0]
        distances, indices = self._faiss_index.search(self._vectors, k + 1)

        if self._effective_mode == "flat":
            indices = indices[:, 1:]
            distances = distances[:, 1:]
        else:
            # HNSW: self may not be at position 0
            row_ids = np.arange(n)
            if (indices[:, 0] == row_ids).all():
                indices = indices[:, 1:]
                distances = distances[:, 1:]
            else:
                out_idx = np.empty((n, k), dtype=indices.dtype)
                out_dist = np.empty((n, k), dtype=distances.dtype)
                for i in range(n):
                    mask = indices[i] != i
                    valid_idx = indices[i][mask]
                    valid_dist = distances[i][mask]
                    out_idx[i] = valid_idx[:k]
                    out_dist[i] = valid_dist[:k]
                indices = out_idx
                distances = out_dist

        if self._metric == "cosine":
            distances = 1.0 - distances
        elif self._metric == "euclidean":
            np.maximum(distances, 0, out=distances)
            np.sqrt(distances, out=distances)

        return KNNGraph.from_arrays(
            indices.astype(np.int32),
            distances.astype(np.float32),
        )
