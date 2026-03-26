"""USearch-based nearest neighbor index.

Wraps USearch (https://github.com/unum-cloud/usearch) for cosine, euclidean,
and Jaccard (binary) kNN search.

Auto mode: exact brute-force for n < 50 000, HNSW for n >= 50 000.
Binary Jaccard always uses HNSW.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tmap.index.types import KNNGraph

_HNSW_THRESHOLD = 50_000
_META_SUFFIX = ".meta"
_VECTORS_SUFFIX = ".vectors.npy"

_METRIC_MAP = {
    "cosine": "cos",
    "euclidean": "l2sq",
}


class USearchIndex:
    """Nearest-neighbor index using USearch.

    Supports cosine, euclidean (dense float vectors), and Jaccard (binary
    0/1 vectors). In ``auto`` mode it uses exact search for small datasets
    and HNSW for larger ones. Binary Jaccard always uses HNSW.

    Args:
        seed: Stored as metadata. USearch itself does not use this value.
        mode: ``"auto"``, ``"exact"``, or ``"hnsw"``.
        connectivity: HNSW connectivity parameter.
        expansion_add: HNSW build depth.
        expansion_search: HNSW search depth.
    """

    def __init__(
        self,
        seed: int | None = None,
        mode: str = "auto",
        connectivity: int = 32,
        expansion_add: int = 256,
        expansion_search: int = 200,
    ) -> None:
        if mode not in {"auto", "exact", "hnsw"}:
            raise ValueError(f"mode must be auto/exact/hnsw, got {mode!r}")
        self._seed = seed
        self._mode = mode
        self._connectivity = connectivity
        self._expansion_add = expansion_add
        self._expansion_search = expansion_search
        self._effective_mode: str | None = None
        self._vectors: NDArray[np.float32] | None = None
        self._binary_vectors: NDArray[np.uint8] | None = None
        self._is_binary: bool = False
        self._usearch_index: Any | None = None  # usearch.index.Index
        self._is_built = False
        self._n_nodes: int = 0
        self._ndim: int = 0
        self._metric: str | None = None

    # -- properties --

    @property
    def is_built(self) -> bool:
        return self._is_built

    @property
    def n_nodes(self) -> int:
        return self._n_nodes

    @property
    def metric(self) -> str | None:
        return self._metric

    @property
    def effective_mode(self) -> str | None:
        return self._effective_mode

    # -- build --

    def build_from_vectors(
        self,
        vectors: NDArray[np.float32],
        metric: str = "euclidean",
    ) -> USearchIndex:
        """Build the index from a matrix of dense vectors.

        Args:
            vectors: Data matrix of shape ``(n_samples, n_features)``.
            metric: ``"euclidean"`` or ``"cosine"``.

        Returns:
            The built index.
        """
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2D, got shape {vectors.shape}")
        if vectors.shape[0] < 2:
            raise ValueError("Need at least 2 vectors to build index")
        if metric not in _METRIC_MAP:
            raise ValueError(
                f"USearchIndex does not support metric={metric!r}. Supported: {list(_METRIC_MAP)}"
            )

        from usearch.index import Index

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        n, d = vectors.shape

        if self._mode == "auto":
            effective = "exact" if n < _HNSW_THRESHOLD else "hnsw"
        else:
            effective = self._mode
        self._effective_mode = effective

        us_metric = _METRIC_MAP[metric]

        if effective == "hnsw":
            idx = Index(
                ndim=d,
                metric=us_metric,
                dtype="f32",
                connectivity=self._connectivity,
                expansion_add=self._expansion_add,
                expansion_search=self._expansion_search,
            )
            keys = np.arange(n, dtype=np.int64)
            idx.add(keys, vectors)
            self._usearch_index = idx

        # For exact mode we skip HNSW build entirely; queries use
        # the module-level ``usearch.index.search`` function on raw vectors.
        self._vectors = vectors
        self._binary_vectors = None
        self._is_binary = False
        self._n_nodes = n
        self._ndim = d
        self._metric = metric
        self._is_built = True
        return self

    def build_from_binary(
        self,
        matrix: NDArray,
    ) -> USearchIndex:
        """Build a Jaccard index from a binary (0/1) matrix.

        Packs the binary matrix to bytes and builds an HNSW index using
        USearch's native bit-vector Jaccard distance. No MinHash encoding
        is needed — distances are computed on the raw bits.

        Args:
            matrix: Binary matrix of shape ``(n_samples, n_features)``
                with 0/1 values.

        Returns:
            The built index.
        """
        matrix = np.asarray(matrix)
        if matrix.ndim != 2:
            raise ValueError(f"matrix must be 2D, got shape {matrix.shape}")
        if matrix.shape[0] < 2:
            raise ValueError("Need at least 2 vectors to build index")
        if not np.all((matrix == 0) | (matrix == 1)):
            raise ValueError(
                "Binary matrix must contain only 0/1 values."
            )

        from usearch.index import Index, MetricKind

        n, d = matrix.shape
        packed = np.packbits(matrix.astype(np.uint8, copy=False), axis=1)
        packed = np.ascontiguousarray(packed)

        idx = Index(
            ndim=d,
            metric=MetricKind.Jaccard,
            dtype="b1x8",
            connectivity=self._connectivity,
            expansion_add=self._expansion_add,
            expansion_search=self._expansion_search,
        )
        keys = np.arange(n, dtype=np.int64)
        idx.add(keys, packed)

        self._usearch_index = idx
        self._binary_vectors = packed
        self._vectors = None
        self._is_binary = True
        self._effective_mode = "hnsw"
        self._n_nodes = n
        self._ndim = d  # number of bits (original feature dimension)
        self._metric = "jaccard"
        self._is_built = True
        return self

    # -- kNN graph (all-vs-all) --

    def add(self, vectors: NDArray) -> NDArray[np.int64]:
        """Append new vectors to an existing index.

        For binary indices, pass a 0/1 matrix with the same number of
        features as ``build_from_binary()``. For dense indices, pass a
        float matrix matching ``build_from_vectors()``.

        Args:
            vectors: New vectors with the same number of features.

        Returns:
            Keys assigned to the new vectors.
        """
        self._check_is_built()

        vectors = np.asarray(vectors)
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2D, got shape {vectors.shape}")

        if self._is_binary:
            if vectors.shape[1] != self._ndim:
                raise ValueError(
                    f"vectors must have {self._ndim} features, got {vectors.shape[1]}"
                )
            if vectors.shape[0] == 0:
                return np.empty(0, dtype=np.int64)
            if not np.all((vectors == 0) | (vectors == 1)):
                raise ValueError("Binary vectors must contain only 0/1 values.")

            packed = np.packbits(vectors.astype(np.uint8, copy=False), axis=1)
            packed = np.ascontiguousarray(packed)

            start = self._n_nodes
            keys = np.arange(start, start + packed.shape[0], dtype=np.int64)

            if self._usearch_index is None:
                raise RuntimeError("HNSW index not available")
            self._usearch_index.add(keys, packed)

            if self._binary_vectors is not None:
                self._binary_vectors = np.concatenate(
                    [self._binary_vectors, packed], axis=0
                )

            self._n_nodes += packed.shape[0]
            return keys

        # Dense float path
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.shape[1] != self._ndim:
            raise ValueError(
                f"vectors must have {self._ndim} features, got {vectors.shape[1]}"
            )
        if vectors.shape[0] == 0:
            return np.empty(0, dtype=np.int64)

        start = self._n_nodes
        keys = np.arange(start, start + vectors.shape[0], dtype=np.int64)

        if self._effective_mode == "exact":
            if self._vectors is None:
                raise RuntimeError("Exact index requires stored vectors")
        else:
            if self._usearch_index is None:
                raise RuntimeError("HNSW index not available")
            self._usearch_index.add(keys, vectors)

        if self._vectors is not None:
            self._vectors = np.concatenate([self._vectors, vectors], axis=0)

        self._n_nodes += vectors.shape[0]
        return keys

    def query_knn(self, k: int) -> KNNGraph:
        """Return the k-nearest-neighbor graph for all indexed points.

        Args:
            k: Number of neighbors per point.

        Returns:
            A ``KNNGraph`` with neighbor indices and distances.
        """
        self._check_is_built()
        if k >= self._n_nodes:
            raise ValueError(f"k={k} must be < n_nodes={self._n_nodes}")

        if self._is_binary:
            if self._binary_vectors is None:
                raise RuntimeError("No binary vectors stored for query_knn().")
            queries = self._binary_vectors
        else:
            if self._vectors is None:
                raise RuntimeError(
                    "Cannot call query_knn() on a loaded index: the original "
                    "vectors were not saved.  Rebuild with build_from_vectors() "
                    "or use query_point()/query_batch() instead."
                )
            queries = self._vectors

        keys, dists = self._search(queries, k + 1)
        keys, dists = self._strip_self(keys, dists, k)
        dists = self._convert_distances(dists)
        return KNNGraph.from_arrays(
            self._safe_int32(keys),
            dists.astype(np.float32),
        )

    # -- single / batch queries --

    def query_point(
        self,
        point: NDArray,
        k: int,
    ) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        """Query neighbors for one point.

        Args:
            point: Query vector (binary 0/1 or dense float, matching the
                index type).
            k: Number of neighbors to return.

        Returns:
            Tuple ``(indices, distances)``.
        """
        self._check_is_built()
        query = self._prepare_query(np.asarray(point).reshape(1, -1))
        keys, dists = self._search(query, k)
        dists = self._convert_distances(dists)
        return self._safe_int32(keys[0]), dists[0].astype(np.float32)

    def query_batch(
        self,
        points: NDArray,
        k: int,
    ) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        """Query neighbors for many points at once.

        Args:
            points: Query matrix (binary 0/1 or dense float, matching the
                index type).
            k: Number of neighbors to return per point.

        Returns:
            Tuple ``(indices, distances)``.
        """
        self._check_is_built()
        queries = self._prepare_query(np.asarray(points))
        keys, dists = self._search(queries, k)
        dists = self._convert_distances(dists)
        return self._safe_int32(keys), dists.astype(np.float32)

    # -- persistence --

    def save(self, path: str | Path) -> None:
        """Save the index to disk.

        Args:
            path: Output file path.
        """
        self._check_is_built()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        meta = self._make_meta()
        if self._effective_mode == "exact":
            if self._vectors is None:
                raise RuntimeError("Exact index cannot be saved without stored vectors.")
            np.save(self._vectors_path(path), self._vectors, allow_pickle=False)
            meta["has_vectors"] = True
        else:
            if self._usearch_index is None:
                raise RuntimeError("HNSW index cannot be saved before it is built.")
            self._usearch_index.save(str(path))
            if self._is_binary and self._binary_vectors is not None:
                np.save(self._vectors_path(path), self._binary_vectors, allow_pickle=False)
            meta["has_vectors"] = self._is_binary
        with open(self._meta_path(path), "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, path: str | Path) -> USearchIndex:
        """Load an index saved with ``save()``.

        Args:
            path: File path written by ``save()``.

        Returns:
            The restored index.
        """
        from usearch.index import Index

        path = Path(path)
        with open(cls._meta_path(path), "rb") as f:
            meta = pickle.load(f)

        instance = cls(
            seed=meta.get("seed"),
            mode=meta.get("mode", "auto"),
            connectivity=meta.get("connectivity", 32),
            expansion_add=meta.get("expansion_add", 256),
            expansion_search=meta.get("expansion_search", 200),
        )
        is_binary = meta.get("is_binary", False)
        instance._is_binary = is_binary

        if meta.get("effective_mode") == "hnsw":
            if not path.exists():
                raise FileNotFoundError(f"USearch index file not found: {path}")
            instance._usearch_index = Index.restore(str(path))
            if is_binary and meta.get("has_vectors"):
                vectors_path = cls._vectors_path(path)
                if not vectors_path.exists():
                    raise FileNotFoundError(
                        f"Binary vectors file not found: {vectors_path}. "
                        f"Cannot restore binary index without stored vectors."
                    )
                instance._binary_vectors = np.ascontiguousarray(
                    np.load(vectors_path, allow_pickle=False)
                )
        elif meta.get("has_vectors"):
            vectors_path = cls._vectors_path(path)
            if not vectors_path.exists():
                raise FileNotFoundError(f"USearch vectors file not found: {vectors_path}")
            instance._vectors = np.ascontiguousarray(
                np.load(vectors_path, allow_pickle=False)
            )
        instance._n_nodes = meta.get("n_nodes", 0)
        instance._ndim = meta.get("ndim", 0)
        instance._metric = meta.get("metric")
        instance._effective_mode = meta.get("effective_mode")
        instance._is_built = True
        return instance

    # -- pickle support (for TMAP.save which uses pickle.dump) --

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        idx = state.pop("_usearch_index")
        if idx is not None:
            state["_usearch_index_bytes"] = bytes(idx.save())
            # For binary, keep _binary_vectors; for dense, drop _vectors
            # (they're redundant with the HNSW index for dense).
            if not state.get("_is_binary"):
                state["_vectors"] = None
        else:
            state["_usearch_index_bytes"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        buf = state.pop("_usearch_index_bytes")
        # Handle loading from older pickles that lack binary fields
        state.setdefault("_is_binary", False)
        state.setdefault("_binary_vectors", None)
        self.__dict__.update(state)
        if buf is not None:
            from usearch.index import Index

            self._usearch_index = Index.restore(buf)
        else:
            self._usearch_index = None

    # -- internals --

    def _check_is_built(self) -> None:
        if not self._is_built:
            raise RuntimeError("Index not built. Call build_from_vectors() first.")

    def _prepare_query(self, data: NDArray) -> NDArray:
        """Convert user-supplied query data to the format USearch expects."""
        if self._is_binary:
            expected = self._ndim
            actual = data.shape[-1] if data.ndim >= 1 else 0
            if actual != expected:
                raise ValueError(
                    f"Query has {actual} features but index was built "
                    f"with {expected}."
                )
            if not np.all((data == 0) | (data == 1)):
                raise ValueError("Query data must contain only 0/1 values.")
            packed = np.packbits(data.astype(np.uint8, copy=False), axis=1)
            return np.ascontiguousarray(packed)
        return np.ascontiguousarray(data, dtype=np.float32)

    def _search(
        self,
        queries: NDArray,
        count: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Run search and return (keys, distances) as 2D arrays."""
        actual_count = min(count, self._n_nodes)
        if self._effective_mode == "exact":
            # No HNSW built — use module-level exact search on raw vectors.
            from usearch.index import MetricKind, search

            us_metric = getattr(
                MetricKind,
                {
                    "cos": "Cos",
                    "l2sq": "L2sq",
                }[_METRIC_MAP[self._metric]],
            )

            if self._vectors is None:
                raise RuntimeError("Exact search requires stored vectors")
            results = search(
                self._vectors,
                queries,
                actual_count,
                metric=us_metric,
                exact=True,
            )
        else:
            if self._usearch_index is None:
                raise RuntimeError("HNSW index not available")
            results = self._usearch_index.search(queries, actual_count)

        keys = np.asarray(results.keys, dtype=np.int64)
        dists = np.asarray(results.distances, dtype=np.float32)
        # Single query returns 1D; ensure 2D.
        if keys.ndim == 1:
            keys = keys.reshape(1, -1)
            dists = dists.reshape(1, -1)
        return keys, dists

    def _strip_self(
        self,
        keys: NDArray[np.int64],
        dists: NDArray[np.float32],
        k: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Remove self-matches from all-vs-all query results."""
        n = keys.shape[0]
        row_ids = np.arange(n, dtype=np.int64)
        if (keys[:, 0] == row_ids).all():
            return keys[:, 1:], dists[:, 1:]
        # Self may not be at position 0 (HNSW or duplicate vectors)
        out_keys = np.empty((n, k), dtype=np.int64)
        out_dists = np.empty((n, k), dtype=np.float32)
        for i in range(n):
            mask = keys[i] != i
            out_keys[i] = keys[i][mask][:k]
            out_dists[i] = dists[i][mask][:k]
        return out_keys, out_dists

    def _convert_distances(self, dists: NDArray[np.float32]) -> NDArray[np.float32]:
        """Convert raw USearch distances to user-facing metric."""
        # Cosine and Jaccard: already returns the correct distance.
        if self._metric == "euclidean":
            # USearch 'l2sq' returns squared L2.
            np.maximum(dists, 0, out=dists)
            np.sqrt(dists, out=dists)
        return dists

    @staticmethod
    def _safe_int32(keys: NDArray[np.int64]) -> NDArray[np.int32]:
        """Cast int64 keys to int32 with overflow guard."""
        if keys.size > 0 and keys.max() > np.iinfo(np.int32).max:
            raise OverflowError(
                f"Index key {keys.max()} exceeds int32 range. KNNGraph requires int32 indices."
            )
        return keys.astype(np.int32)

    def _make_meta(self) -> dict:
        return {
            "backend": "usearch",
            "n_nodes": self._n_nodes,
            "ndim": self._ndim,
            "seed": self._seed,
            "metric": self._metric,
            "mode": self._mode,
            "effective_mode": self._effective_mode,
            "connectivity": self._connectivity,
            "expansion_add": self._expansion_add,
            "expansion_search": self._expansion_search,
            "is_binary": self._is_binary,
        }

    @staticmethod
    def _meta_path(path: Path) -> Path:
        return Path(str(path) + _META_SUFFIX)

    @staticmethod
    def _vectors_path(path: Path) -> Path:
        return Path(str(path) + _VECTORS_SUFFIX)
