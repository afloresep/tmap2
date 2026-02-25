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

from tmap.index.base import Index
from tmap.index.types import KNNGraph

_HNSW_THRESHOLD = 50_000


class FaissIndex(Index):
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
        super().__init__(seed=seed)
        if mode not in {"auto", "flat", "hnsw"}:
            raise ValueError(f"mode must be auto/flat/hnsw, got {mode!r}")
        self._mode = mode
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construction = hnsw_ef_construction
        self._hnsw_ef_search = hnsw_ef_search
        self._effective_mode: str | None = None
        self._vectors: NDArray[np.float32] | None = None
        self._faiss_index = None

    @property
    def effective_mode(self) -> str | None:
        """The actual index mode after building (None if not yet built)."""
        return self._effective_mode

    def _build_from_vectors(self, vectors: NDArray[np.float32], metric: str) -> None:
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

    def _query_single(
        self,
        point: NDArray[np.float32],
        k: int,
    ) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
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

    def _save_implementation(self, path: Path) -> None:
        import faiss

        faiss.write_index(self._faiss_index, str(path))

        extra = {
            "mode": self._mode,
            "effective_mode": self._effective_mode,
            "hnsw_m": self._hnsw_m,
            "hnsw_ef_construction": self._hnsw_ef_construction,
            "hnsw_ef_search": self._hnsw_ef_search,
        }
        with open(str(path) + ".faiss_meta", "wb") as f:
            pickle.dump(extra, f)

    def _load_implementation(self, path: Path) -> None:
        import faiss

        self._faiss_index = faiss.read_index(str(path))

        meta_path = str(path) + ".faiss_meta"
        try:
            with open(meta_path, "rb") as f:
                extra = pickle.load(f)
            self._mode = extra.get("mode", "auto")
            self._effective_mode = extra.get("effective_mode")
            self._hnsw_m = extra.get("hnsw_m", 32)
            self._hnsw_ef_construction = extra.get("hnsw_ef_construction", 40)
            self._hnsw_ef_search = extra.get("hnsw_ef_search", 64)
        except FileNotFoundError:
            self._effective_mode = "flat"

        self._vectors = None
