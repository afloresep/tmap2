"""FAISS-based exact/approximate nearest neighbor index.

Wraps Facebook AI Similarity Search (https://github.com/facebookresearch/faiss)
for cosine/euclidean kNN search. Supports GPU acceleration and automatic index
type selection (flat, IVF, IVFPQ) based on dataset size.

Requires: ``pip install faiss-cpu`` or ``pip install faiss-gpu``
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tmap.index.base import Index
from tmap.index.types import KNNGraph

# Auto-selection thresholds
_IVF_THRESHOLD = 50_000
_IVFPQ_THRESHOLD = 500_000


def _largest_divisor_leq(d: int, max_val: int) -> int:
    """Return the largest divisor of *d* that is <= *max_val*."""
    for m in range(min(max_val, d), 0, -1):
        if d % m == 0:
            return m
    return 1  # pragma: no cover – d >= 1 always


class FaissIndex(Index):
    """Nearest neighbor index using FAISS.

    Supports exact (flat) and approximate (IVF, IVFPQ) search modes.
    Use ``mode="auto"`` (default) to let the index pick the best strategy
    based on dataset size.

    Parameters
    ----------
    seed : int or None
        Random seed (used for IVF training sample selection).
    use_gpu : bool
        If True, transfer the index to GPU 0 for faster search.
    mode : str
        Index mode. One of:

        - ``"auto"`` (default): flat for n < 50k, IVF for n < 500k, IVFPQ above.
        - ``"flat"``: Exact brute-force search. O(n) query time.
        - ``"ivf"``: Inverted file index. Approximate, no compression.
        - ``"ivfpq"``: Inverted file + product quantization. Approximate, compressed.
    nprobe : int
        Number of cells to probe during IVF/IVFPQ search. Higher = better
        recall, slower search. Default 32.
    """

    def __init__(
        self,
        seed: int | None = None,
        use_gpu: bool = False,
        mode: str = "auto",
        nprobe: int = 32,
    ) -> None:
        super().__init__(seed=seed)
        if mode not in {"auto", "flat", "ivf", "ivfpq"}:
            raise ValueError(f"mode must be auto/flat/ivf/ivfpq, got {mode!r}")
        self._use_gpu = use_gpu
        self._mode = mode
        self._nprobe = nprobe
        self._effective_mode: str | None = None  # resolved after build
        self._vectors: NDArray[np.float32] | None = None
        self._faiss_index = None  # faiss.Index (typed as Any to avoid import)

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

        # Resolve auto mode
        if self._mode == "auto":
            if n < _IVF_THRESHOLD:
                effective = "flat"
            elif n < _IVFPQ_THRESHOLD:
                effective = "ivf"
            else:
                effective = "ivfpq"
        else:
            effective = self._mode
        self._effective_mode = effective

        if effective == "flat":
            index = self._build_flat(d, metric_type)
            index.add(vectors)
        elif effective == "ivf":
            index = self._build_ivf(vectors, n, d, metric_type)
        elif effective == "ivfpq":
            index = self._build_ivfpq(vectors, n, d, metric_type)
        else:
            raise ValueError(f"Unknown effective mode {effective!r}")

        if self._use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        self._faiss_index = index
        self._vectors = vectors

    def _build_flat(self, d: int, metric_type: int) -> Any:
        import faiss

        if metric_type == faiss.METRIC_INNER_PRODUCT:
            return faiss.IndexFlatIP(d)
        return faiss.IndexFlatL2(d)

    def _build_ivf(
        self,
        vectors: NDArray[np.float32],
        n: int,
        d: int,
        metric_type: int,
    ) -> Any:
        import faiss

        nlist = max(int(math.sqrt(n)), 1)
        quantizer = self._build_flat(d, metric_type)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, metric_type)
        index.nprobe = self._nprobe

        # Train on a subset
        n_train = min(n, max(nlist * 40, 100_000))
        train_data = self._sample_train_data(vectors, n_train)
        index.train(train_data)
        index.add(vectors)
        return index

    def _build_ivfpq(
        self,
        vectors: NDArray[np.float32],
        n: int,
        d: int,
        metric_type: int,
    ) -> Any:
        import faiss

        nlist = max(int(4 * math.sqrt(n)), 1)
        m = _largest_divisor_leq(d, 64)
        nbits = 8
        quantizer = self._build_flat(d, metric_type)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, metric_type)
        index.nprobe = self._nprobe

        # Train on a subset
        n_train = min(n, max(nlist * 40, 500_000))
        train_data = self._sample_train_data(vectors, n_train)
        index.train(train_data)
        index.add(vectors)
        return index

    def _sample_train_data(self, vectors: NDArray[np.float32], n_train: int) -> NDArray[np.float32]:
        n = vectors.shape[0]
        if n_train >= n:
            return vectors
        rng = np.random.default_rng(self._seed)
        idx = rng.choice(n, size=n_train, replace=False)
        return vectors[idx]

    def _query_all(self, k: int) -> KNNGraph:
        n = self._vectors.shape[0]
        distances, indices = self._faiss_index.search(self._vectors, k + 1)

        if self._effective_mode == "flat":
            # Exact index: self is always at position 0
            indices = indices[:, 1:]
            distances = distances[:, 1:]
        else:
            # Approximate index: self may not be at position 0
            row_ids = np.arange(n)
            if (indices[:, 0] == row_ids).all():
                # Fast path
                indices = indices[:, 1:]
                distances = distances[:, 1:]
            else:
                # Slow path: per-row filter
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

        # GPU index must be transferred back to CPU before saving
        index = self._faiss_index
        if self._use_gpu:
            index = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index, str(path))

        # Save extra metadata (mode, nprobe)
        extra = {
            "mode": self._mode,
            "effective_mode": self._effective_mode,
            "nprobe": self._nprobe,
            "use_gpu": self._use_gpu,
        }
        with open(str(path) + ".faiss_meta", "wb") as f:
            pickle.dump(extra, f)

    def _load_implementation(self, path: Path) -> None:
        import faiss

        self._faiss_index = faiss.read_index(str(path))
        if self._use_gpu:
            res = faiss.StandardGpuResources()
            self._faiss_index = faiss.index_cpu_to_gpu(res, 0, self._faiss_index)

        # Load extra metadata
        meta_path = str(path) + ".faiss_meta"
        try:
            with open(meta_path, "rb") as f:
                extra = pickle.load(f)
            self._mode = extra.get("mode", "auto")
            self._effective_mode = extra.get("effective_mode")
            self._nprobe = extra.get("nprobe", 32)
        except FileNotFoundError:
            # Backwards-compatible: old indices saved without extra metadata
            self._effective_mode = "flat"

        # Vectors are not persisted; _query_all won't work after load.
        # _query_single works fine without stored vectors.
        self._vectors = None
