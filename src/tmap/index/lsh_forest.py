"""
LSH Forest implementation for approximate nearest neighbor search.

This module provides a custom Numba-accelerated implementation:
- Numba JIT for hash band computation (parallel, 50-100x faster than datasketch)
- Sorted arrays with binary search for O(log n) lookups
- Numba JIT for distance computation and linear scan

The LSH Forest is optimized for Jaccard similarity on MinHash signatures,
which is the core algorithm for TMAP's fingerprint-based visualization.
"""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ._lsh_numba import (
    compute_distances_to_candidates,
    compute_hash_bands,
    compute_hash_bands_weighted,
    compute_weighted_distances_to_candidates,
    jaccard_distance,
    linear_scan_batch,
    linear_scan_batch_weighted,
    query_lsh_forest_batch,
    weighted_jaccard_distance,
)
from .types import KNNGraph

if TYPE_CHECKING:
    pass

__all__ = ["LSHForest"]


class LSHForest:
    """
    LSH Forest data structure for approximate nearest neighbor search.

    Uses a custom Numba-accelerated implementation:
    - Parallel hash band computation (replaces datasketch's Python loops)
    - Sorted arrays with binary search for efficient lookups
    - Parallel linear scan for k-NN graph construction

    Args:
        d: Dimensionality of MinHash vectors (number of permutations). Default: 128
        l: Number of prefix trees (bands). Default: 8
        store: Store signatures for linear scan and distance queries. Default: True
        weighted: Whether using weighted MinHash signatures. Default: False

    Example:
        >>> from tmap.index.encoders import MinHash
        >>> from tmap.index import LSHForest
        >>>
        >>> # Create MinHash signatures (Numba-accelerated)
        >>> mh = MinHash(num_perm=128)
        >>> sigs = mh.batch_from_binary_array(fingerprints)
        >>>
        >>> # Build LSH Forest
        >>> lsh = LSHForest(d=128, l=8)
        >>> lsh.batch_add(sigs)
        >>> lsh.index()
        >>>
        >>> # Build k-NN graph (Numba-accelerated linear scan)
        >>> knn_graph = lsh.get_knn_graph(k=20, kc=10)

    Performance:
        - 1M signatures batch_add: ~1s (vs 300s+ with datasketch)
        - Linear scan uses Numba parallel processing
        - Signature storage is contiguous for cache efficiency
    """

    def __init__(
        self,
        d: int = 128,
        l: int = 8,
        store: bool = True,
        weighted: bool = False,
    ) -> None:
        if d <= 0:
            raise ValueError("d must be positive")
        if l <= 0:
            raise ValueError("l must be positive")
        if l > d:
            raise ValueError("l cannot be greater than d")

        self._d = d
        self._l = l
        self._k = d // l  # Band width
        self._store = store
        self._weighted = weighted

        # Signature storage for linear scan and distance queries
        # Collected in list during adds, converted to contiguous array in index()
        self._signatures_list: list[NDArray[np.uint64]] = []
        self._signatures: NDArray[np.uint64] | None = None

        # LSH index structures (built in index())
        self._hash_bands: NDArray[np.uint64] | None = None
        self._sorted_hashes_flat: NDArray[np.uint64] | None = None
        self._sorted_indices_flat: NDArray[np.int32] | None = None
        self._band_offsets: NDArray[np.int64] | None = None

        # State tracking
        self._n_indexed: int = 0
        self._is_indexed: bool = False
        self._needs_reindex: bool = False

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def size(self) -> int:
        """Number of indexed MinHash signatures."""
        return self._n_indexed

    @property
    def is_clean(self) -> bool:
        """Whether the index is up-to-date (index() called after last add)."""
        return self._is_indexed and not self._needs_reindex

    @property
    def d(self) -> int:
        """Number of permutations (signature dimensionality)."""
        return self._d

    @property
    def l(self) -> int:
        """Number of prefix trees."""
        return self._l

    @property
    def is_indexed(self) -> int:
        """Number of prefix trees."""
        return self._is_indexed 
    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _validate_signature_shape(self, signature: NDArray[np.uint64], batch: bool = False) -> None:
        """Validate signature shape matches configuration."""
        if self._weighted:
            if batch:
                if signature.ndim != 3 or signature.shape[1:] != (self._d, 2):
                    raise ValueError(
                        f"Expected shape (n, {self._d}, 2) for weighted, got {signature.shape}"
                    )
            else:
                if signature.shape != (self._d, 2):
                    raise ValueError(
                        f"Expected shape ({self._d}, 2) for weighted, got {signature.shape}"
                    )
        else:
            if batch:
                if signature.ndim != 2 or signature.shape[1] != self._d:
                    raise ValueError(f"Expected shape (n, {self._d}), got {signature.shape}")
            else:
                if signature.shape != (self._d,):
                    raise ValueError(f"Expected shape ({self._d},), got {signature.shape}")

    def _compute_distance(
        self, sig_a: NDArray[np.uint64], sig_b: NDArray[np.uint64]
    ) -> float:
        """Compute distance using appropriate method based on weighted flag."""
        if self._weighted:
            return weighted_jaccard_distance(sig_a, sig_b)
        else:
            return jaccard_distance(sig_a, sig_b)

    # =========================================================================
    # Add methods
    # =========================================================================

    def add(self, signature: NDArray[np.uint64]) -> None:
        """
        Add a MinHash signature to the LSH forest.

        Args:
            signature: MinHash vector of shape (d,) or (d, 2) for weighted

        Note:
            Call index() after adding signatures to build/update the index.
        """
        self._validate_signature_shape(signature)

        # Store signature
        if self._store:
            self._signatures_list.append(signature.copy())

        self._needs_reindex = True

    def batch_add(self, signatures: NDArray[np.uint64]) -> None:
        """
        Add multiple MinHash signatures to the LSH forest (optimized).

        This is now very fast - just appends signatures to a list.
        The actual hash computation is deferred to index().

        Args:
            signatures: MinHash vectors of shape (n, d) or (n, d, 2) for weighted

        Note:
            Call index() after adding signatures to build/update the index.
        """
        self._validate_signature_shape(signatures, batch=True)

        n_samples = signatures.shape[0]

        if self._store:
            # Store each signature individually for incremental adds
            for i in range(n_samples):
                self._signatures_list.append(signatures[i].copy())

        self._needs_reindex = True

    # =========================================================================
    # Index method
    # =========================================================================

    def index(self) -> None:
        """
        Build/rebuild the LSH forest index.

        Must be called after adding signatures with add() or batch_add().
        This is where the Numba-accelerated hash computation happens.
        """
        if not self._signatures_list:
            # Nothing to index
            self._n_indexed = 0
            self._is_indexed = True
            self._needs_reindex = False
            return

        # Convert list to contiguous array for efficient Numba access
        self._signatures = np.stack(self._signatures_list)
        n = self._signatures.shape[0]

        # Compute hash bands for all signatures (Numba-parallel)
        if self._weighted:
            self._hash_bands = compute_hash_bands_weighted(
                self._signatures, self._l, self._k
            )
        else:
            self._hash_bands = compute_hash_bands(
                self._signatures, self._l, self._k
            )

        # Build sorted hash tables for each band
        # We flatten all bands into single arrays with offsets for Numba compatibility
        sorted_hashes_list = []
        sorted_indices_list = []
        band_sizes = []

        for band in range(self._l):
            band_hashes = self._hash_bands[:, band]
            # Sort by hash value
            sort_order = np.argsort(band_hashes)
            sorted_hashes_list.append(band_hashes[sort_order])
            sorted_indices_list.append(sort_order.astype(np.int32))
            band_sizes.append(n)

        # Flatten into contiguous arrays
        self._sorted_hashes_flat = np.concatenate(sorted_hashes_list)
        self._sorted_indices_flat = np.concatenate(sorted_indices_list)

        # Compute offsets
        self._band_offsets = np.zeros(self._l + 1, dtype=np.int64)
        self._band_offsets[1:] = np.cumsum(band_sizes)

        self._n_indexed = n
        self._is_indexed = True
        self._needs_reindex = False

    # =========================================================================
    # Query methods
    # =========================================================================

    def query(self, signature: NDArray[np.uint64], k: int) -> NDArray[np.int32]:
        """
        Query the LSH forest for k-nearest neighbors.

        Uses LSH tree traversal only (no linear scan). For better accuracy,
        use query_linear_scan().

        Args:
            signature: Query MinHash vector of shape (d,) or (d, 2)
            k: Number of nearest neighbors to retrieve

        Returns:
            Array of neighbor indices, shape (k,) or fewer if not enough neighbors
        """
        if not self._is_indexed:
            raise RuntimeError("Must call index() before querying")
        if self._n_indexed == 0:
            return np.array([], dtype=np.int32)

        self._validate_signature_shape(signature)

        # Compute hash bands for query
        if self._weighted:
            query_bands = compute_hash_bands_weighted(
                signature[np.newaxis, :, :], self._l, self._k
            )
        else:
            query_bands = compute_hash_bands(
                signature[np.newaxis, :], self._l, self._k
            )

        if (
            self._sorted_hashes_flat is None
            or self._sorted_indices_flat is None
            or self._band_offsets is None
        ):
            raise RuntimeError("Index structures not initialized")

        # Query using Numba batch function (single query)
        candidates, counts = query_lsh_forest_batch(
            query_bands,
            self._sorted_hashes_flat,
            self._sorted_indices_flat,
            self._band_offsets,
            k,
        )

        # Extract valid candidates
        n_valid = counts[0]
        return candidates[0, :n_valid].copy()

    def query_by_id(self, id: int, k: int) -> NDArray[np.int32]:
        """
        Query k-nearest neighbors for an indexed signature by its ID.

        Args:
            id: Index of the query signature (0-based, order of insertion)
            k: Number of nearest neighbors to retrieve

        Returns:
            Array of neighbor indices

        Raises:
            ValueError: If store=False (signatures not retained)
        """
        if not self._store:
            raise ValueError("query_by_id requires store=True")
        if self._signatures is None:
            raise RuntimeError("Must call index() before querying")
        if id < 0 or id >= self._n_indexed:
            raise IndexError(f"ID {id} out of range [0, {self._n_indexed})")

        return self.query(self._signatures[id], k)

    # =========================================================================
    # Linear scan methods (Numba-accelerated)
    # =========================================================================

    def linear_scan(
        self,
        signature: NDArray[np.uint64],
        indices: NDArray[np.int32] | list[int],
        k: int = 10,
    ) -> list[tuple[float, int]]:
        """
        Query a subset of indexed signatures using linear scan.

        Computes exact distances to all specified candidates and returns top-k.

        Args:
            signature: Query MinHash vector
            indices: Subset of indices to search
            k: Number of nearest neighbors to retrieve

        Returns:
            List of (distance, index) tuples, sorted by distance
        """
        if not self._store:
            raise ValueError("linear_scan requires store=True")
        if self._signatures is None:
            raise RuntimeError("Must call index() before linear scan")

        self._validate_signature_shape(signature)

        indices_arr = np.asarray(indices, dtype=np.int32)
        if len(indices_arr) == 0:
            return []

        # Get candidate signatures
        candidates = self._signatures[indices_arr]

        # Compute distances
        if self._weighted:
            distances = compute_weighted_distances_to_candidates(signature, candidates)
        else:
            distances = compute_distances_to_candidates(signature, candidates)

        # Get top-k
        actual_k = min(k, len(indices_arr))
        top_k_idx = np.argpartition(distances, actual_k - 1)[:actual_k]
        top_k_idx = top_k_idx[np.argsort(distances[top_k_idx])]

        return [(float(distances[i]), int(indices_arr[i])) for i in top_k_idx]

    def query_linear_scan(
        self,
        signature: NDArray[np.uint64],
        k: int,
        kc: int = 10,
    ) -> list[tuple[float, int]]:
        """
        Query k-nearest neighbors with LSH forest + linear scan combination.

        First retrieves k*kc candidates using LSH forest, then performs
        linear scan on candidates to find exact k nearest neighbors.

        Args:
            signature: Query MinHash vector
            k: Number of nearest neighbors to retrieve
            kc: Multiplier for LSH forest retrieval (retrieves k*kc candidates)

        Returns:
            List of (distance, index) tuples, sorted by distance
        """
        # Get candidates from LSH
        candidates = self.query(signature, k * kc)

        if len(candidates) == 0:
            return []

        # Linear scan on candidates
        return self.linear_scan(signature, candidates, k)

    def query_linear_scan_by_id(
        self,
        id: int,
        k: int,
        kc: int = 10,
    ) -> list[tuple[float, int]]:
        """
        Query k-nearest neighbors by ID with LSH forest + linear scan.

        Args:
            id: Index of the query signature
            k: Number of nearest neighbors to retrieve
            kc: Multiplier for LSH forest retrieval

        Returns:
            List of (distance, index) tuples, sorted by distance
        """
        if not self._store:
            raise ValueError("query_linear_scan_by_id requires store=True")
        if self._signatures is None:
            raise RuntimeError("Must call index() before querying")
        if id < 0 or id >= self._n_indexed:
            raise IndexError(f"ID {id} out of range [0, {self._n_indexed})")

        results = self.query_linear_scan(self._signatures[id], k + 1, kc)

        # Exclude self from results
        return [(d, i) for d, i in results if i != id][:k]

    # =========================================================================
    # k-NN Graph methods (main output for TMAP pipeline)
    # =========================================================================

    def get_all_nearest_neighbors(
        self,
        k: int,
        kc: int = 10,
    ) -> NDArray[np.int32]:
        """
        Get k-nearest neighbors of all indexed signatures.

        Args:
            k: Number of nearest neighbors per point
            kc: Multiplier for LSH forest retrieval

        Returns:
            Flattened array of neighbor indices, shape (n * k,).
            Use reshape(n, k) to get per-point neighbors.
        """
        knn = self.get_knn_graph(k, kc)
        return knn.indices.flatten()

    def get_knn_graph(
        self,
        k: int,
        kc: int = 10,
    ) -> KNNGraph:
        """
        Construct the k-nearest neighbor graph of all indexed signatures.

        This is the primary output method - produces input for MSTBuilder.
        Uses Numba-accelerated parallel operations throughout.

        Args:
            k: Number of nearest neighbors per point
            kc: Multiplier for LSH forest retrieval

        Returns:
            KNNGraph with indices and distances arrays
        """
        if not self._store:
            raise ValueError("get_knn_graph requires store=True")
        if self._signatures is None or self._n_indexed == 0:
            raise RuntimeError("Must add signatures and call index() first")

        max_candidates = k * kc

        if (
            self._hash_bands is None
            or self._sorted_hashes_flat is None
            or self._sorted_indices_flat is None
            or self._band_offsets is None
        ):
            raise RuntimeError("Index structures not initialized")

        # Batch query all signatures using Numba-parallel
        all_candidates, candidate_counts = query_lsh_forest_batch(
            self._hash_bands,
            self._sorted_hashes_flat,
            self._sorted_indices_flat,
            self._band_offsets,
            max_candidates,
        )

        # Numba-accelerated linear scan
        if self._weighted:
            indices, distances = linear_scan_batch_weighted(
                self._signatures,
                self._signatures,
                all_candidates,
                candidate_counts,
                k,
            )
        else:
            indices, distances = linear_scan_batch(
                self._signatures,
                self._signatures,
                all_candidates,
                candidate_counts,
                k,
            )

        return KNNGraph(indices=indices, distances=distances)

    # =========================================================================
    # Distance methods
    # =========================================================================

    @staticmethod
    def get_distance(
        sig_a: NDArray[np.uint64],
        sig_b: NDArray[np.uint64],
    ) -> float:
        """
        Calculate Jaccard distance between two MinHash signatures.

        Args:
            sig_a: First MinHash vector
            sig_b: Second MinHash vector

        Returns:
            Jaccard distance (0.0 to 1.0)
        """
        return jaccard_distance(sig_a, sig_b)

    @staticmethod
    def get_weighted_distance(
        sig_a: NDArray[np.uint64],
        sig_b: NDArray[np.uint64],
    ) -> float:
        """
        Calculate weighted Jaccard distance between two weighted MinHash signatures.

        Args:
            sig_a: First weighted MinHash vector, shape (d, 2)
            sig_b: Second weighted MinHash vector, shape (d, 2)

        Returns:
            Weighted Jaccard distance (0.0 to 1.0)
        """
        return weighted_jaccard_distance(sig_a, sig_b)

    def get_distance_by_id(self, a: int, b: int) -> float:
        """
        Calculate Jaccard distance between two indexed signatures.

        Args:
            a: Index of first signature
            b: Index of second signature

        Returns:
            Jaccard distance

        Raises:
            ValueError: If store=False
        """
        if not self._store:
            raise ValueError("get_distance_by_id requires store=True")
        if self._signatures is None:
            raise RuntimeError("Must call index() first")

        return self._compute_distance(self._signatures[a], self._signatures[b])

    def get_all_distances(
        self,
        signature: NDArray[np.uint64],
    ) -> NDArray[np.float32]:
        """
        Calculate distances from a signature to all indexed signatures.

        Args:
            signature: Query MinHash vector

        Returns:
            Array of distances, shape (n_indexed,)

        Raises:
            ValueError: If store=False
        """
        if not self._store:
            raise ValueError("get_all_distances requires store=True")
        if self._signatures is None:
            raise RuntimeError("Must call index() first")

        self._validate_signature_shape(signature)

        if self._weighted:
            return compute_weighted_distances_to_candidates(signature, self._signatures)
        else:
            return compute_distances_to_candidates(signature, self._signatures)

    # =========================================================================
    # Storage / Retrieval
    # =========================================================================

    def get_hash(self, id: int) -> NDArray[np.uint64]:
        """
        Retrieve the MinHash signature of an indexed entry.

        Args:
            id: Index of the signature (0-based, order of insertion)

        Returns:
            MinHash vector, shape (d,) or (d, 2) for weighted

        Raises:
            ValueError: If store=False
            IndexError: If id out of range
        """
        if not self._store:
            raise ValueError("get_hash requires store=True")
        if self._signatures is None:
            raise RuntimeError("Must call index() first")
        if id < 0 or id >= self._n_indexed:
            raise IndexError(f"ID {id} out of range [0, {self._n_indexed})")

        return self._signatures[id].copy()

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: str) -> None:
        """
        Serialize the LSH forest to disk.

        Args:
            path: File path for serialization
        """
        state = {
            "d": self._d,
            "l": self._l,
            "k": self._k,
            "store": self._store,
            "weighted": self._weighted,
            "signatures": self._signatures,
            "signatures_list": self._signatures_list,
            "hash_bands": self._hash_bands,
            "sorted_hashes_flat": self._sorted_hashes_flat,
            "sorted_indices_flat": self._sorted_indices_flat,
            "band_offsets": self._band_offsets,
            "n_indexed": self._n_indexed,
            "is_indexed": self._is_indexed,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> LSHForest:
        """
        Load a serialized LSH forest from disk.

        Args:
            path: File path to load from

        Returns:
            Restored LSHForest instance, ready for queries
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        instance = cls(
            d=state["d"],
            l=state["l"],
            store=state["store"],
            weighted=state["weighted"],
        )
        instance._k = state["k"]
        instance._signatures = state["signatures"]
        instance._signatures_list = state["signatures_list"]
        instance._hash_bands = state["hash_bands"]
        instance._sorted_hashes_flat = state["sorted_hashes_flat"]
        instance._sorted_indices_flat = state["sorted_indices_flat"]
        instance._band_offsets = state["band_offsets"]
        instance._n_indexed = state["n_indexed"]
        instance._is_indexed = state["is_indexed"]
        instance._needs_reindex = False

        return instance

    # =========================================================================
    # State methods
    # =========================================================================

    def clear(self) -> None:
        """Clear all added data and computed indices."""
        self._signatures_list = []
        self._signatures = None
        self._hash_bands = None
        self._sorted_hashes_flat = None
        self._sorted_indices_flat = None
        self._band_offsets = None
        self._n_indexed = 0
        self._is_indexed = False
        self._needs_reindex = False
