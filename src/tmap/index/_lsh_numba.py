"""
Numba JIT-accelerated functions for LSH Forest operations.

This module provides high-performance distance computation and linear scan
operations used by LSHForest for k-NN graph construction.

Key optimizations:
1. Vectorized distance computation across candidates
2. Parallel linear scan over all queries for k-NN graph
3. Efficient top-k selection using partial sorting
"""

import numpy as np
from numpy.typing import NDArray

try:
    import numba
    from numba import prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    prange = range  # fallback for type checking


if NUMBA_AVAILABLE:

    @numba.njit(cache=True)
    def jaccard_distance(sig_a: NDArray[np.uint64], sig_b: NDArray[np.uint64]) -> float:
        """
        Compute Jaccard distance between two MinHash signatures.

        Distance = 1 - (number of matching hash values) / num_perm

        Args:
            sig_a: First MinHash signature of shape (d,)
            sig_b: Second MinHash signature of shape (d,)

        Returns:
            Jaccard distance (0.0 to 1.0)
        """
        d = len(sig_a)
        matches = 0
        for i in range(d):
            if sig_a[i] == sig_b[i]:
                matches += 1
        return 1.0 - matches / d

    @numba.njit(cache=True)
    def weighted_jaccard_distance(
        sig_a: NDArray[np.uint64], sig_b: NDArray[np.uint64]
    ) -> float:
        """
        Compute weighted Jaccard distance between two weighted MinHash signatures.

        For weighted MinHash, both columns (k, y_k) must match for a row to count.
        Distance = 1 - (number of matching rows) / num_perm

        Args:
            sig_a: First weighted MinHash signature of shape (d, 2)
            sig_b: Second weighted MinHash signature of shape (d, 2)

        Returns:
            Weighted Jaccard distance (0.0 to 1.0)
        """
        d = sig_a.shape[0]
        matches = 0
        for i in range(d):
            if sig_a[i, 0] == sig_b[i, 0] and sig_a[i, 1] == sig_b[i, 1]:
                matches += 1
        return 1.0 - matches / d

    @numba.njit(parallel=True, cache=True)
    def compute_distances_to_candidates(
        query: NDArray[np.uint64],
        candidates: NDArray[np.uint64],
    ) -> NDArray[np.float32]:
        """
        Compute Jaccard distances from query to all candidates (vectorized).

        Args:
            query: Query signature of shape (d,)
            candidates: Candidate signatures of shape (n_candidates, d)

        Returns:
            Distances array of shape (n_candidates,)
        """
        n_candidates = candidates.shape[0]
        d = query.shape[0]
        distances = np.empty(n_candidates, dtype=np.float32)

        for i in prange(n_candidates):
            matches = 0
            for j in range(d):
                if query[j] == candidates[i, j]:
                    matches += 1
            distances[i] = 1.0 - matches / d

        return distances

    @numba.njit(parallel=True, cache=True)
    def compute_weighted_distances_to_candidates(
        query: NDArray[np.uint64],
        candidates: NDArray[np.uint64],
    ) -> NDArray[np.float32]:
        """
        Compute weighted Jaccard distances from query to all candidates.

        Args:
            query: Query signature of shape (d, 2)
            candidates: Candidate signatures of shape (n_candidates, d, 2)

        Returns:
            Distances array of shape (n_candidates,)
        """
        n_candidates = candidates.shape[0]
        d = query.shape[0]
        distances = np.empty(n_candidates, dtype=np.float32)

        for i in prange(n_candidates):
            matches = 0
            for j in range(d):
                if query[j, 0] == candidates[i, j, 0] and query[j, 1] == candidates[i, j, 1]:
                    matches += 1
            distances[i] = 1.0 - matches / d

        return distances

    @numba.njit(cache=True)
    def _argsort_topk(arr: NDArray[np.float32], k: int) -> NDArray[np.int32]:
        """Get indices of k smallest elements (partial sort)."""
        n = len(arr)
        if k >= n:
            return np.argsort(arr).astype(np.int32)

        # Use argpartition-like approach
        indices = np.arange(n, dtype=np.int32)

        # Simple selection sort for top-k (efficient for small k)
        for i in range(k):
            min_idx = i
            for j in range(i + 1, n):
                if arr[indices[j]] < arr[indices[min_idx]]:
                    min_idx = j
            # Swap
            indices[i], indices[min_idx] = indices[min_idx], indices[i]

        return indices[:k]

    @numba.njit(parallel=True, cache=True)
    def linear_scan_batch(
        queries: NDArray[np.uint64],
        signatures: NDArray[np.uint64],
        candidate_indices: NDArray[np.int32],
        candidate_counts: NDArray[np.int32],
        k: int,
    ) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        """
        Perform linear scan for multiple queries in parallel.

        For each query, computes distances to its candidates and returns top-k.

        Args:
            queries: Query signatures of shape (n_queries, d)
            signatures: All stored signatures of shape (n_total, d)
            candidate_indices: Candidate indices for each query, shape (n_queries, max_candidates)
                              Padded with -1 for queries with fewer candidates
            candidate_counts: Number of valid candidates per query, shape (n_queries,)
            k: Number of nearest neighbors to return per query

        Returns:
            Tuple of:
            - indices: (n_queries, k) array of neighbor indices
            - distances: (n_queries, k) array of distances
        """
        n_queries = queries.shape[0]
        d = queries.shape[1]

        result_indices = np.full((n_queries, k), -1, dtype=np.int32)
        result_distances = np.full((n_queries, k), np.float32(2.0), dtype=np.float32)

        for q in prange(n_queries):
            n_cand = candidate_counts[q]
            if n_cand == 0:
                continue

            # Compute distances to candidates
            cand_distances = np.empty(n_cand, dtype=np.float32)
            for i in range(n_cand):
                cand_idx = candidate_indices[q, i]
                if cand_idx < 0:
                    cand_distances[i] = 2.0  # Invalid candidate
                elif cand_idx == q:
                    cand_distances[i] = 2.0  # Exclude self
                else:
                    # Compute Jaccard distance
                    matches = 0
                    for j in range(d):
                        if queries[q, j] == signatures[cand_idx, j]:
                            matches += 1
                    cand_distances[i] = 1.0 - matches / d

            # Get top-k indices
            actual_k = min(k, n_cand)
            top_k = _argsort_topk(cand_distances, actual_k)

            # Only assign valid results (distance < 2.0 means valid neighbor)
            result_idx = 0
            for i in range(actual_k):
                idx = top_k[i]
                if cand_distances[idx] < 2.0:
                    result_indices[q, result_idx] = candidate_indices[q, idx]
                    result_distances[q, result_idx] = cand_distances[idx]
                    result_idx += 1

        return result_indices, result_distances

    @numba.njit(parallel=True, cache=True)
    def linear_scan_batch_weighted(
        queries: NDArray[np.uint64],
        signatures: NDArray[np.uint64],
        candidate_indices: NDArray[np.int32],
        candidate_counts: NDArray[np.int32],
        k: int,
    ) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        """
        Perform linear scan for multiple weighted queries in parallel.

        Args:
            queries: Query signatures of shape (n_queries, d, 2)
            signatures: All stored signatures of shape (n_total, d, 2)
            candidate_indices: Candidate indices, shape (n_queries, max_candidates)
            candidate_counts: Number of valid candidates per query
            k: Number of nearest neighbors to return

        Returns:
            Tuple of (indices, distances) arrays
        """
        n_queries = queries.shape[0]
        d = queries.shape[1]

        result_indices = np.full((n_queries, k), -1, dtype=np.int32)
        result_distances = np.full((n_queries, k), np.float32(2.0), dtype=np.float32)

        for q in prange(n_queries):
            n_cand = candidate_counts[q]
            if n_cand == 0:
                continue

            cand_distances = np.empty(n_cand, dtype=np.float32)
            for i in range(n_cand):
                cand_idx = candidate_indices[q, i]
                if cand_idx < 0 or cand_idx == q:
                    cand_distances[i] = 2.0
                else:
                    matches = 0
                    for j in range(d):
                        if (
                            queries[q, j, 0] == signatures[cand_idx, j, 0]
                            and queries[q, j, 1] == signatures[cand_idx, j, 1]
                        ):
                            matches += 1
                    cand_distances[i] = 1.0 - matches / d

            actual_k = min(k, n_cand)
            top_k = _argsort_topk(cand_distances, actual_k)

            # Only assign valid results (distance < 2.0 means valid neighbor)
            result_idx = 0
            for i in range(actual_k):
                idx = top_k[i]
                if cand_distances[idx] < 2.0:
                    result_indices[q, result_idx] = candidate_indices[q, idx]
                    result_distances[q, result_idx] = cand_distances[idx]
                    result_idx += 1

        return result_indices, result_distances

else:
    # Fallback implementations when Numba is not available

    def jaccard_distance(sig_a: NDArray[np.uint64], sig_b: NDArray[np.uint64]) -> float:
        """Compute Jaccard distance (fallback)."""
        return float(1.0 - np.mean(sig_a == sig_b))

    def weighted_jaccard_distance(
        sig_a: NDArray[np.uint64], sig_b: NDArray[np.uint64]
    ) -> float:
        """Compute weighted Jaccard distance (fallback)."""
        matches = np.all(sig_a == sig_b, axis=1)
        return float(1.0 - np.mean(matches))

    def compute_distances_to_candidates(
        query: NDArray[np.uint64],
        candidates: NDArray[np.uint64],
    ) -> NDArray[np.float32]:
        """Compute distances to candidates (fallback)."""
        matches = (candidates == query).mean(axis=1)
        return (1.0 - matches).astype(np.float32)

    def compute_weighted_distances_to_candidates(
        query: NDArray[np.uint64],
        candidates: NDArray[np.uint64],
    ) -> NDArray[np.float32]:
        """Compute weighted distances to candidates (fallback)."""
        matches = np.all(candidates == query, axis=2).mean(axis=1)
        return (1.0 - matches).astype(np.float32)

    def linear_scan_batch(
        queries: NDArray[np.uint64],
        signatures: NDArray[np.uint64],
        candidate_indices: NDArray[np.int32],
        candidate_counts: NDArray[np.int32],
        k: int,
    ) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        """Linear scan batch (fallback)."""
        n_queries = queries.shape[0]
        result_indices = np.full((n_queries, k), -1, dtype=np.int32)
        result_distances = np.full((n_queries, k), 2.0, dtype=np.float32)

        for q in range(n_queries):
            n_cand = candidate_counts[q]
            if n_cand == 0:
                continue

            cand_idx = candidate_indices[q, :n_cand]
            cand_sigs = signatures[cand_idx]
            distances = compute_distances_to_candidates(queries[q], cand_sigs)

            # Exclude self
            self_mask = cand_idx == q
            distances[self_mask] = 2.0

            # Get top-k
            actual_k = min(k, n_cand)
            top_k = np.argpartition(distances, actual_k - 1)[:actual_k]
            top_k = top_k[np.argsort(distances[top_k])]

            # Only assign valid results (distance < 2.0 means valid neighbor)
            valid_mask = distances[top_k] < 2.0
            valid_top_k = top_k[valid_mask]
            n_valid = len(valid_top_k)
            result_indices[q, :n_valid] = cand_idx[valid_top_k]
            result_distances[q, :n_valid] = distances[valid_top_k]

        return result_indices, result_distances

    def linear_scan_batch_weighted(
        queries: NDArray[np.uint64],
        signatures: NDArray[np.uint64],
        candidate_indices: NDArray[np.int32],
        candidate_counts: NDArray[np.int32],
        k: int,
    ) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        """Linear scan batch for weighted signatures (fallback)."""
        n_queries = queries.shape[0]
        result_indices = np.full((n_queries, k), -1, dtype=np.int32)
        result_distances = np.full((n_queries, k), 2.0, dtype=np.float32)

        for q in range(n_queries):
            n_cand = candidate_counts[q]
            if n_cand == 0:
                continue

            cand_idx = candidate_indices[q, :n_cand]
            cand_sigs = signatures[cand_idx]
            distances = compute_weighted_distances_to_candidates(queries[q], cand_sigs)

            self_mask = cand_idx == q
            distances[self_mask] = 2.0

            actual_k = min(k, n_cand)
            top_k = np.argpartition(distances, actual_k - 1)[:actual_k]
            top_k = top_k[np.argsort(distances[top_k])]

            # Only assign valid results (distance < 2.0 means valid neighbor)
            valid_mask = distances[top_k] < 2.0
            valid_top_k = top_k[valid_mask]
            n_valid = len(valid_top_k)
            result_indices[q, :n_valid] = cand_idx[valid_top_k]
            result_distances[q, :n_valid] = distances[valid_top_k]

        return result_indices, result_distances
