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

import numba
from numba import prange


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
def weighted_jaccard_distance(sig_a: NDArray[np.uint64], sig_b: NDArray[np.uint64]) -> float:
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
    exclude_self: bool = True,
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
        exclude_self: If True, candidate with index == query index is
            skipped (for self-kNN).  Set False for external queries.

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
            elif exclude_self and cand_idx == q:
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
    exclude_self: bool = True,
) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
    """
    Perform linear scan for multiple weighted queries in parallel.

    Args:
        queries: Query signatures of shape (n_queries, d, 2)
        signatures: All stored signatures of shape (n_total, d, 2)
        candidate_indices: Candidate indices, shape (n_queries, max_candidates)
        candidate_counts: Number of valid candidates per query
        k: Number of nearest neighbors to return
        exclude_self: If True, candidate with index == query index is
            skipped (for self-kNN).  Set False for external queries.

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
            if cand_idx < 0 or (exclude_self and cand_idx == q):
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


@numba.njit(parallel=True, cache=True)
def compute_hash_bands(
    signatures: NDArray[np.uint64],
    l: int,
    k: int,
) -> NDArray[np.uint64]:
    """
    Compute L hash bands for all N signatures in parallel.

    Each band hashes k consecutive values from the signature.
    This is the core LSH operation - similar signatures will have
    matching bands with high probability.

    Args:
        signatures: (N, d) uint64 array of MinHash signatures
        l: Number of bands (prefix trees)
        k: Band width (d // l)

    Returns:
        hash_bands: (N, l) uint64 array of hash values per band
    """
    n = signatures.shape[0]
    hash_bands = np.empty((n, l), dtype=np.uint64)

    for i in prange(n):
        for band in range(l):
            start = band * k
            end = start + k
            # Inline hash computation for speed
            GOLDEN = np.uint64(0x9E3779B97F4A7C15)
            h = np.uint64(0)
            for j in range(start, end):
                v = signatures[i, j]
                v ^= v >> np.uint64(33)
                v *= GOLDEN
                v ^= v >> np.uint64(33)
                h ^= v + GOLDEN + (h << np.uint64(6)) + (h >> np.uint64(2))
            hash_bands[i, band] = h

    return hash_bands


@numba.njit(parallel=True, cache=True)
def compute_hash_bands_weighted(
    signatures: NDArray[np.uint64],
    l: int,
    k: int,
) -> NDArray[np.uint64]:
    """
    Compute L hash bands for weighted MinHash signatures.

    For weighted MinHash, only uses the first column (k values) for LSH looku

    Args:
        signatures: (N, d, 2) uint64 array of weighted MinHash signatures
        l: Number of bands
        k: Band width

    Returns:
        hash_bands: (N, l) uint64 array of hash values
    """
    n = signatures.shape[0]
    hash_bands = np.empty((n, l), dtype=np.uint64)

    for i in prange(n):
        for band in range(l):
            start = band * k
            end = start + k
            GOLDEN = np.uint64(0x9E3779B97F4A7C15)
            h = np.uint64(0)
            for j in range(start, end):
                # Only use first column for weighted MinHash LSH
                v = signatures[i, j, 0]
                v ^= v >> np.uint64(33)
                v *= GOLDEN
                v ^= v >> np.uint64(33)
                h ^= v + GOLDEN + (h << np.uint64(6)) + (h >> np.uint64(2))
            hash_bands[i, band] = h

    return hash_bands


@numba.njit(cache=True)
def _binary_search_left(arr: NDArray[np.uint64], value: np.uint64) -> int:
    """Binary search for leftmost position where arr[i] >= value."""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


@numba.njit(cache=True)
def _binary_search_right(arr: NDArray[np.uint64], value: np.uint64) -> int:
    """Binary search for leftmost position where arr[i] > value."""
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    return lo


@numba.njit(cache=True)
def query_single_band(
    query_hash: np.uint64,
    sorted_hashes: NDArray[np.uint64],
    sorted_indices: NDArray[np.int32],
) -> NDArray[np.int32]:
    """
    Query a single band's hash table using binary search.

    Args:
        query_hash: Hash value of the query's band
        sorted_hashes: Sorted array of hash values for this band
        sorted_indices: Corresponding signature indices (sorted by hash)

    Returns:
        Array of matching signature indices (may be empty)
    """
    if len(sorted_hashes) == 0:
        return np.empty(0, dtype=np.int32)

    # Find range of matching hashes
    left = _binary_search_left(sorted_hashes, query_hash)
    right = _binary_search_right(sorted_hashes, query_hash)

    if left >= right:
        return np.empty(0, dtype=np.int32)

    return sorted_indices[left:right].copy()


@numba.njit(cache=True)
def query_lsh_forest_single(
    query_bands: NDArray[np.uint64],
    sorted_hashes_list: tuple,
    sorted_indices_list: tuple,
    max_results: int,
) -> NDArray[np.int32]:
    """
    Query LSH forest for a single signature.

    Searches all bands and collects unique candidates up to max_results.
    Uses exact band-hash matching — candidate recall depends on band width
    (k = d/l) and data similarity.

    Args:
        query_bands: (l,) array of hash values for query
        sorted_hashes_list: Tuple of L sorted hash arrays
        sorted_indices_list: Tuple of L sorted index arrays
        max_results: Maximum number of candidates to return

    Returns:
        Array of candidate indices (unique, up to max_results)
    """
    l = len(query_bands)

    # Collect candidates from all bands
    # Use a fixed-size array and track seen indices
    candidates = np.empty(max_results * l, dtype=np.int32)
    n_candidates = 0

    # Simple seen tracking - for small result sets this is fast enough
    seen = np.zeros(max_results * l * 2, dtype=np.int32)
    seen_count = 0

    for band in range(l):
        matches = query_single_band(
            query_bands[band],
            sorted_hashes_list[band],
            sorted_indices_list[band],
        )

        for j in range(len(matches)):
            idx = matches[j]
            # Check if already seen (linear search - ok for small sets)
            is_seen = False
            for s in range(seen_count):
                if seen[s] == idx:
                    is_seen = True
                    break

            if not is_seen:
                if n_candidates < max_results:
                    candidates[n_candidates] = idx
                    n_candidates += 1
                if seen_count < len(seen):
                    seen[seen_count] = idx
                    seen_count += 1

                if n_candidates >= max_results:
                    return candidates[:n_candidates]

    return candidates[:n_candidates]


@numba.njit(parallel=True, cache=True)
def query_lsh_forest_batch(
    query_bands: NDArray[np.uint64],
    sorted_hashes_flat: NDArray[np.uint64],
    sorted_indices_flat: NDArray[np.int32],
    band_offsets: NDArray[np.int64],
    max_results: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """
    Query LSH forest for multiple signatures in parallel.

    Uses flattened arrays for Numba compatibility.

    Args:
        query_bands: (n_queries, l) array of hash values
        sorted_hashes_flat: Flattened sorted hashes for all bands
        sorted_indices_flat: Flattened sorted indices for all bands
        band_offsets: (l+1,) offsets into flat arrays for each band
        max_results: Maximum candidates per query

    Returns:
        candidates: (n_queries, max_results) padded with -1
        counts: (n_queries,) number of valid candidates per query
    """
    n_queries = query_bands.shape[0]
    l = query_bands.shape[1]

    candidates = np.full((n_queries, max_results), -1, dtype=np.int32)
    counts = np.zeros(n_queries, dtype=np.int32)

    for q in prange(n_queries):
        # Track seen indices for this query
        seen = np.zeros(max_results * 2, dtype=np.int32)
        seen_count = 0
        n_cand = 0

        for band in range(l):
            # Get this band's slice of the flat arrays
            start = band_offsets[band]
            end = band_offsets[band + 1]
            band_hashes = sorted_hashes_flat[start:end]
            band_indices = sorted_indices_flat[start:end]

            query_hash = query_bands[q, band]

            # Binary search for matching range
            left = _binary_search_left(band_hashes, query_hash)
            right = _binary_search_right(band_hashes, query_hash)

            # Add unique matches
            for i in range(left, right):
                idx = band_indices[i]

                # Check if seen
                is_seen = False
                for s in range(seen_count):
                    if seen[s] == idx:
                        is_seen = True
                        break

                if not is_seen:
                    if n_cand < max_results:
                        candidates[q, n_cand] = idx
                        n_cand += 1
                    if seen_count < len(seen):
                        seen[seen_count] = idx
                        seen_count += 1

                    if n_cand >= max_results:
                        break

            if n_cand >= max_results:
                break

        counts[q] = n_cand

    return candidates, counts
