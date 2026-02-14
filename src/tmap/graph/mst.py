"""
Minimum Spanning Tree construction from k-NN graph.
MST finds the tree that:
1. Connects all nodes
2. Minimizes total edge weight

For k-NN graphs, edge weight = distance. So MST connects
all points using the shortest possible total distance.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]
from scipy.sparse.csgraph import minimum_spanning_tree  # type: ignore[import-untyped]
from typing import cast

from tmap.graph.types import Tree
from tmap.index.types import KNNGraph

try:
    from numba import njit  # type: ignore[import-untyped]

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover - optional dependency
    njit = None
    _HAS_NUMBA = False


if _HAS_NUMBA:

    @njit(cache=True)  # type: ignore[untyped-decorator]
    def _reduce_sorted_min_numba(
        keys: NDArray[np.int64],
        u: NDArray[np.int32],
        v: NDArray[np.int32],
        w: NDArray[np.float32],
    ) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]:
        """Reduce sorted undirected edges to minimum weight per unique key."""
        n = keys.shape[0]
        if n == 0:
            return (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32),
            )

        out_u = np.empty(n, dtype=np.int32)
        out_v = np.empty(n, dtype=np.int32)
        out_w = np.empty(n, dtype=np.float32)

        cur_key = keys[0]
        cur_u = u[0]
        cur_v = v[0]
        cur_w = w[0]
        m = 0

        for i in range(1, n):
            if keys[i] != cur_key:
                out_u[m] = cur_u
                out_v[m] = cur_v
                out_w[m] = cur_w
                m += 1

                cur_key = keys[i]
                cur_u = u[i]
                cur_v = v[i]
                cur_w = w[i]
            elif w[i] < cur_w:
                cur_w = w[i]

        out_u[m] = cur_u
        out_v[m] = cur_v
        out_w[m] = cur_w
        m += 1

        return out_u[:m], out_v[:m], out_w[:m]


def _reduce_sorted_min_numpy(
    keys: NDArray[np.int64],
    u: NDArray[np.int32],
    v: NDArray[np.int32],
    w: NDArray[np.float32],
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]:
    """NumPy fallback for min-reduction over sorted undirected edge keys."""
    if keys.size == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )

    starts = np.concatenate(([0], np.flatnonzero(np.diff(keys)) + 1))
    w_min = np.minimum.reduceat(w, starts).astype(np.float32, copy=False)
    u_min = u[starts]
    v_min = v[starts]
    return u_min, v_min, w_min


def _reduce_sorted_min(
    keys: NDArray[np.int64],
    u: NDArray[np.int32],
    v: NDArray[np.int32],
    w: NDArray[np.float32],
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]:
    """Select the minimum edge weight for each sorted undirected edge key."""
    if _HAS_NUMBA:
        return cast(
            tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]],
            _reduce_sorted_min_numba(keys, u, v, w),
        )
    return _reduce_sorted_min_numpy(keys, u, v, w)


class MSTBuilder:
    """
    Build MST from k-NN graph.

    Example:
        knn = index.query_knn(k=20)
        builder = MSTBuilder(bias_factor=0.1)
        tree = builder.build(knn)
    """

    def __init__(self, bias_factor: float = 0.0) -> None:
        """
        Initialize MST builder.

        Args:
            bias_factor: How much to prefer closer neighbors (0-1).
                0.0 = standard MST (no bias)
                0.1 = slight preference for close neighbors
                0.5 = strong preference
                1.0 = very strong (may hurt global structure)

        Start with 0.0, increase if layout looks too "stringy".
        """
        if not 0.0 <= bias_factor <= 1.0:
            raise ValueError(f"bias_factor must be in [0, 1], got {bias_factor}")
        self.bias_factor = bias_factor

    def build(self, knn: KNNGraph) -> Tree:
        """
        Build MST from k-NN graph.

        Steps:
        1. Convert directed k-NN to undirected sparse adjacency matrix
           using union-min symmetrization (preserves one-way neighbors)
        2. Apply optional rank bias directly while building adjacency
        3. Run scipy's MST algorithm
        4. Extract edges and build Tree object
        """
        # Step 1-2: Build sparse adjacency matrix with optional bias
        adj = self._knn_to_sparse(knn)

        # Step 3: Compute MST using scipy
        # minimum_spanning_tree returns a sparse matrix with MST edges
        mst_sparse = minimum_spanning_tree(adj)

        # Step 4: Extract edges and weights
        edges, weights = self._sparse_to_edges(mst_sparse)

        # Find good root (highest degree node)
        root = self._find_root(edges, knn.n_nodes)

        return Tree(
            n_nodes=knn.n_nodes,
            edges=edges,
            weights=weights,
            root=root,
        )

    def _knn_to_sparse(self, knn: KNNGraph) -> csr_matrix:
        """
        Convert KNNGraph to scipy sparse matrix.

        k-NN is directional (i->neighbors[i]), but MST needs undirected edges.
        We build an undirected edge set by union of directed edges and keep
        the minimum observed weight per undirected pair.

        This avoids dropping one-way neighbors (a common source of fragmented
        forests when using strict mutual-kNN intersection).
        """
        n = knn.n_nodes
        k = knn.k

        # Build directed COO arrays from k-NN table.
        rows = np.repeat(np.arange(n, dtype=np.int32), k)
        cols = np.asarray(knn.indices.ravel(), dtype=np.int32)
        weights = np.asarray(knn.distances.ravel(), dtype=np.float32)

        # Remove invalid nodes and self-loops.
        valid = (cols >= 0) & (cols < n) & (cols != rows) & np.isfinite(weights)
        if not np.any(valid):
            return csr_matrix((n, n), dtype=np.float32)

        rows = rows[valid]
        cols = cols[valid]
        weights = weights[valid]

        # Apply rank bias directly on directed edges before undirected reduction.
        if self.bias_factor > 0:
            ranks = np.tile(np.arange(k, dtype=np.float32), n)[valid]
            weights = weights * (1.0 + self.bias_factor * (ranks / float(k)))

        # Canonicalize directed edges (i, j) and (j, i) to the same key.
        u = np.minimum(rows, cols).astype(np.int32, copy=False)
        v = np.maximum(rows, cols).astype(np.int32, copy=False)
        keys = u.astype(np.int64) * np.int64(n) + v.astype(np.int64)

        # Group by key and keep minimum weight per undirected pair.
        order = np.argsort(keys, kind="mergesort")
        keys_sorted = keys[order]
        u_sorted = u[order]
        v_sorted = v[order]
        w_sorted = weights[order]
        u_min, v_min, w_min = _reduce_sorted_min(keys_sorted, u_sorted, v_sorted, w_sorted)

        # Build symmetric sparse matrix for undirected MST.
        rows_sym = np.concatenate((u_min, v_min))
        cols_sym = np.concatenate((v_min, u_min))
        data_sym = np.concatenate((w_min, w_min)).astype(np.float32, copy=False)
        adj = csr_matrix((data_sym, (rows_sym, cols_sym)), shape=(n, n), dtype=np.float32)

        return adj

    def _sparse_to_edges(
        self,
        mst: csr_matrix,
    ) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        """Extract edge list from sparse MST matrix."""
        # Get non-zero entries
        rows, cols = mst.nonzero()

        if len(rows) == 0:
            return (
                np.empty((0, 2), dtype=np.int32),
                np.empty(0, dtype=np.float32),
            )

        weights = np.asarray(mst[rows, cols], dtype=np.float32).ravel()

        # Stack into edge array
        edges = np.column_stack([rows, cols]).astype(np.int32)
        return edges, weights

    def _find_root(self, edges: NDArray[np.int32], n_nodes: int) -> int:
        """
        Find good root node (highest degree).

        High-degree nodes make better roots because:
        - More balanced tree depth
        - Better for radial layouts
        """
        degree = np.zeros(n_nodes, dtype=np.int32)
        for src, tgt in edges:
            degree[src] += 1
            degree[tgt] += 1

        return int(np.argmax(degree))
