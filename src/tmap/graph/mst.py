"""
Minimum Spanning Tree construction from k-NN graph.

ALGORITHM BACKGROUND
--------------------
MST finds the tree that:
1. Connects all nodes
2. Minimizes total edge weight

For k-NN graphs, edge weight = distance. So MST connects
all points using the shortest possible total distance.

We use scipy.sparse.csgraph.minimum_spanning_tree which
implements Kruskal's algorithm with union-find. O(E log E).


MST BIAS FEATURE
----------------
Your requirement: "bias MST toward close NNs ab initio"

The idea: For visualization, we WANT close points to be connected.
But MST purely minimizes total weight, which might skip some
"obvious" connections.

Solution: Modify edge weights before MST:
    modified_weight = weight * (1 + bias_factor * rank / k)

Where rank = how close this neighbor is (0=closest, k-1=furthest).
This makes the MST prefer closer neighbors even if it costs more globally.


DESIGN NOTE: No inheritance here
--------------------------------
MSTBuilder is a simple class with one job. No need for ABC/interface.
If you later want multiple MST algorithms, THEN add abstraction.

"Make it work, make it right, make it fast" - in that order.
Don't over-engineer upfront.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from tmap.index.types import KNNGraph
from tmap.graph.types import Tree


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
        1. Convert k-NN to sparse adjacency matrix
        2. Apply bias to edge weights (if enabled)
        3. Run scipy's MST algorithm
        4. Extract edges and build Tree object
        """
        # Step 1: Build sparse adjacency matrix
        adj = self._knn_to_sparse(knn)

        # Step 2: Apply neighbor rank bias
        if self.bias_factor > 0:
            adj = self._apply_bias(adj, knn)

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

        k-NN is directional (i->neighbors[i]), but we want undirected.
        We symmetrize by taking minimum of (i,j) and (j,i) weights.
        """
        n = knn.n_nodes
        k = knn.k

        # Build COO format arrays
        rows = np.repeat(np.arange(n), k)
        cols = knn.indices.ravel()
        data = knn.distances.ravel()

        # Remove self-loops and invalid entries (-1)
        valid = (cols >= 0) & (cols != rows)
        rows = rows[valid]
        cols = cols[valid]
        data = data[valid]

        # Create sparse matrix
        adj = csr_matrix((data, (rows, cols)), shape=(n, n))

        # Symmetrize: adj = min(adj, adj.T)
        # For undirected graph, both directions should have same weight
        adj_t = adj.T
        adj = adj.minimum(adj_t)

        return adj

    def _apply_bias(self, adj: csr_matrix, knn: KNNGraph) -> csr_matrix:
        """
        Apply neighbor rank bias to edge weights.

        Closer neighbors (lower rank) get slightly lower weights,
        making MST prefer them even if it's not globally optimal.
        """
        n = knn.n_nodes
        k = knn.k

        # Create rank matrix: rank[i,j] = rank of j among i's neighbors
        rows = np.repeat(np.arange(n), k)
        cols = knn.indices.ravel()
        ranks = np.tile(np.arange(k), n)  # 0, 1, 2, ..., k-1 repeated

        valid = (cols >= 0) & (cols != rows)
        rows = rows[valid]
        cols = cols[valid]
        ranks = ranks[valid]

        # Bias factor: multiply weight by (1 + bias * rank/k)
        # Rank 0 -> multiply by 1.0 (no change)
        # Rank k-1 -> multiply by (1 + bias)
        bias_multiplier = 1.0 + self.bias_factor * (ranks / k)

        # Apply to adjacency matrix
        adj = adj.copy()
        for i, (r, c, mult) in enumerate(zip(rows, cols, bias_multiplier)):
            if adj[r, c] > 0:
                adj[r, c] *= mult

        return adj

    def _sparse_to_edges(
        self,
        mst: csr_matrix,
    ) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        """Extract edge list from sparse MST matrix."""
        # Get non-zero entries
        rows, cols = mst.nonzero()
        weights = np.array(mst[rows, cols]).ravel()

        # Stack into edge array
        edges = np.column_stack([rows, cols]).astype(np.int32)
        weights = weights.astype(np.float32)

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
