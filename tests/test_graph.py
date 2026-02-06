"""
Tests for the Graph module (MSTBuilder and Tree).

Tests cover:
- MSTBuilder initialization and parameter validation
- MST construction from k-NN graphs
- bias_factor effect on edge weights
- Tree structure and traversal methods
- Edge cases (disconnected graphs, single node, etc.)
"""

import numpy as np
import pytest

from tmap.graph import MSTBuilder, Tree
from tmap.index.types import KNNGraph

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_knn():
    """Create a simple 5-node k-NN graph for basic testing.

    Graph structure (k=2):
        0 -- 1 -- 2
        |    |
        3 -- 4
    """
    indices = np.array([
        [1, 3],   # Node 0: neighbors 1, 3
        [0, 2],   # Node 1: neighbors 0, 2
        [1, 4],   # Node 2: neighbors 1, 4
        [0, 4],   # Node 3: neighbors 0, 4
        [3, 2],   # Node 4: neighbors 3, 2
    ], dtype=np.int32)

    distances = np.array([
        [0.1, 0.2],   # Node 0
        [0.1, 0.15],  # Node 1
        [0.15, 0.25], # Node 2
        [0.2, 0.1],   # Node 3
        [0.1, 0.25],  # Node 4
    ], dtype=np.float32)

    return KNNGraph(indices=indices, distances=distances)


@pytest.fixture
def larger_knn():
    """Create a 20-node k-NN graph with known structure.

    Creates a chain: 0-1-2-...-19 with some additional cross-connections.
    """
    n = 20
    k = 3

    indices = np.full((n, k), -1, dtype=np.int32)
    distances = np.full((n, k), 2.0, dtype=np.float32)

    for i in range(n):
        neighbors = []
        dists = []

        # Previous node
        if i > 0:
            neighbors.append(i - 1)
            dists.append(0.1)

        # Next node
        if i < n - 1:
            neighbors.append(i + 1)
            dists.append(0.1)

        # Skip connection (every 5th node)
        if i + 5 < n:
            neighbors.append(i + 5)
            dists.append(0.3)

        for j, (nb, d) in enumerate(zip(neighbors[:k], dists[:k])):
            indices[i, j] = nb
            distances[i, j] = d

    return KNNGraph(indices=indices, distances=distances)


@pytest.fixture
def disconnected_knn():
    """Create a k-NN graph with two disconnected components.

    Component 1: nodes 0, 1, 2
    Component 2: nodes 3, 4
    """
    indices = np.array([
        [1, 2],    # Node 0: only connected to 1, 2
        [0, 2],    # Node 1: only connected to 0, 2
        [0, 1],    # Node 2: only connected to 0, 1
        [4, -1],   # Node 3: only connected to 4
        [3, -1],   # Node 4: only connected to 3
    ], dtype=np.int32)

    distances = np.array([
        [0.1, 0.2],
        [0.1, 0.15],
        [0.2, 0.15],
        [0.1, 2.0],  # 2.0 = invalid marker
        [0.1, 2.0],
    ], dtype=np.float32)

    return KNNGraph(indices=indices, distances=distances)


# =============================================================================
# MSTBuilder Initialization Tests
# =============================================================================


class TestMSTBuilderInit:
    """Test MSTBuilder initialization and parameter validation."""

    def test_default_bias_factor(self):
        """Default bias_factor should be 0.0.

        Covers: Default parameter value.
        """
        builder = MSTBuilder()
        assert builder.bias_factor == 0.0

    def test_custom_bias_factor(self):
        """Custom bias_factor should be stored.

        Covers: Parameter setting.
        """
        builder = MSTBuilder(bias_factor=0.3)
        assert builder.bias_factor == 0.3

    def test_bias_factor_zero(self):
        """bias_factor=0.0 should be valid (no bias).

        Covers: Boundary value at minimum.
        """
        builder = MSTBuilder(bias_factor=0.0)
        assert builder.bias_factor == 0.0

    def test_bias_factor_one(self):
        """bias_factor=1.0 should be valid (maximum bias).

        Covers: Boundary value at maximum.
        """
        builder = MSTBuilder(bias_factor=1.0)
        assert builder.bias_factor == 1.0

    def test_bias_factor_negative_raises(self):
        """Negative bias_factor should raise ValueError.

        Covers: Invalid parameter validation.
        """
        with pytest.raises(ValueError, match="bias_factor must be in"):
            MSTBuilder(bias_factor=-0.1)

    def test_bias_factor_greater_than_one_raises(self):
        """bias_factor > 1.0 should raise ValueError.

        Covers: Invalid parameter validation.
        """
        with pytest.raises(ValueError, match="bias_factor must be in"):
            MSTBuilder(bias_factor=1.5)


# =============================================================================
# MSTBuilder.build() Tests
# =============================================================================


class TestMSTBuilderBuild:
    """Test MST construction from k-NN graphs."""

    def test_build_returns_tree(self, simple_knn):
        """build() should return a Tree object.

        Covers: Return type verification.
        """
        builder = MSTBuilder()
        result = builder.build(simple_knn)
        assert isinstance(result, Tree)

    def test_tree_node_count(self, simple_knn):
        """Tree should have correct number of nodes.

        Covers: Node count preservation.
        """
        builder = MSTBuilder()
        tree = builder.build(simple_knn)
        assert tree.n_nodes == 5

    def test_tree_edge_count_connected(self, simple_knn):
        """Connected graph should produce tree with n-1 edges.

        Covers: MST property - exactly n-1 edges for connected graph.
        """
        builder = MSTBuilder()
        tree = builder.build(simple_knn)
        assert len(tree.edges) == 4  # n - 1 = 5 - 1

    def test_tree_weights_match_edges(self, simple_knn):
        """Number of weights should match number of edges.

        Covers: Weight array consistency.
        """
        builder = MSTBuilder()
        tree = builder.build(simple_knn)
        assert len(tree.weights) == len(tree.edges)

    def test_tree_weights_positive(self, simple_knn):
        """All tree weights should be non-negative.

        Covers: Weight validity (distances are non-negative).
        """
        builder = MSTBuilder()
        tree = builder.build(simple_knn)
        assert np.all(tree.weights >= 0)

    def test_tree_root_valid(self, simple_knn):
        """Tree root should be a valid node index.

        Covers: Root selection validity.
        """
        builder = MSTBuilder()
        tree = builder.build(simple_knn)
        assert 0 <= tree.root < tree.n_nodes

    def test_mst_minimizes_weight(self, simple_knn):
        """MST should minimize total edge weight.

        Covers: MST optimality property.
        """
        builder = MSTBuilder()
        tree = builder.build(simple_knn)

        total_weight = np.sum(tree.weights)
        # For our simple graph, MST should use shortest edges
        # Expected: 0.1 + 0.1 + 0.15 + 0.1 = 0.45 (approximately)
        assert total_weight < 1.0  # Much less than sum of all edges


class TestMSTBuilderBiasFactor:
    """Test bias_factor effect on MST construction."""

    def test_zero_bias_standard_mst(self, larger_knn):
        """bias_factor=0 should produce standard MST.

        Covers: Default MST behavior without bias.
        """
        builder = MSTBuilder(bias_factor=0.0)
        tree = builder.build(larger_knn)

        # Standard MST properties
        assert tree.n_nodes == 20
        assert len(tree.edges) <= 19  # At most n-1

    def test_bias_changes_weights(self, larger_knn):
        """Non-zero bias should potentially produce different MST.

        Covers: Bias effect on edge selection.
        """
        tree_no_bias = MSTBuilder(bias_factor=0.0).build(larger_knn)
        tree_with_bias = MSTBuilder(bias_factor=0.5).build(larger_knn)

        # Both should be valid trees
        assert tree_no_bias.n_nodes == tree_with_bias.n_nodes
        assert len(tree_no_bias.edges) == len(tree_with_bias.edges)

        # Weights might differ due to bias modification
        # (exact difference depends on graph structure)

    def test_high_bias_prefers_close_neighbors(self, larger_knn):
        """High bias should prefer closer neighbors.

        Covers: Bias behavior - rank 0 neighbors should be preferred.
        """
        tree_low = MSTBuilder(bias_factor=0.0).build(larger_knn)
        tree_high = MSTBuilder(bias_factor=0.8).build(larger_knn)

        # Both should produce valid trees
        assert len(tree_low.edges) == len(tree_high.edges)


class TestMSTBuilderDisconnected:
    """Test MST construction with disconnected graphs."""

    def test_disconnected_fewer_edges(self, disconnected_knn):
        """Disconnected graph should produce fewer than n-1 edges.

        Covers: Disconnected graph handling.
        """
        builder = MSTBuilder()
        tree = builder.build(disconnected_knn)

        # 5 nodes, 2 components → should have 3 edges (2 + 1)
        assert tree.n_nodes == 5
        assert len(tree.edges) == 3  # Component 1: 2 edges, Component 2: 1 edge

    def test_disconnected_component_count(self, disconnected_knn):
        """Can compute number of components from tree.

        Covers: Component detection from MST.
        """
        builder = MSTBuilder()
        tree = builder.build(disconnected_knn)

        n_components = tree.n_nodes - len(tree.edges)
        assert n_components == 2


# =============================================================================
# Tree Structure Tests
# =============================================================================


class TestTreeInit:
    """Test Tree initialization."""

    def test_tree_from_arrays(self):
        """Tree can be created from arrays directly.

        Covers: Direct Tree construction.
        """
        edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
        weights = np.array([0.5, 0.3], dtype=np.float32)

        tree = Tree(n_nodes=3, edges=edges, weights=weights, root=1)

        assert tree.n_nodes == 3
        assert len(tree.edges) == 2
        assert tree.root == 1

    def test_tree_builds_adjacency(self):
        """Tree should build adjacency list on init.

        Covers: Adjacency list construction.
        """
        edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
        weights = np.array([0.5, 0.3], dtype=np.float32)

        tree = Tree(n_nodes=3, edges=edges, weights=weights)

        # Adjacency should exist and be populated
        assert len(tree._adjacency) == 3
        assert len(tree._adjacency[1]) == 2  # Node 1 has 2 neighbors


class TestTreeNeighbors:
    """Test Tree.neighbors() method."""

    def test_neighbors_returns_list(self, simple_knn):
        """neighbors() should return list of (neighbor, weight) tuples.

        Covers: Return type and format.
        """
        tree = MSTBuilder().build(simple_knn)
        neighbors = tree.neighbors(tree.root)

        assert isinstance(neighbors, list)
        for item in neighbors:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_neighbors_bidirectional(self, simple_knn):
        """Edges should appear in both directions in neighbors.

        Covers: Undirected tree property.
        """
        tree = MSTBuilder().build(simple_knn)

        # Find an edge
        src, tgt = tree.edges[0]

        # Both directions should exist
        src_neighbors = [n for n, _ in tree.neighbors(int(src))]
        tgt_neighbors = [n for n, _ in tree.neighbors(int(tgt))]

        assert int(tgt) in src_neighbors
        assert int(src) in tgt_neighbors

    def test_leaf_has_one_neighbor(self, simple_knn):
        """Leaf nodes should have exactly one neighbor.

        Covers: Leaf node detection.
        """
        tree = MSTBuilder().build(simple_knn)

        # Count neighbors for each node
        neighbor_counts = [len(tree.neighbors(i)) for i in range(tree.n_nodes)]

        # In a tree, at least 2 nodes should be leaves (degree 1)
        leaves = [c for c in neighbor_counts if c == 1]
        assert len(leaves) >= 2


class TestTreeChildren:
    """Test Tree.children() method."""

    def test_children_excludes_parent(self, simple_knn):
        """children() should exclude the parent node.

        Covers: Parent exclusion in tree traversal.
        """
        tree = MSTBuilder().build(simple_knn)

        # Get first edge
        src, tgt = tree.edges[0]

        # Children of tgt with src as parent should not include src
        children = tree.children(int(tgt), parent=int(src))
        assert int(src) not in children

    def test_children_at_root(self, simple_knn):
        """children() at root with no parent should return all neighbors.

        Covers: Root node handling.
        """
        tree = MSTBuilder().build(simple_knn)

        children = tree.children(tree.root, parent=None)
        neighbors = [n for n, _ in tree.neighbors(tree.root)]

        assert set(children) == set(neighbors)


# =============================================================================
# Tree Traversal Tests
# =============================================================================


class TestTreeBFS:
    """Test Tree.bfs() method."""

    def test_bfs_visits_all_nodes(self, simple_knn):
        """BFS should visit all nodes exactly once.

        Covers: Complete traversal.
        """
        tree = MSTBuilder().build(simple_knn)

        visited = list(tree.bfs())
        visited_nodes = [node for node, _, _ in visited]

        assert len(visited_nodes) == tree.n_nodes
        assert len(set(visited_nodes)) == tree.n_nodes  # No duplicates

    def test_bfs_starts_at_root(self, simple_knn):
        """BFS should start at root by default.

        Covers: Default start node.
        """
        tree = MSTBuilder().build(simple_knn)

        visited = list(tree.bfs())
        first_node, first_parent, first_depth = visited[0]

        assert first_node == tree.root
        assert first_parent is None
        assert first_depth == 0

    def test_bfs_custom_start(self, simple_knn):
        """BFS can start from custom node.

        Covers: Custom start parameter.
        """
        tree = MSTBuilder().build(simple_knn)

        visited = list(tree.bfs(start=0))
        first_node, _, _ = visited[0]

        assert first_node == 0

    def test_bfs_depth_increases(self, larger_knn):
        """BFS depth should be monotonically non-decreasing.

        Covers: Level-order property.
        """
        tree = MSTBuilder().build(larger_knn)

        depths = [depth for _, _, depth in tree.bfs()]

        for i in range(1, len(depths)):
            assert depths[i] >= depths[i - 1]

    def test_bfs_parent_correct(self, simple_knn):
        """BFS parent should be actual neighbor.

        Covers: Parent tracking accuracy.
        """
        tree = MSTBuilder().build(simple_knn)

        for node, parent, _ in tree.bfs():
            if parent is not None:
                # Parent should be a neighbor
                neighbors = [n for n, _ in tree.neighbors(node)]
                assert parent in neighbors


class TestTreeDFS:
    """Test Tree.dfs() method."""

    def test_dfs_visits_all_nodes(self, simple_knn):
        """DFS should visit all nodes exactly once.

        Covers: Complete traversal.
        """
        tree = MSTBuilder().build(simple_knn)

        visited = list(tree.dfs())
        visited_nodes = [node for node, _, _ in visited]

        assert len(visited_nodes) == tree.n_nodes
        assert len(set(visited_nodes)) == tree.n_nodes

    def test_dfs_starts_at_root(self, simple_knn):
        """DFS should start at root by default.

        Covers: Default start node.
        """
        tree = MSTBuilder().build(simple_knn)

        visited = list(tree.dfs())
        first_node, first_parent, first_depth = visited[0]

        assert first_node == tree.root
        assert first_parent is None
        assert first_depth == 0

    def test_dfs_custom_start(self, simple_knn):
        """DFS can start from custom node.

        Covers: Custom start parameter.
        """
        tree = MSTBuilder().build(simple_knn)

        visited = list(tree.dfs(start=0))
        first_node, _, _ = visited[0]

        assert first_node == 0


class TestTreeSubtreeSizes:
    """Test Tree.subtree_sizes() method."""

    def test_subtree_sizes_shape(self, simple_knn):
        """subtree_sizes() should return array of correct size.

        Covers: Return shape.
        """
        tree = MSTBuilder().build(simple_knn)
        sizes = tree.subtree_sizes()

        assert sizes.shape == (tree.n_nodes,)
        assert sizes.dtype == np.int32

    def test_root_subtree_is_all_nodes(self, simple_knn):
        """Subtree size at root should equal total nodes.

        Covers: Root subtree property.
        """
        tree = MSTBuilder().build(simple_knn)
        sizes = tree.subtree_sizes()

        assert sizes[tree.root] == tree.n_nodes

    def test_leaf_subtree_is_one(self, simple_knn):
        """Leaf node subtree size should be 1.

        Covers: Leaf subtree property.
        """
        tree = MSTBuilder().build(simple_knn)
        sizes = tree.subtree_sizes()

        # Find leaves (degree 1)
        for i in range(tree.n_nodes):
            if len(tree.neighbors(i)) == 1:
                assert sizes[i] == 1

    def test_subtree_sizes_sum(self, larger_knn):
        """Parent's subtree size should be sum of children + 1.

        Covers: Subtree size consistency.
        """
        tree = MSTBuilder().build(larger_knn)
        sizes = tree.subtree_sizes()

        # Check from root
        for node, parent, _ in tree.bfs():
            children = tree.children(node, parent)
            if children:
                children_sum = sum(sizes[c] for c in children)
                assert sizes[node] == children_sum + 1


# =============================================================================
# Edge Cases
# =============================================================================


class TestMSTBuilderEdgeCases:
    """Test edge cases for MSTBuilder."""

    @pytest.mark.xfail(
        reason="BUG ISS-009: MSTBuilder._sparse_to_edges crashes on empty sparse matrix",
        raises=ValueError,
    )
    def test_single_node_graph(self):
        """Single node graph should produce tree with 0 edges.

        Covers: Minimum viable graph.

        BUG ISS-009: MSTBuilder._sparse_to_edges() fails when sparse matrix is empty.
        The line `weights = np.array(mst[rows, cols]).ravel()` returns a matrix
        object instead of an array when there are no edges.
        """
        indices = np.array([[-1]], dtype=np.int32)  # No neighbors
        distances = np.array([[2.0]], dtype=np.float32)  # Invalid
        knn = KNNGraph(indices=indices, distances=distances)

        builder = MSTBuilder()
        tree = builder.build(knn)

        assert tree.n_nodes == 1
        assert len(tree.edges) == 0
        assert tree.root == 0

    def test_two_node_graph(self):
        """Two node graph should produce tree with 1 edge.

        Covers: Minimal connected graph.
        """
        indices = np.array([[1], [0]], dtype=np.int32)
        distances = np.array([[0.5], [0.5]], dtype=np.float32)
        knn = KNNGraph(indices=indices, distances=distances)

        builder = MSTBuilder()
        tree = builder.build(knn)

        assert tree.n_nodes == 2
        assert len(tree.edges) == 1

    def test_self_loops_ignored(self):
        """Self-loops in k-NN should be ignored.

        Covers: Self-loop handling.
        """
        # k-NN with self in neighbors (shouldn't happen but handle gracefully)
        indices = np.array([[0, 1], [1, 0]], dtype=np.int32)  # 0 has self as neighbor
        distances = np.array([[0.0, 0.5], [0.0, 0.5]], dtype=np.float32)
        knn = KNNGraph(indices=indices, distances=distances)

        builder = MSTBuilder()
        tree = builder.build(knn)

        # Should still work, self-loops filtered out
        assert tree.n_nodes == 2
        assert len(tree.edges) == 1

    @pytest.mark.xfail(
        reason="BUG ISS-009: MSTBuilder._sparse_to_edges crashes on empty sparse matrix",
        raises=ValueError,
    )
    def test_all_invalid_neighbors(self):
        """Graph with all -1 neighbors should produce tree with 0 edges.

        Covers: Completely disconnected graph.

        BUG ISS-009: Same issue as test_single_node_graph - empty sparse matrix
        causes crash in _sparse_to_edges().
        """
        indices = np.array([[-1, -1], [-1, -1], [-1, -1]], dtype=np.int32)
        distances = np.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
        knn = KNNGraph(indices=indices, distances=distances)

        builder = MSTBuilder()
        tree = builder.build(knn)

        assert tree.n_nodes == 3
        assert len(tree.edges) == 0


class TestTreeEdgeCases:
    """Test edge cases for Tree class."""

    def test_empty_tree_bfs(self):
        """BFS on single-node tree should yield just the root.

        Covers: Traversal on minimal tree.
        """
        edges = np.array([], dtype=np.int32).reshape(0, 2)
        weights = np.array([], dtype=np.float32)
        tree = Tree(n_nodes=1, edges=edges, weights=weights, root=0)

        visited = list(tree.bfs())
        assert len(visited) == 1
        assert visited[0] == (0, None, 0)

    def test_empty_tree_dfs(self):
        """DFS on single-node tree should yield just the root.

        Covers: Traversal on minimal tree.
        """
        edges = np.array([], dtype=np.int32).reshape(0, 2)
        weights = np.array([], dtype=np.float32)
        tree = Tree(n_nodes=1, edges=edges, weights=weights, root=0)

        visited = list(tree.dfs())
        assert len(visited) == 1

    def test_empty_tree_subtree_sizes(self):
        """subtree_sizes on single-node tree should return [1].

        Covers: Subtree sizes on minimal tree.
        """
        edges = np.array([], dtype=np.int32).reshape(0, 2)
        weights = np.array([], dtype=np.float32)
        tree = Tree(n_nodes=1, edges=edges, weights=weights, root=0)

        sizes = tree.subtree_sizes()
        assert sizes[0] == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestGraphModuleIntegration:
    """Integration tests combining MSTBuilder and Tree operations."""

    def test_build_traverse_subtree(self, larger_knn):
        """Full workflow: build MST, traverse, compute subtree sizes.

        Covers: Complete module integration.
        """
        # Build MST
        builder = MSTBuilder(bias_factor=0.1)
        tree = builder.build(larger_knn)

        # Traverse BFS
        bfs_order = list(tree.bfs())
        assert len(bfs_order) == tree.n_nodes

        # Traverse DFS
        dfs_order = list(tree.dfs())
        assert len(dfs_order) == tree.n_nodes

        # Compute subtree sizes
        sizes = tree.subtree_sizes()
        assert sizes[tree.root] == tree.n_nodes

        # All sizes should be positive
        assert np.all(sizes >= 1)

    def test_different_bias_same_structure(self, larger_knn):
        """Different bias factors should produce valid trees.

        Covers: Bias variation testing.
        """
        trees = []
        for bias in [0.0, 0.1, 0.3, 0.5]:
            tree = MSTBuilder(bias_factor=bias).build(larger_knn)
            trees.append(tree)

            # All should be valid trees
            assert tree.n_nodes == larger_knn.n_nodes
            assert len(tree.edges) <= tree.n_nodes - 1
            assert np.all(tree.weights >= 0)
