"""Tests for tree extraction and Tree traversal helpers."""

import numpy as np
import pytest

from tmap.graph import Tree, tree_from_knn_graph
from tmap.index.types import KNNGraph
from tmap.layout import OGDF_AVAILABLE


@pytest.fixture
def simple_knn():
    """Create a simple 5-node connected k-NN graph."""
    indices = np.array(
        [
            [1, 3],
            [0, 2],
            [1, 4],
            [0, 4],
            [3, 2],
        ],
        dtype=np.int32,
    )
    distances = np.array(
        [
            [0.1, 0.2],
            [0.1, 0.15],
            [0.15, 0.25],
            [0.2, 0.1],
            [0.1, 0.25],
        ],
        dtype=np.float32,
    )
    return KNNGraph(indices=indices, distances=distances)


@pytest.fixture
def disconnected_knn():
    """Create a k-NN graph with two disconnected components."""
    indices = np.array(
        [
            [1, 2],
            [0, 2],
            [0, 1],
            [4, -1],
            [3, -1],
        ],
        dtype=np.int32,
    )
    distances = np.array(
        [
            [0.1, 0.2],
            [0.1, 0.15],
            [0.2, 0.15],
            [0.1, 2.0],
            [0.1, 2.0],
        ],
        dtype=np.float32,
    )
    return KNNGraph(indices=indices, distances=distances)


@pytest.fixture
def sample_tree():
    """Manual tree used for traversal and neighbor tests."""
    edges = np.array([[1, 0], [1, 2], [0, 3], [3, 4]], dtype=np.int32)
    weights = np.array([0.1, 0.15, 0.2, 0.1], dtype=np.float32)
    return Tree(n_nodes=5, edges=edges, weights=weights, root=1)


def _make_chain_tree(n: int, weight: float = 1.0) -> Tree:
    """Chain: 0-1-2-..-(n-1)."""
    edges = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.int32)
    weights = np.full(n - 1, weight, dtype=np.float32)
    return Tree(n_nodes=n, edges=edges, weights=weights, root=0)


def _make_star_tree(n: int, weight: float = 1.0) -> Tree:
    """Star: 0 connected to 1,2,..,n-1."""
    edges = np.array([[0, i] for i in range(1, n)], dtype=np.int32)
    weights = np.full(n - 1, weight, dtype=np.float32)
    return Tree(n_nodes=n, edges=edges, weights=weights, root=0)


def _make_disconnected_tree() -> Tree:
    """Two components: 0-1-2 and 3-4."""
    edges = np.array([[0, 1], [1, 2], [3, 4]], dtype=np.int32)
    weights = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    return Tree(n_nodes=5, edges=edges, weights=weights, root=0)


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
class TestTreeFromKNNGraph:
    """Test OGDF-backed tree extraction from k-NN graphs."""

    def test_returns_tree(self, simple_knn):
        tree = tree_from_knn_graph(simple_knn)
        assert isinstance(tree, Tree)

    def test_connected_graph_has_n_minus_one_edges(self, simple_knn):
        tree = tree_from_knn_graph(simple_knn)
        assert tree.n_nodes == 5
        assert len(tree.edges) == 4
        assert len(tree.weights) == 4

    def test_disconnected_graph_has_fewer_edges(self, disconnected_knn):
        tree = tree_from_knn_graph(disconnected_knn)
        assert tree.n_nodes == 5
        assert len(tree.edges) == 3

    def test_weights_are_non_negative(self, simple_knn):
        tree = tree_from_knn_graph(simple_knn)
        assert np.all(tree.weights >= 0)

    def test_single_node_graph(self):
        knn = KNNGraph(
            indices=np.array([[-1]], dtype=np.int32),
            distances=np.array([[2.0]], dtype=np.float32),
        )
        tree = tree_from_knn_graph(knn)
        assert tree.n_nodes == 1
        assert len(tree.edges) == 0
        assert tree.root == 0

    def test_self_loops_ignored(self):
        knn = KNNGraph(
            indices=np.array([[0, 1], [1, 0]], dtype=np.int32),
            distances=np.array([[0.0, 0.5], [0.0, 0.5]], dtype=np.float32),
        )
        tree = tree_from_knn_graph(knn)
        assert tree.n_nodes == 2
        assert len(tree.edges) == 1


class TestTreeInit:
    """Test Tree initialization."""

    def test_tree_from_arrays(self):
        edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
        weights = np.array([0.5, 0.3], dtype=np.float32)
        tree = Tree(n_nodes=3, edges=edges, weights=weights, root=1)

        assert tree.n_nodes == 3
        assert len(tree.edges) == 2
        assert tree.root == 1

    def test_tree_builds_adjacency(self):
        edges = np.array([[0, 1], [1, 2]], dtype=np.int32)
        weights = np.array([0.5, 0.3], dtype=np.float32)
        tree = Tree(n_nodes=3, edges=edges, weights=weights)

        assert len(tree._adjacency) == 3
        assert len(tree._adjacency[1]) == 2


class TestTreeNeighbors:
    """Test Tree.neighbors()."""

    def test_neighbors_returns_list(self, sample_tree):
        neighbors = sample_tree.neighbors(sample_tree.root)
        assert isinstance(neighbors, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in neighbors)

    def test_neighbors_bidirectional(self, sample_tree):
        src, tgt = sample_tree.edges[0]
        src_neighbors = [n for n, _ in sample_tree.neighbors(int(src))]
        tgt_neighbors = [n for n, _ in sample_tree.neighbors(int(tgt))]

        assert int(tgt) in src_neighbors
        assert int(src) in tgt_neighbors

    def test_leaf_has_one_neighbor(self, sample_tree):
        neighbor_counts = [len(sample_tree.neighbors(i)) for i in range(sample_tree.n_nodes)]
        leaves = [count for count in neighbor_counts if count == 1]
        assert len(leaves) >= 2


class TestTreeChildren:
    """Test Tree.children()."""

    def test_children_excludes_parent(self, sample_tree):
        src, tgt = sample_tree.edges[0]
        children = sample_tree.children(int(tgt), parent=int(src))
        assert int(src) not in children

    def test_children_at_root(self, sample_tree):
        children = sample_tree.children(sample_tree.root, parent=None)
        neighbors = [n for n, _ in sample_tree.neighbors(sample_tree.root)]
        assert set(children) == set(neighbors)


class TestTreeBFS:
    """Test Tree.bfs()."""

    def test_bfs_visits_all_nodes(self, sample_tree):
        visited_nodes = [node for node, _, _ in sample_tree.bfs()]
        assert len(visited_nodes) == sample_tree.n_nodes
        assert len(set(visited_nodes)) == sample_tree.n_nodes

    def test_bfs_starts_at_root(self, sample_tree):
        first_node, first_parent, first_depth = next(sample_tree.bfs())
        assert first_node == sample_tree.root
        assert first_parent is None
        assert first_depth == 0

    def test_bfs_custom_start(self, sample_tree):
        first_node, _, _ = next(sample_tree.bfs(start=0))
        assert first_node == 0

    def test_bfs_depth_increases(self):
        depths = [depth for _, _, depth in _make_chain_tree(20).bfs()]
        for i in range(1, len(depths)):
            assert depths[i] >= depths[i - 1]

    def test_bfs_parent_is_neighbor(self, sample_tree):
        for node, parent, _ in sample_tree.bfs():
            if parent is None:
                continue
            neighbors = [n for n, _ in sample_tree.neighbors(node)]
            assert parent in neighbors


class TestTreeDFS:
    """Test Tree.dfs()."""

    def test_dfs_visits_all_nodes(self, sample_tree):
        visited_nodes = [node for node, _, _ in sample_tree.dfs()]
        assert len(visited_nodes) == sample_tree.n_nodes
        assert len(set(visited_nodes)) == sample_tree.n_nodes

    def test_dfs_starts_at_root(self, sample_tree):
        first_node, first_parent, first_depth = next(sample_tree.dfs())
        assert first_node == sample_tree.root
        assert first_parent is None
        assert first_depth == 0

    def test_dfs_custom_start(self, sample_tree):
        first_node, _, _ = next(sample_tree.dfs(start=0))
        assert first_node == 0


class TestTreeSubtreeSizes:
    """Test Tree.subtree_sizes()."""

    def test_subtree_sizes_shape(self, sample_tree):
        sizes = sample_tree.subtree_sizes()
        assert sizes.shape == (sample_tree.n_nodes,)
        assert sizes.dtype == np.int32

    def test_root_subtree_is_all_nodes(self, sample_tree):
        sizes = sample_tree.subtree_sizes()
        assert sizes[sample_tree.root] == sample_tree.n_nodes

    def test_leaf_subtree_is_one(self, sample_tree):
        sizes = sample_tree.subtree_sizes()
        for i in range(sample_tree.n_nodes):
            if len(sample_tree.neighbors(i)) == 1:
                assert sizes[i] == 1

    def test_parent_subtree_is_children_plus_one(self):
        tree = _make_chain_tree(20)
        sizes = tree.subtree_sizes()
        for node, parent, _ in tree.bfs():
            children = tree.children(node, parent)
            if children:
                assert sizes[node] == sum(sizes[child] for child in children) + 1


class TestTreeEdgeCases:
    """Test edge cases for Tree."""

    def test_empty_tree_bfs(self):
        tree = Tree(
            n_nodes=1,
            edges=np.empty((0, 2), dtype=np.int32),
            weights=np.empty(0, dtype=np.float32),
            root=0,
        )
        assert list(tree.bfs()) == [(0, None, 0)]

    def test_empty_tree_dfs(self):
        tree = Tree(
            n_nodes=1,
            edges=np.empty((0, 2), dtype=np.int32),
            weights=np.empty(0, dtype=np.float32),
            root=0,
        )
        assert list(tree.dfs()) == [(0, None, 0)]

    def test_empty_tree_subtree_sizes(self):
        tree = Tree(
            n_nodes=1,
            edges=np.empty((0, 2), dtype=np.int32),
            weights=np.empty(0, dtype=np.float32),
            root=0,
        )
        assert tree.subtree_sizes()[0] == 1


class TestGraphModuleIntegration:
    """Integration tests combining tree extraction and traversal."""

    @pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
    def test_extract_traverse_and_measure(self, simple_knn):
        tree = tree_from_knn_graph(simple_knn)

        bfs_order = list(tree.bfs())
        dfs_order = list(tree.dfs())
        sizes = tree.subtree_sizes()

        assert len(bfs_order) == tree.n_nodes
        assert len(dfs_order) == tree.n_nodes
        assert sizes[tree.root] == tree.n_nodes
        assert np.all(sizes >= 1)


class TestTreePath:
    """Tests for Tree.path()."""

    def test_same_node(self):
        tree = _make_chain_tree(5)
        assert tree.path(2, 2) == [2]

    def test_adjacent(self):
        tree = _make_chain_tree(5)
        assert tree.path(0, 1) == [0, 1]

    def test_chain_forward(self):
        tree = _make_chain_tree(5)
        assert tree.path(0, 4) == [0, 1, 2, 3, 4]

    def test_chain_reverse(self):
        tree = _make_chain_tree(5)
        assert tree.path(4, 0) == [4, 3, 2, 1, 0]

    def test_star_through_hub(self):
        tree = _make_star_tree(5)
        assert tree.path(1, 3) == [1, 0, 3]

    def test_invalid_from_idx(self):
        tree = _make_chain_tree(3)
        with pytest.raises(ValueError, match="from_idx"):
            tree.path(-1, 1)

    def test_invalid_to_idx(self):
        tree = _make_chain_tree(3)
        with pytest.raises(ValueError, match="to_idx"):
            tree.path(0, 10)

    def test_disconnected_raises(self):
        tree = _make_disconnected_tree()
        with pytest.raises(IndexError, match="No path"):
            tree.path(0, 4)


class TestTreeDistance:
    """Tests for Tree.distance()."""

    def test_self_distance_zero(self):
        tree = _make_chain_tree(5, weight=2.0)
        assert tree.distance(2, 2) == 0.0

    def test_adjacent_distance(self):
        tree = _make_chain_tree(5, weight=1.5)
        assert tree.distance(0, 1) == pytest.approx(1.5)

    def test_chain_sum(self):
        tree = _make_chain_tree(5, weight=1.0)
        assert tree.distance(0, 4) == pytest.approx(4.0)

    def test_symmetry(self):
        tree = _make_chain_tree(5, weight=0.5)
        assert tree.distance(1, 3) == pytest.approx(tree.distance(3, 1))

    def test_non_uniform_weights(self):
        tree = _make_disconnected_tree()
        assert tree.distance(0, 2) == pytest.approx(3.0)

    def test_disconnected_raises(self):
        tree = _make_disconnected_tree()
        with pytest.raises(IndexError, match="No path"):
            tree.distance(0, 4)


class TestTreeSubtree:
    """Tests for Tree.subtree()."""

    def test_depth_zero(self):
        tree = _make_chain_tree(5)
        assert tree.subtree(2, depth=0) == [2]

    def test_depth_one(self):
        tree = _make_chain_tree(5)
        result = tree.subtree(2, depth=1)
        assert set(result) == {1, 2, 3}
        assert result[0] == 2

    def test_unlimited(self):
        tree = _make_chain_tree(5)
        assert set(tree.subtree(0)) == {0, 1, 2, 3, 4}

    def test_star_from_hub(self):
        tree = _make_star_tree(5)
        assert set(tree.subtree(0, depth=1)) == {0, 1, 2, 3, 4}

    def test_invalid_index(self):
        tree = _make_chain_tree(3)
        with pytest.raises(ValueError, match="node_idx"):
            tree.subtree(10)


class TestTreeDistancesFrom:
    """Tests for Tree.distances_from()."""

    def test_source_zero(self):
        tree = _make_chain_tree(4, weight=1.0)
        np.testing.assert_allclose(tree.distances_from(0), [0.0, 1.0, 2.0, 3.0])

    def test_shape_and_dtype(self):
        dists = _make_chain_tree(5).distances_from(0)
        assert dists.shape == (5,)
        assert dists.dtype == np.float32

    def test_middle_start(self):
        tree = _make_chain_tree(5, weight=1.0)
        np.testing.assert_allclose(tree.distances_from(2), [2.0, 1.0, 0.0, 1.0, 2.0])

    def test_disconnected_inf(self):
        dists = _make_disconnected_tree().distances_from(0)
        assert dists[0] == 0.0
        assert np.isfinite(dists[1])
        assert np.isfinite(dists[2])
        assert np.isinf(dists[3])
        assert np.isinf(dists[4])

    def test_star_distances(self):
        dists = _make_star_tree(4, weight=2.0).distances_from(0)
        np.testing.assert_allclose(dists, [0.0, 2.0, 2.0, 2.0])

    def test_invalid_source(self):
        tree = _make_chain_tree(3)
        with pytest.raises(ValueError, match="source"):
            tree.distances_from(-1)
