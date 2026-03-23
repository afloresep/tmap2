"""Tests for tmap.graph.analysis module."""

from __future__ import annotations

import numpy as np
import pytest

from tmap.graph.analysis import node_diversity
from tmap.graph.types import Tree


def _make_tree(n_nodes, edges, weights):
    """Build a Tree from edge list and weights."""
    return Tree(
        n_nodes=n_nodes,
        edges=np.array(edges, dtype=np.int32),
        weights=np.array(weights, dtype=np.float32),
    )


class TestNodeDiversity:
    """Tests for node_diversity()."""

    def test_simple_chain(self):
        # 0 --1.0-- 1 --3.0-- 2 --5.0-- 3
        tree = _make_tree(4, [[0, 1], [1, 2], [2, 3]], [1.0, 3.0, 5.0])

        div = node_diversity(tree, method="mean")
        # Node 0 (leaf): [1.0] → mean=1.0
        assert abs(div[0] - 1.0) < 1e-6
        # Node 1 (internal): [1.0, 3.0] → mean=2.0
        assert abs(div[1] - 2.0) < 1e-6
        # Node 2 (internal): [3.0, 5.0] → mean=4.0
        assert abs(div[2] - 4.0) < 1e-6
        # Node 3 (leaf): [5.0] → mean=5.0
        assert abs(div[3] - 5.0) < 1e-6

    def test_method_max(self):
        # Star: 0 is center, connected to 1,2,3
        tree = _make_tree(4, [[0, 1], [0, 2], [0, 3]], [1.0, 4.0, 2.0])
        div = node_diversity(tree, method="max")
        # Node 0: max(1,4,2) = 4
        assert abs(div[0] - 4.0) < 1e-6
        # Leaves: single edge weight
        assert abs(div[1] - 1.0) < 1e-6
        assert abs(div[2] - 4.0) < 1e-6
        assert abs(div[3] - 2.0) < 1e-6

    def test_method_median(self):
        # Star: 0 connected to 1,2,3 with weights 1,4,2
        tree = _make_tree(4, [[0, 1], [0, 2], [0, 3]], [1.0, 4.0, 2.0])
        div = node_diversity(tree, method="median")
        # Node 0: median(1,4,2) = 2
        assert abs(div[0] - 2.0) < 1e-6

    def test_leaf_single_edge(self):
        # Single edge: 0 --7.0-- 1
        tree = _make_tree(2, [[0, 1]], [7.0])
        div = node_diversity(tree, method="mean")
        assert abs(div[0] - 7.0) < 1e-6
        assert abs(div[1] - 7.0) < 1e-6

    def test_returns_float64(self):
        tree = _make_tree(3, [[0, 1], [1, 2]], [1.0, 2.0])
        div = node_diversity(tree)
        assert div.dtype == np.float64
        assert div.shape == (3,)

    def test_invalid_method_raises(self):
        tree = _make_tree(2, [[0, 1]], [1.0])
        with pytest.raises(ValueError, match="method"):
            node_diversity(tree, method="invalid")

    def test_export(self):
        from tmap.graph import node_diversity as nd

        assert callable(nd)
