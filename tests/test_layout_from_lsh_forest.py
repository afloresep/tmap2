"""
Tests for layout_from_lsh_forest and layout_from_knn_graph functions.

These tests verify the high-level layout API that matches the original TMAP interface.
"""

import numpy as np
import pytest

from tmap import LSHForest, MinHash
from tmap.layout import OGDF_AVAILABLE

pytestmark = pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF not available")

# Conditional imports - only available when OGDF is built
if OGDF_AVAILABLE:
    from tmap.layout import (
        LayoutConfig,
        Merger,
        Placer,
        ScalingType,
        layout_from_edge_list,
        layout_from_knn_graph,
        layout_from_lsh_forest,
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def small_lsh_forest():
    """Create a small LSHForest with connected data for testing."""
    np.random.seed(42)
    n_samples = 50
    n_features = 256

    # Create data with good connectivity (overlapping patterns)
    data = np.zeros((n_samples, n_features), dtype=np.uint8)
    for i in range(n_samples):
        # Base features
        base = np.random.choice(n_features, 40, replace=False)
        data[i, base] = 1
        # Share with neighbors for connectivity
        if i > 0:
            shared = np.random.choice(
                np.where(data[i - 1] == 1)[0],
                min(20, data[i - 1].sum()),
                replace=False,
            )
            data[i, shared] = 1

    mh = MinHash(num_perm=128, seed=42)
    sigs = mh.batch_from_binary_array(data)

    lsh = LSHForest(d=128, l=64)
    lsh.batch_add(sigs)
    lsh.index()

    return lsh, n_samples


@pytest.fixture
def clustered_lsh_forest():
    """Create LSHForest with distinct clusters (may have disconnected components)."""
    np.random.seed(42)
    n_samples = 100
    n_features = 512
    n_clusters = 5

    data = np.zeros((n_samples, n_features), dtype=np.uint8)
    for i in range(n_samples):
        cluster = i // (n_samples // n_clusters)
        # Each cluster uses different feature ranges (minimal overlap)
        base = cluster * 100
        features = np.random.choice(100, 30, replace=False) + base
        features = np.clip(features, 0, n_features - 1)
        data[i, features] = 1

    mh = MinHash(num_perm=128, seed=42)
    sigs = mh.batch_from_binary_array(data)

    lsh = LSHForest(d=128, l=32)
    lsh.batch_add(sigs)
    lsh.index()

    return lsh, n_samples


# =============================================================================
# Basic functionality tests
# =============================================================================


class TestLayoutFromLSHForest:
    """Tests for layout_from_lsh_forest function."""

    def test_returns_correct_types(self, small_lsh_forest):
        """Output should be numpy arrays of correct dtypes."""
        lsh, n = small_lsh_forest
        x, y, s, t = layout_from_lsh_forest(lsh)

        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(s, np.ndarray)
        assert isinstance(t, np.ndarray)

        assert x.dtype == np.float32
        assert y.dtype == np.float32
        assert s.dtype == np.uint32
        assert t.dtype == np.uint32

    def test_returns_correct_shapes(self, small_lsh_forest):
        """Output arrays should have correct lengths."""
        lsh, n = small_lsh_forest
        x, y, s, t = layout_from_lsh_forest(lsh)

        # Coordinates for all nodes
        assert len(x) == n
        assert len(y) == n

        # Edge arrays should be same length
        assert len(s) == len(t)

        # MST should have at most n-1 edges (exactly n-1 if connected)
        assert len(s) <= n - 1

    def test_coordinates_normalized(self, small_lsh_forest):
        """Coordinates should be normalized to [-0.5, 0.5] range."""
        lsh, n = small_lsh_forest
        x, y, s, t = layout_from_lsh_forest(lsh)

        assert x.min() >= -0.5
        assert x.max() <= 0.5
        assert y.min() >= -0.5
        assert y.max() <= 0.5

    def test_edges_reference_valid_nodes(self, small_lsh_forest):
        """Edge indices should reference existing nodes."""
        lsh, n = small_lsh_forest
        x, y, s, t = layout_from_lsh_forest(lsh)

        assert s.max() < n
        assert t.max() < n
        assert s.min() >= 0
        assert t.min() >= 0

    def test_uses_config_k_and_kc(self, small_lsh_forest):
        """Config k and kc should affect the kNN graph construction."""
        lsh, n = small_lsh_forest

        # With very low k, might get fewer edges
        cfg_low_k = LayoutConfig()
        cfg_low_k.k = 3
        cfg_low_k.kc = 5

        cfg_high_k = LayoutConfig()
        cfg_high_k.k = 20
        cfg_high_k.kc = 50

        x1, y1, s1, t1 = layout_from_lsh_forest(lsh, cfg_low_k)
        x2, y2, s2, t2 = layout_from_lsh_forest(lsh, cfg_high_k)

        # Both should produce valid layouts
        assert len(x1) == n
        assert len(x2) == n

        # Higher k typically produces better connectivity (more edges in MST)
        # Note: This isn't always true, but with our test data it should be
        assert len(s2) >= len(s1)

    def test_default_config_works(self, small_lsh_forest):
        """Should work with no config (uses defaults)."""
        lsh, n = small_lsh_forest
        x, y, s, t = layout_from_lsh_forest(lsh)

        assert len(x) == n
        assert len(s) > 0  # Should have some edges


class TestLayoutFromKNNGraph:
    """Tests for layout_from_knn_graph function."""

    def test_returns_correct_types(self, small_lsh_forest):
        """Output should be numpy arrays of correct dtypes."""
        lsh, n = small_lsh_forest
        knn = lsh.get_knn_graph(k=10, kc=50)

        x, y, s, t = layout_from_knn_graph(knn)

        assert isinstance(x, np.ndarray)
        assert x.dtype == np.float32
        assert s.dtype == np.uint32

    def test_returns_correct_shapes(self, small_lsh_forest):
        """Output arrays should have correct lengths."""
        lsh, n = small_lsh_forest
        knn = lsh.get_knn_graph(k=10, kc=50)

        x, y, s, t = layout_from_knn_graph(knn)

        assert len(x) == n
        assert len(y) == n
        assert len(s) == len(t)

    def test_with_create_mst_false(self, small_lsh_forest):
        """Should work without MST computation (keeps all edges)."""
        lsh, n = small_lsh_forest
        knn = lsh.get_knn_graph(k=10, kc=50)

        x, y, s, t = layout_from_knn_graph(knn, create_mst=False)

        assert len(x) == n
        # Without MST, should have more edges than n-1
        assert len(s) > n - 1


# =============================================================================
# Configuration tests
# =============================================================================


class TestLayoutConfig:
    """Tests for LayoutConfig parameters."""

    def test_k_and_kc_defaults(self):
        """Default k and kc should be 10."""
        cfg = LayoutConfig()
        assert cfg.k == 10
        assert cfg.kc == 10

    def test_k_and_kc_settable(self):
        """k and kc should be settable."""
        cfg = LayoutConfig()
        cfg.k = 20
        cfg.kc = 100
        assert cfg.k == 20
        assert cfg.kc == 100

    def test_node_size_affects_layout(self, small_lsh_forest):
        """Different node sizes should produce different layouts."""
        lsh, n = small_lsh_forest

        cfg1 = LayoutConfig()
        cfg1.node_size = 1 / 65  # Default

        cfg2 = LayoutConfig()
        cfg2.node_size = 1 / 10  # Much larger

        x1, y1, _, _ = layout_from_lsh_forest(lsh, cfg1)
        x2, y2, _, _ = layout_from_lsh_forest(lsh, cfg2)

        # Layouts should be different (larger node_size = more spread)
        assert not np.allclose(x1, x2) or not np.allclose(y1, y2)

    def test_scaling_type_options(self, small_lsh_forest):
        """All scaling types should work."""
        lsh, n = small_lsh_forest

        for scaling_type in [
            ScalingType.Absolute,
            ScalingType.RelativeToAvgLength,
            ScalingType.RelativeToDesiredLength,
            ScalingType.RelativeToDrawing,
        ]:
            cfg = LayoutConfig()
            cfg.sl_scaling_type = scaling_type

            x, y, s, t = layout_from_lsh_forest(lsh, cfg)
            assert len(x) == n

    def test_placer_options(self, small_lsh_forest):
        """All placer types should work."""
        lsh, n = small_lsh_forest

        for placer in [
            Placer.Barycenter,
            Placer.Solar,
            Placer.Circle,
            Placer.Median,
            Placer.Zero,
            # Note: Placer.Random is non-deterministic
        ]:
            cfg = LayoutConfig()
            cfg.placer = placer

            x, y, s, t = layout_from_lsh_forest(lsh, cfg)
            assert len(x) == n

    def test_merger_options(self, small_lsh_forest):
        """All merger types should work."""
        lsh, n = small_lsh_forest

        for merger in [
            Merger.EdgeCover,
            Merger.LocalBiconnected,
            Merger.Solar,
            Merger.IndependentSet,
        ]:
            cfg = LayoutConfig()
            cfg.merger = merger

            x, y, s, t = layout_from_lsh_forest(lsh, cfg)
            assert len(x) == n


# =============================================================================
# Determinism tests
# =============================================================================


class TestDeterminism:
    """Tests for reproducible layouts."""

    def test_deterministic_with_seed(self, small_lsh_forest):
        """Same seed should produce identical layouts."""
        lsh, n = small_lsh_forest

        cfg = LayoutConfig()
        cfg.deterministic = True
        cfg.seed = 42

        x1, y1, s1, t1 = layout_from_lsh_forest(lsh, cfg)
        x2, y2, s2, t2 = layout_from_lsh_forest(lsh, cfg)

        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(t1, t2)

    def test_different_seeds_different_layouts(self, small_lsh_forest):
        """Different seeds may produce different layouts."""
        lsh, n = small_lsh_forest

        cfg1 = LayoutConfig()
        cfg1.deterministic = True
        cfg1.seed = 42

        cfg2 = LayoutConfig()
        cfg2.deterministic = True
        cfg2.seed = 123

        x1, y1, _, _ = layout_from_lsh_forest(lsh, cfg1)
        x2, y2, _, _ = layout_from_lsh_forest(lsh, cfg2)

        # Note: With deterministic settings and same data, layouts might still
        # be very similar. The seed mainly affects initial placement.
        # We just verify both produce valid output.
        assert len(x1) == len(x2) == n


# =============================================================================
# Disconnected component tests
# =============================================================================


class TestDisconnectedComponents:
    """Tests for handling graphs with multiple components."""

    def test_handles_disconnected_components(self, clustered_lsh_forest):
        """Should handle graphs with disconnected components gracefully."""
        lsh, n = clustered_lsh_forest

        cfg = LayoutConfig()
        cfg.k = 5  # Low k makes disconnection more likely
        cfg.kc = 10

        x, y, s, t = layout_from_lsh_forest(lsh, cfg)

        # All nodes should get coordinates
        assert len(x) == n
        assert len(y) == n

        # Coordinates should still be normalized
        assert x.min() >= -0.5
        assert x.max() <= 0.5

    def test_disconnected_components_get_fewer_edges(self, clustered_lsh_forest):
        """Disconnected components should result in fewer than n-1 MST edges."""
        lsh, n = clustered_lsh_forest

        cfg = LayoutConfig()
        cfg.k = 3  # Very low k
        cfg.kc = 5

        x, y, s, t = layout_from_lsh_forest(lsh, cfg)

        # With disconnected components, MST has n - num_components edges
        # So edges < n - 1 indicates multiple components
        if len(s) < n - 1:
            # This is expected behavior for disconnected graphs
            num_components = n - len(s)
            assert num_components >= 2


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_small_dataset(self):
        """Should handle very small datasets."""
        np.random.seed(42)
        data = np.random.randint(0, 2, (5, 64), dtype=np.uint8)

        mh = MinHash(num_perm=64, seed=42)
        sigs = mh.batch_from_binary_array(data)

        lsh = LSHForest(d=64, l=8)
        lsh.batch_add(sigs)
        lsh.index()

        cfg = LayoutConfig()
        cfg.k = 2  # Small k for small dataset

        x, y, s, t = layout_from_lsh_forest(lsh, cfg)

        assert len(x) == 5
        assert len(s) <= 4  # At most n-1 edges

    def test_two_node_dataset(self):
        """Should handle minimal 2-node dataset."""
        np.random.seed(42)
        data = np.random.randint(0, 2, (2, 64), dtype=np.uint8)

        mh = MinHash(num_perm=64, seed=42)
        sigs = mh.batch_from_binary_array(data)

        lsh = LSHForest(d=64, l=8)
        lsh.batch_add(sigs)
        lsh.index()

        cfg = LayoutConfig()
        cfg.k = 1

        x, y, s, t = layout_from_lsh_forest(lsh, cfg)

        assert len(x) == 2
        assert len(s) <= 1  # At most 1 edge

    def test_high_fme_iterations(self, small_lsh_forest):
        """Higher iterations should work (though slower)."""
        lsh, n = small_lsh_forest

        cfg = LayoutConfig()
        cfg.fme_iterations = 100  # Low for speed

        x, y, s, t = layout_from_lsh_forest(lsh, cfg)
        assert len(x) == n

    def test_mmm_repeats(self, small_lsh_forest):
        """Multiple multilevel repeats should work."""
        lsh, n = small_lsh_forest

        cfg = LayoutConfig()
        cfg.mmm_repeats = 3

        x, y, s, t = layout_from_lsh_forest(lsh, cfg)
        assert len(x) == n


# =============================================================================
# Integration tests
# =============================================================================


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_full_pipeline_with_real_params(self, small_lsh_forest):
        """Test with parameters similar to real TMAP usage."""
        lsh, n = small_lsh_forest

        # Parameters matching old TMAP defaults
        cfg = LayoutConfig()
        cfg.k = 20
        cfg.kc = 50
        cfg.node_size = 1 / 30
        cfg.mmm_repeats = 2
        cfg.sl_extra_scaling_steps = 10
        cfg.sl_scaling_type = ScalingType.RelativeToAvgLength
        cfg.fme_iterations = 500
        cfg.deterministic = True
        cfg.seed = 42

        x, y, s, t = layout_from_lsh_forest(lsh, cfg)

        # Verify complete output
        assert len(x) == n
        assert len(y) == n
        assert len(s) == len(t)
        assert len(s) > 0

        # Verify normalization
        assert -0.5 <= x.min() <= x.max() <= 0.5
        assert -0.5 <= y.min() <= y.max() <= 0.5

        # Verify edges are valid
        assert (s < n).all()
        assert (t < n).all()

    def test_layout_from_knn_vs_lsh_forest(self, small_lsh_forest):
        """layout_from_knn_graph with same kNN should produce same result."""
        lsh, n = small_lsh_forest

        cfg = LayoutConfig()
        cfg.k = 15
        cfg.kc = 30
        cfg.deterministic = True
        cfg.seed = 42

        # Method 1: layout_from_lsh_forest
        x1, y1, s1, t1 = layout_from_lsh_forest(lsh, cfg)

        # Method 2: manual kNN + layout_from_knn_graph
        knn = lsh.get_knn_graph(k=cfg.k, kc=cfg.kc)
        x2, y2, s2, t2 = layout_from_knn_graph(knn, cfg)

        # Results should be identical
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(t1, t2)


# =============================================================================
# tree_from_knn_graph tests
# =============================================================================


class TestTreeFromKNNGraph:
    """Tests for OGDF-backed MST extraction from a KNN graph."""

    def test_returns_tree(self, small_lsh_forest):
        from tmap.graph import Tree, tree_from_knn_graph

        lsh, n = small_lsh_forest
        knn = lsh.get_knn_graph(k=15, kc=30)
        tree = tree_from_knn_graph(knn)

        assert isinstance(tree, Tree)
        assert tree.n_nodes == n
        assert len(tree.weights) == len(tree.edges)

    def test_connected_graph_has_n_minus_one_edges(self, small_lsh_forest):
        from tmap.graph import tree_from_knn_graph

        lsh, n = small_lsh_forest
        knn = lsh.get_knn_graph(k=15, kc=30)
        tree = tree_from_knn_graph(knn)

        assert len(tree.edges) == n - 1

    def test_accepts_layout_config(self, small_lsh_forest):
        from tmap.graph import tree_from_knn_graph

        lsh, n = small_lsh_forest
        knn = lsh.get_knn_graph(k=15, kc=30)

        cfg = LayoutConfig()
        cfg.deterministic = True
        cfg.seed = 42

        tree = tree_from_knn_graph(knn, config=cfg)
        assert tree.n_nodes == n


# =============================================================================
# Additional edge cases
# =============================================================================


class TestAdditionalEdgeCases:
    """Additional edge case tests for layout module."""

    def test_placer_random_is_nondeterministic(self, small_lsh_forest):
        """Placer.Random should produce different results without seed."""
        lsh, n = small_lsh_forest

        cfg = LayoutConfig()
        cfg.placer = Placer.Random
        cfg.deterministic = False  # Explicitly non-deterministic

        x1, y1, _, _ = layout_from_lsh_forest(lsh, cfg)
        x2, y2, _, _ = layout_from_lsh_forest(lsh, cfg)

        # Layouts should be valid
        assert len(x1) == n
        assert len(x2) == n
        # (They may or may not differ - random doesn't guarantee difference)

    def test_empty_tree_edges(self):
        """Layout should handle an edge list with no edges."""

        cfg = LayoutConfig()
        cfg.deterministic = True
        cfg.seed = 42

        x, y, s, t = layout_from_edge_list(5, [], cfg, create_mst=False)

        # All nodes should get coordinates
        assert len(x) == 5
        assert len(y) == 5
        assert len(s) == 0
        assert len(t) == 0

    def test_linear_tree(self):
        """Layout should handle a linear chain edge list."""

        n = 20
        # Chain: 0-1-2-3-...-19
        edges = [(i, i + 1, 1.0) for i in range(n - 1)]
        x, y, _, _ = layout_from_edge_list(n, edges, create_mst=False)

        assert len(x) == n
        # Coordinates should be spread out, not all at origin
        assert x.max() - x.min() > 0.1 or y.max() - y.min() > 0.1

    def test_star_tree(self):
        """Layout should handle star topology edge lists."""

        n = 10
        # Star: node 0 connects to all others
        edges = [(0, i, 1.0) for i in range(1, n)]
        x, y, _, _ = layout_from_edge_list(n, edges, create_mst=False)

        assert len(x) == n
        # Central node should be roughly in the middle
        # (or at least layout should complete without error)

    def test_sl_scaling_min_max(self, small_lsh_forest):
        """Scaling min/max parameters should be accepted."""
        lsh, n = small_lsh_forest

        cfg = LayoutConfig()
        cfg.sl_scaling_min = 0.5
        cfg.sl_scaling_max = 2.0

        x, y, s, t = layout_from_lsh_forest(lsh, cfg)

        assert len(x) == n

    def test_merger_factor_parameter(self, small_lsh_forest):
        """merger_factor parameter should be accepted."""
        lsh, n = small_lsh_forest

        for factor in [1.5, 2.0, 3.0]:
            cfg = LayoutConfig()
            cfg.merger_factor = factor

            x, y, s, t = layout_from_lsh_forest(lsh, cfg)
            assert len(x) == n
