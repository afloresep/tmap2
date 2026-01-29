"""
End-to-end tests for the full TMAP2 pipeline.

Tests the complete flow:
    data -> MinHash -> LSHForest -> MSTBuilder -> ForceDirectedLayout -> TmapViz

Each stage is also tested independently to isolate failures.
"""

import numpy as np
import pytest

from tmap.layout import OGDF_AVAILABLE

# Skip all tests if OGDF not available
pytestmark = pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def similar_signatures():
    """Generate MinHash signatures with high similarity for reliable LSH.

    Creates 100 signatures in 10 clusters. Within each cluster, signatures
    share 95% of their values (very high similarity = reliable LSH collisions).
    With l=32, band size = 4, P(band match) = 0.95^4 ≈ 81%.
    """
    rng = np.random.default_rng(42)
    n_samples = 100
    d = 128
    n_clusters = 10
    samples_per_cluster = n_samples // n_clusters

    signatures = np.zeros((n_samples, d), dtype=np.uint64)

    for cluster_id in range(n_clusters):
        # Create cluster template
        template = rng.integers(0, 2**63, size=d, dtype=np.uint64)

        for i in range(samples_per_cluster):
            idx = cluster_id * samples_per_cluster + i
            # Start with template, modify only 5% of values (95% similarity)
            sig = template.copy()
            modify_mask = rng.random(d) < 0.05
            sig[modify_mask] = rng.integers(0, 2**63, size=np.sum(modify_mask), dtype=np.uint64)
            signatures[idx] = sig

    return signatures


@pytest.fixture
def small_similar_signatures():
    """Small dataset with similar signatures for quick tests (10 samples)."""
    rng = np.random.default_rng(123)
    d = 64

    # Create 2 clusters of 5 samples each with 95% similarity
    signatures = np.zeros((10, d), dtype=np.uint64)

    for cluster_id in range(2):
        template = rng.integers(0, 2**63, size=d, dtype=np.uint64)
        for i in range(5):
            idx = cluster_id * 5 + i
            sig = template.copy()
            modify_mask = rng.random(d) < 0.05  # 95% similarity
            sig[modify_mask] = rng.integers(0, 2**63, size=np.sum(modify_mask), dtype=np.uint64)
            signatures[idx] = sig

    return signatures


@pytest.fixture
def clustered_fingerprints():
    """Generate synthetic binary fingerprints with high similarity clusters.

    For testing the full pipeline including MinHash encoding.
    Creates 20 samples in 4 clusters with high intra-cluster similarity.
    """
    rng = np.random.default_rng(42)
    n_samples = 20
    n_bits = 512
    n_clusters = 4

    data = np.zeros((n_samples, n_bits), dtype=np.uint8)

    for cluster_id in range(n_clusters):
        # Create cluster template (~30% density for higher overlap)
        template = (rng.random(n_bits) < 0.3).astype(np.uint8)

        # 5 samples per cluster with only 10% bit flips (90% similarity)
        start_idx = cluster_id * 5
        for i in range(start_idx, start_idx + 5):
            sample = template.copy()
            flip_mask = rng.random(n_bits) < 0.1
            sample[flip_mask] = 1 - sample[flip_mask]
            data[i] = sample

    return data


# =============================================================================
# Stage 1: MinHash Encoding
# =============================================================================


class TestMinHashEncoding:
    """Test MinHash encoder produces valid signatures."""

    def test_minhash_output_shape(self, clustered_fingerprints):
        from tmap import MinHash

        mh = MinHash(num_perm=128, seed=42)
        signatures = mh.batch_from_binary_array(clustered_fingerprints)

        assert signatures.shape == (20, 128)
        assert signatures.dtype == np.uint64

    def test_minhash_deterministic(self, clustered_fingerprints):
        from tmap import MinHash

        mh1 = MinHash(num_perm=64, seed=42)
        mh2 = MinHash(num_perm=64, seed=42)

        sig1 = mh1.batch_from_binary_array(clustered_fingerprints)
        sig2 = mh2.batch_from_binary_array(clustered_fingerprints)

        np.testing.assert_array_equal(sig1, sig2)


# =============================================================================
# Stage 2: LSHForest k-NN Graph
# =============================================================================


class TestLSHForestKNN:
    """Test LSHForest produces valid k-NN graph."""

    def test_knn_graph_shape(self, similar_signatures):
        from tmap import LSHForest

        lsh = LSHForest(d=128, l=32)  # More bands = smaller band size = better recall
        lsh.batch_add(similar_signatures)
        lsh.index()

        knn = lsh.get_knn_graph(k=10, kc=20)

        assert knn.indices.shape == (100, 10)
        assert knn.distances.shape == (100, 10)
        assert knn.indices.dtype == np.int32
        assert knn.distances.dtype == np.float32

    def test_knn_distances_valid(self, similar_signatures):
        from tmap import LSHForest

        lsh = LSHForest(d=128, l=32)
        lsh.batch_add(similar_signatures)
        lsh.index()

        knn = lsh.get_knn_graph(k=10, kc=20)

        # Valid entries have indices >= 0 and distances < 2.0 (2.0 is invalid marker)
        valid_mask = (knn.indices >= 0) & (knn.distances < 2.0)
        valid_distances = knn.distances[valid_mask]

        # Should have found at least some neighbors
        assert len(valid_distances) > 0, "No valid neighbors found"

        # Valid distances should be in range [0, 1] (Jaccard distance)
        assert np.all(valid_distances >= 0)
        assert np.all(valid_distances <= 1)


# =============================================================================
# Stage 3: MST Building
# =============================================================================


class TestMSTBuilder:
    """Test MST construction from k-NN graph."""

    def test_mst_edge_count(self, similar_signatures):
        from tmap import LSHForest
        from tmap.graph.mst import MSTBuilder

        lsh = LSHForest(d=128, l=32)
        lsh.batch_add(similar_signatures)
        lsh.index()
        knn = lsh.get_knn_graph(k=10, kc=20)

        builder = MSTBuilder(bias_factor=0.1)
        tree = builder.build(knn)

        # With clustered data, MST may have disconnected components
        # Each cluster of 10 samples should have 9 edges (fully connected within)
        # So minimum expected edges = n_clusters * (cluster_size - 1) = 10 * 9 = 90
        assert tree.n_nodes == 100
        assert len(tree.edges) >= 90  # At least intra-cluster edges
        assert len(tree.edges) <= 99  # At most n-1 (fully connected)
        assert len(tree.weights) == len(tree.edges)

    def test_mst_weights_positive(self, similar_signatures):
        from tmap import LSHForest
        from tmap.graph.mst import MSTBuilder

        lsh = LSHForest(d=128, l=32)
        lsh.batch_add(similar_signatures)
        lsh.index()
        knn = lsh.get_knn_graph(k=10, kc=20)

        tree = MSTBuilder().build(knn)

        assert np.all(tree.weights >= 0)


# =============================================================================
# Stage 4: Force-Directed Layout
# =============================================================================


class TestForceDirectedLayout:
    """Test layout computation from tree."""

    def test_layout_coordinate_shape(self, similar_signatures):
        from tmap import LSHForest
        from tmap.graph.mst import MSTBuilder
        from tmap.layout import ForceDirectedLayout

        lsh = LSHForest(d=128, l=32)
        lsh.batch_add(similar_signatures)
        lsh.index()
        knn = lsh.get_knn_graph(k=10, kc=20)

        tree = MSTBuilder(bias_factor=0.1).build(knn)

        layout = ForceDirectedLayout(seed=42, max_iterations=100)
        coords = layout.compute(tree)

        assert len(coords.x) == 100
        assert len(coords.y) == 100
        assert coords.x.dtype == np.float32
        assert coords.y.dtype == np.float32

    def test_layout_deterministic(self, small_similar_signatures):
        from tmap import LSHForest
        from tmap.graph.mst import MSTBuilder
        from tmap.layout import ForceDirectedLayout

        lsh = LSHForest(d=64)
        lsh.batch_add(small_similar_signatures)
        lsh.index()
        knn = lsh.get_knn_graph(k=5, kc=10)

        tree = MSTBuilder().build(knn)

        layout1 = ForceDirectedLayout(seed=42, max_iterations=50)
        layout2 = ForceDirectedLayout(seed=42, max_iterations=50)

        coords1 = layout1.compute(tree)
        coords2 = layout2.compute(tree)

        np.testing.assert_array_almost_equal(coords1.x, coords2.x)
        np.testing.assert_array_almost_equal(coords1.y, coords2.y)


# =============================================================================
# Stage 5: Visualization
# =============================================================================


class TestTmapViz:
    """Test visualization generation."""

    def test_render_produces_html(self, similar_signatures):
        from tmap import LSHForest
        from tmap.graph.mst import MSTBuilder
        from tmap.layout import ForceDirectedLayout
        from tmap.visualization.tmapviz import TmapViz

        lsh = LSHForest(d=128, l=32)
        lsh.batch_add(similar_signatures)
        lsh.index()
        knn = lsh.get_knn_graph(k=10, kc=20)

        tree = MSTBuilder(bias_factor=0.1).build(knn)

        layout = ForceDirectedLayout(seed=42, max_iterations=100)
        coords = layout.compute(tree)

        # Create visualization
        viz = TmapViz()
        viz.set_points(coords.x, coords.y)

        # Add some metadata
        labels = [f"Point_{i}" for i in range(100)]
        values = np.arange(100, dtype=float)

        viz.add_label("id", labels)
        viz.add_color_layout("index", values.tolist(), categorical=False, color="viridis")

        html = viz.render()

        # Validate HTML output
        assert isinstance(html, str)
        assert len(html) > 1000  # Should be substantial
        assert "<!DOCTYPE html>" in html or "<html" in html
        assert "regl" in html.lower() or "scatterplot" in html.lower()

    def test_viz_with_categorical_colors(self, small_similar_signatures):
        from tmap import LSHForest
        from tmap.graph.mst import MSTBuilder
        from tmap.layout import ForceDirectedLayout
        from tmap.visualization.tmapviz import TmapViz

        lsh = LSHForest(d=64)
        lsh.batch_add(small_similar_signatures)
        lsh.index()
        knn = lsh.get_knn_graph(k=5, kc=10)

        tree = MSTBuilder().build(knn)
        layout = ForceDirectedLayout(seed=42, max_iterations=50)
        coords = layout.compute(tree)

        viz = TmapViz()
        viz.set_points(coords.x, coords.y)

        # Categorical color layout
        categories = ["A", "B", "C"] * 3 + ["A"]  # 10 items
        viz.add_color_layout("category", categories, categorical=True, color="tab10")

        html = viz.render()
        assert "<html" in html


# =============================================================================
# Full Pipeline Integration Test
# =============================================================================


class TestFullPipeline:
    """Complete end-to-end integration tests."""

    def test_full_pipeline_100_points(self, similar_signatures):
        """Test complete pipeline with 100 points (using similar signatures)."""
        from tmap import LSHForest
        from tmap.graph.mst import MSTBuilder
        from tmap.layout import ForceDirectedLayout
        from tmap.visualization.tmapviz import TmapViz

        # Stage 1: Use pre-generated similar signatures
        assert similar_signatures.shape[0] == 100

        # Stage 2: Build k-NN
        lsh = LSHForest(d=128, l=32)
        lsh.batch_add(similar_signatures)
        lsh.index()
        knn = lsh.get_knn_graph(k=10, kc=20)
        assert knn.indices.shape[0] == 100

        # Stage 3: Build MST
        # With clustered data, may have disconnected components
        tree = MSTBuilder(bias_factor=0.1).build(knn)
        assert tree.n_nodes == 100
        assert len(tree.edges) >= 90  # At least intra-cluster connections

        # Stage 4: Compute layout
        layout = ForceDirectedLayout(seed=42, max_iterations=100)
        coords = layout.compute(tree)
        assert len(coords.x) == 100

        # Stage 5: Visualize
        viz = TmapViz()
        viz.title = "End-to-End Test"
        viz.set_points(coords.x, coords.y)

        # Add multiple metadata columns
        viz.add_label("index", [str(i) for i in range(100)])
        viz.add_color_layout("value", list(range(100)), categorical=False, color="plasma")

        html = viz.render()

        # Final validation
        assert len(html) > 5000
        assert "End-to-End Test" in html

    def test_full_pipeline_with_minhash(self, clustered_fingerprints):
        """Test full pipeline including MinHash encoding."""
        from tmap import MinHash, LSHForest
        from tmap.graph.mst import MSTBuilder
        from tmap.layout import ForceDirectedLayout
        from tmap.visualization.tmapviz import TmapViz

        # Stage 1: MinHash encoding
        mh = MinHash(num_perm=128, seed=42)
        signatures = mh.batch_from_binary_array(clustered_fingerprints)
        assert signatures.shape == (20, 128)

        # Stage 2: Build k-NN (with high kc for better recall on small dataset)
        lsh = LSHForest(d=128, l=16)
        lsh.batch_add(signatures)
        lsh.index()
        knn = lsh.get_knn_graph(k=10, kc=50)

        # Verify we found some valid neighbors
        valid_neighbors = np.sum(knn.indices >= 0)
        assert valid_neighbors > 0, "No valid neighbors found in k-NN"

        # Stage 3: Build MST
        tree = MSTBuilder(bias_factor=0.1).build(knn)
        assert tree.n_nodes == 20

        # Stage 4: Compute layout
        layout = ForceDirectedLayout(seed=42, max_iterations=50)
        coords = layout.compute(tree)
        assert len(coords.x) == 20

        # Stage 5: Visualize
        viz = TmapViz()
        viz.set_points(coords.x, coords.y)
        html = viz.render()
        assert "<html" in html

    def test_pipeline_deterministic(self):
        """Verify pipeline produces identical results with same signatures."""
        from tmap import LSHForest
        from tmap.graph.mst import MSTBuilder
        from tmap.layout import ForceDirectedLayout

        def create_similar_sigs(seed):
            """Create signatures with 95% similarity within clusters."""
            rng = np.random.default_rng(seed)
            d = 64
            sigs = np.zeros((20, d), dtype=np.uint64)
            for cluster_id in range(4):
                template = rng.integers(0, 2**63, size=d, dtype=np.uint64)
                for i in range(5):
                    idx = cluster_id * 5 + i
                    sig = template.copy()
                    modify_mask = rng.random(d) < 0.05
                    sig[modify_mask] = rng.integers(0, 2**63, size=np.sum(modify_mask), dtype=np.uint64)
                    sigs[idx] = sig
            return sigs

        def run_pipeline(seed):
            signatures = create_similar_sigs(seed)

            lsh = LSHForest(d=64, l=16)
            lsh.batch_add(signatures)
            lsh.index()
            knn = lsh.get_knn_graph(k=10, kc=20)

            tree = MSTBuilder().build(knn)

            layout = ForceDirectedLayout(seed=seed, max_iterations=50)
            coords = layout.compute(tree)

            return coords.x, coords.y

        x1, y1 = run_pipeline(seed=999)
        x2, y2 = run_pipeline(seed=999)

        np.testing.assert_array_almost_equal(x1, x2)
        np.testing.assert_array_almost_equal(y1, y2)

    def test_pipeline_with_layout_config(self, small_similar_signatures):
        """Test pipeline with custom layout configuration."""
        from tmap import LSHForest
        from tmap.graph.mst import MSTBuilder
        from tmap.layout import LayoutConfig, layout_from_edge_list

        lsh = LSHForest(d=64, l=16)
        lsh.batch_add(small_similar_signatures)
        lsh.index()
        knn = lsh.get_knn_graph(k=5, kc=20)

        tree = MSTBuilder().build(knn)

        # Convert tree to edge list format
        edges = [
            (int(tree.edges[i, 0]), int(tree.edges[i, 1]), float(tree.weights[i]))
            for i in range(len(tree.edges))
        ]

        # Use layout_from_edge_list with custom config
        config = LayoutConfig()
        config.fme_iterations = 50
        config.deterministic = True
        config.seed = 42

        x, y, s, t = layout_from_edge_list(tree.n_nodes, edges, config, create_mst=False)

        assert len(x) == 10
        assert len(y) == 10
        # With 2 clusters of 5, may have 8 edges (2 * 4) if clusters disconnected
        assert len(s) >= 8  # At least intra-cluster edges


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimal_dataset_2_points(self):
        """Test with minimum viable dataset (2 points)."""
        from tmap import LSHForest
        from tmap.graph.mst import MSTBuilder
        from tmap.layout import ForceDirectedLayout

        # Create 2 nearly identical signatures (95% same)
        rng = np.random.default_rng(42)
        d = 32
        template = rng.integers(0, 2**63, size=d, dtype=np.uint64)
        sig1 = template.copy()
        sig2 = template.copy()
        # Modify just 1-2 values for sig2
        sig2[0] = rng.integers(0, 2**63, dtype=np.uint64)

        signatures = np.stack([sig1, sig2])

        lsh = LSHForest(d=32, l=16)  # More bands for better recall
        lsh.batch_add(signatures)
        lsh.index()
        knn = lsh.get_knn_graph(k=1, kc=10)

        tree = MSTBuilder().build(knn)
        assert tree.n_nodes == 2
        assert len(tree.edges) == 1

        layout = ForceDirectedLayout(seed=42, max_iterations=10)
        coords = layout.compute(tree)

        assert len(coords.x) == 2
        assert len(coords.y) == 2

    def test_high_k_value(self, small_similar_signatures):
        """Test with k approaching n (should still work)."""
        from tmap import LSHForest
        from tmap.graph.mst import MSTBuilder
        from tmap.layout import ForceDirectedLayout

        lsh = LSHForest(d=64, l=16)
        lsh.batch_add(small_similar_signatures)
        lsh.index()

        # k=9 is max for 10 points (can't include self)
        knn = lsh.get_knn_graph(k=9, kc=20)

        tree = MSTBuilder().build(knn)
        layout = ForceDirectedLayout(seed=42, max_iterations=50)
        coords = layout.compute(tree)

        assert len(coords.x) == 10
