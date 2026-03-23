"""
End-to-end tests for the full TMAP2 pipeline.

Tests the complete flow:
    data -> MinHash -> LSHForest -> KNNGraph -> OGDF MST/Layout -> TmapViz

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
# Stage 3: Tree extraction from k-NN
# =============================================================================


def _make_layout_config(seed: int, iterations: int):
    from tmap.layout import LayoutConfig

    config = LayoutConfig()
    config.deterministic = True
    config.seed = seed
    config.fme_iterations = iterations
    return config


class TestTreeFromKNNGraph:
    """Test MST extraction from a k-NN graph."""

    def test_mst_edge_count(self, similar_signatures):
        from tmap import LSHForest
        from tmap.graph import tree_from_knn_graph

        lsh = LSHForest(d=128, l=32)
        lsh.batch_add(similar_signatures)
        lsh.index()
        knn = lsh.get_knn_graph(k=10, kc=20)

        tree = tree_from_knn_graph(knn)

        assert tree.n_nodes == 100
        assert 90 <= len(tree.edges) <= 99
        assert len(tree.weights) == len(tree.edges)

    def test_mst_weights_positive(self, similar_signatures):
        from tmap import LSHForest
        from tmap.graph import tree_from_knn_graph

        lsh = LSHForest(d=128, l=32)
        lsh.batch_add(similar_signatures)
        lsh.index()
        knn = lsh.get_knn_graph(k=10, kc=20)

        tree = tree_from_knn_graph(knn)

        assert np.all(tree.weights >= 0)


# =============================================================================
# Stage 4: OGDF layout from k-NN
# =============================================================================


class TestOGDFLayout:
    """Test layout computation directly from a k-NN graph."""

    def test_layout_coordinate_shape(self, similar_signatures):
        from tmap import LSHForest
        from tmap.layout import layout_from_knn_graph

        lsh = LSHForest(d=128, l=32)
        lsh.batch_add(similar_signatures)
        lsh.index()
        knn = lsh.get_knn_graph(k=10, kc=20)

        config = _make_layout_config(seed=42, iterations=100)
        x, y, s, t = layout_from_knn_graph(knn, config=config, create_mst=True)

        assert len(x) == 100
        assert len(y) == 100
        assert x.dtype == np.float32
        assert y.dtype == np.float32
        assert s.dtype == np.uint32
        assert t.dtype == np.uint32

    def test_layout_deterministic(self, small_similar_signatures):
        from tmap import LSHForest
        from tmap.layout import layout_from_knn_graph

        lsh = LSHForest(d=64)
        lsh.batch_add(small_similar_signatures)
        lsh.index()
        knn = lsh.get_knn_graph(k=5, kc=10)

        config = _make_layout_config(seed=42, iterations=50)
        x1, y1, s1, t1 = layout_from_knn_graph(knn, config=config, create_mst=True)
        x2, y2, s2, t2 = layout_from_knn_graph(knn, config=config, create_mst=True)

        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_almost_equal(x1, x2)
        np.testing.assert_array_almost_equal(y1, y2)


# =============================================================================
# Stage 5: Visualization
# =============================================================================


class TestTmapViz:
    """Test visualization generation."""

    def test_render_produces_html(self, similar_signatures):
        from tmap import LSHForest
        from tmap.layout import layout_from_knn_graph
        from tmap.visualization.tmapviz import TmapViz

        lsh = LSHForest(d=128, l=32)
        lsh.batch_add(similar_signatures)
        lsh.index()
        knn = lsh.get_knn_graph(k=10, kc=20)

        x, y, s, t = layout_from_knn_graph(
            knn,
            config=_make_layout_config(seed=42, iterations=100),
            create_mst=True,
        )

        viz = TmapViz()
        viz.set_points(x, y)
        viz.set_edges(s, t)
        viz.add_label("id", [f"Point_{i}" for i in range(100)])
        viz.add_color_layout("index", np.arange(100, dtype=float).tolist(), color="viridis")

        html = viz.to_html()

        assert isinstance(html, str)
        assert len(html) > 1000
        assert "<!DOCTYPE html>" in html or "<html" in html
        assert "regl" in html.lower() or "scatterplot" in html.lower()

    def test_viz_with_categorical_colors(self, small_similar_signatures):
        from tmap import LSHForest
        from tmap.layout import layout_from_knn_graph
        from tmap.visualization.tmapviz import TmapViz

        lsh = LSHForest(d=64)
        lsh.batch_add(small_similar_signatures)
        lsh.index()
        knn = lsh.get_knn_graph(k=5, kc=10)

        x, y, s, t = layout_from_knn_graph(
            knn,
            config=_make_layout_config(seed=42, iterations=50),
            create_mst=True,
        )

        viz = TmapViz()
        viz.set_points(x, y)
        viz.set_edges(s, t)
        viz.add_color_layout("category", ["A", "B", "C"] * 3 + ["A"], categorical=True)

        html = viz.to_html()
        assert "<html" in html


# =============================================================================
# Full Pipeline Integration Test
# =============================================================================


class TestFullPipeline:
    """Complete end-to-end integration tests."""

    def test_full_pipeline_100_points(self, similar_signatures):
        """Test complete pipeline with 100 points."""
        from tmap import LSHForest
        from tmap.graph import tree_from_knn_graph
        from tmap.layout import layout_from_knn_graph
        from tmap.visualization.tmapviz import TmapViz

        lsh = LSHForest(d=128, l=32)
        lsh.batch_add(similar_signatures)
        lsh.index()
        knn = lsh.get_knn_graph(k=10, kc=20)

        tree = tree_from_knn_graph(knn)
        assert tree.n_nodes == 100
        assert len(tree.edges) >= 90

        x, y, s, t = layout_from_knn_graph(
            knn,
            config=_make_layout_config(seed=42, iterations=100),
            create_mst=True,
        )
        assert len(x) == 100

        viz = TmapViz()
        viz.title = "End-to-End Test"
        viz.set_points(x, y)
        viz.set_edges(s, t)
        viz.add_label("index", [str(i) for i in range(100)])
        viz.add_color_layout("value", list(range(100)), categorical=False, color="plasma")

        html = viz.to_html()

        assert len(html) > 5000
        assert "End-to-End Test" in html

    def test_full_pipeline_with_minhash(self, clustered_fingerprints):
        """Test full pipeline including MinHash encoding."""
        from tmap import LSHForest, MinHash
        from tmap.graph import tree_from_knn_graph
        from tmap.layout import layout_from_knn_graph
        from tmap.visualization.tmapviz import TmapViz

        mh = MinHash(num_perm=128, seed=42)
        signatures = mh.batch_from_binary_array(clustered_fingerprints)
        assert signatures.shape == (20, 128)

        lsh = LSHForest(d=128, l=16)
        lsh.batch_add(signatures)
        lsh.index()
        knn = lsh.get_knn_graph(k=10, kc=50)

        valid_neighbors = np.sum(knn.indices >= 0)
        assert valid_neighbors > 0, "No valid neighbors found in k-NN"

        tree = tree_from_knn_graph(knn)
        assert tree.n_nodes == 20

        x, y, _, _ = layout_from_knn_graph(
            knn,
            config=_make_layout_config(seed=42, iterations=50),
            create_mst=True,
        )
        assert len(x) == 20

        viz = TmapViz()
        viz.set_points(x, y)
        html = viz.to_html()
        assert "<html" in html

    def test_pipeline_deterministic(self):
        """Verify pipeline produces identical results with same signatures."""
        from tmap import LSHForest
        from tmap.layout import layout_from_knn_graph

        def create_similar_sigs(seed: int) -> np.ndarray:
            rng = np.random.default_rng(seed)
            d = 64
            signatures = np.zeros((20, d), dtype=np.uint64)
            for cluster_id in range(4):
                template = rng.integers(0, 2**63, size=d, dtype=np.uint64)
                for i in range(5):
                    idx = cluster_id * 5 + i
                    sig = template.copy()
                    modify_mask = rng.random(d) < 0.05
                    sig[modify_mask] = rng.integers(
                        0, 2**63, size=np.sum(modify_mask), dtype=np.uint64
                    )
                    signatures[idx] = sig
            return signatures

        def run_pipeline(seed: int) -> tuple[np.ndarray, np.ndarray]:
            signatures = create_similar_sigs(seed)

            lsh = LSHForest(d=64, l=16)
            lsh.batch_add(signatures)
            lsh.index()
            knn = lsh.get_knn_graph(k=10, kc=20)

            x, y, _, _ = layout_from_knn_graph(
                knn,
                config=_make_layout_config(seed=seed, iterations=50),
                create_mst=True,
            )
            return x, y

        x1, y1 = run_pipeline(seed=999)
        x2, y2 = run_pipeline(seed=999)

        np.testing.assert_array_almost_equal(x1, x2)
        np.testing.assert_array_almost_equal(y1, y2)

    def test_pipeline_with_layout_config(self, small_similar_signatures):
        """Test pipeline with custom layout configuration."""
        from tmap import LSHForest
        from tmap.graph import tree_from_knn_graph
        from tmap.layout import layout_from_knn_graph

        lsh = LSHForest(d=64, l=16)
        lsh.batch_add(small_similar_signatures)
        lsh.index()
        knn = lsh.get_knn_graph(k=5, kc=20)

        config = _make_layout_config(seed=42, iterations=50)
        tree = tree_from_knn_graph(knn, config=config)
        x, y, s, t = layout_from_knn_graph(knn, config=config, create_mst=True)

        assert len(x) == 10
        assert len(y) == 10
        assert len(s) == len(t) == len(tree.edges)
        assert len(s) >= 8


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimal_dataset_2_points(self):
        """Test with minimum viable dataset."""
        from tmap import LSHForest
        from tmap.graph import tree_from_knn_graph
        from tmap.layout import layout_from_knn_graph

        rng = np.random.default_rng(42)
        d = 32
        template = rng.integers(0, 2**63, size=d, dtype=np.uint64)
        sig1 = template.copy()
        sig2 = template.copy()
        sig2[0] = rng.integers(0, 2**63, dtype=np.uint64)

        signatures = np.stack([sig1, sig2])

        lsh = LSHForest(d=32, l=16)
        lsh.batch_add(signatures)
        lsh.index()
        knn = lsh.get_knn_graph(k=1, kc=10)

        tree = tree_from_knn_graph(knn)
        assert tree.n_nodes == 2
        assert len(tree.edges) == 1

        x, y, _, _ = layout_from_knn_graph(
            knn,
            config=_make_layout_config(seed=42, iterations=10),
            create_mst=True,
        )
        assert len(x) == 2
        assert len(y) == 2

    def test_high_k_value(self, small_similar_signatures):
        """Test with k approaching n."""
        from tmap import LSHForest
        from tmap.layout import layout_from_knn_graph

        lsh = LSHForest(d=64, l=16)
        lsh.batch_add(small_similar_signatures)
        lsh.index()
        knn = lsh.get_knn_graph(k=9, kc=20)

        x, y, _, _ = layout_from_knn_graph(
            knn,
            config=_make_layout_config(seed=42, iterations=50),
            create_mst=True,
        )

        assert len(x) == 10
