"""Tests for USearchIndex and estimator dense-metric integration."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from tmap.index.types import KNNGraph
from tmap.index.usearch_index import USearchIndex
from tmap.layout import OGDF_AVAILABLE

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _clustered_dense_data(
    n_samples: int = 60,
    n_features: int = 32,
    n_clusters: int = 3,
    seed: int = 42,
) -> np.ndarray:
    """Generate clustered dense float data for testing."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, n_features)).astype(np.float32)
    data = np.empty((n_samples, n_features), dtype=np.float32)
    for i in range(n_samples):
        c = i % n_clusters
        data[i] = centers[c] + rng.standard_normal(n_features).astype(np.float32) * 0.1
    return data


K = 5


# ---------------------------------------------------------------------------
# USearchIndex tests
# ---------------------------------------------------------------------------


class TestUSearchIndex:
    def _make_index(self, **kwargs):
        kwargs.setdefault("seed", 42)
        return USearchIndex(**kwargs)

    @pytest.mark.parametrize("metric", ["cosine", "euclidean"])
    def test_knn_shape_and_dtype(self, metric: str) -> None:
        data = _clustered_dense_data()
        index = self._make_index()
        index.build_from_vectors(data, metric=metric)
        knn = index.query_knn(k=K)

        assert isinstance(knn, KNNGraph)
        assert knn.indices.shape == (data.shape[0], K)
        assert knn.distances.shape == (data.shape[0], K)
        assert knn.indices.dtype == np.int32
        assert knn.distances.dtype == np.float32

    @pytest.mark.parametrize("metric", ["cosine", "euclidean"])
    def test_knn_contract(self, metric: str) -> None:
        data = _clustered_dense_data()
        index = self._make_index()
        index.build_from_vectors(data, metric=metric)
        knn = index.query_knn(k=K)

        assert np.all(knn.indices >= 0)
        assert np.all(knn.indices < data.shape[0])
        assert np.all(knn.distances >= -1e-6)
        for i in range(data.shape[0]):
            assert i not in knn.indices[i]

    def test_query_single(self) -> None:
        data = _clustered_dense_data()
        index = self._make_index()
        index.build_from_vectors(data, metric="euclidean")

        indices, distances = index.query_point(data[0], k=K)
        assert indices.shape == (K,)
        assert distances.shape == (K,)
        assert indices.dtype == np.int32
        assert distances.dtype == np.float32

    @pytest.mark.parametrize("metric", ["cosine", "euclidean"])
    def test_query_batch(self, metric: str) -> None:
        data = _clustered_dense_data()
        index = self._make_index()
        index.build_from_vectors(data, metric=metric)

        query = data[:5]
        indices, distances = index.query_batch(query, k=K)

        assert indices.shape == (5, K)
        assert distances.shape == (5, K)
        assert indices.dtype == np.int32
        assert distances.dtype == np.float32
        assert np.all(indices >= 0)
        assert np.all(indices < data.shape[0])
        assert np.all(distances >= -1e-6)

    def test_unsupported_metric_raises(self) -> None:
        index = USearchIndex()
        with pytest.raises(ValueError, match="does not support metric"):
            index.build_from_vectors(_clustered_dense_data(), metric="jaccard")

    def test_auto_mode_picks_exact_for_small(self) -> None:
        index = self._make_index(mode="auto")
        data = _clustered_dense_data(n_samples=100)
        index.build_from_vectors(data, metric="euclidean")
        assert index.effective_mode == "exact"

    def test_exact_mode_knn_contract(self) -> None:
        """Exact mode: shape, dtype, valid indices, no self-match."""
        index = self._make_index(mode="exact")
        data = _clustered_dense_data(n_samples=60)
        index.build_from_vectors(data, metric="cosine")
        knn = index.query_knn(k=K)

        assert knn.indices.shape == (60, K)
        assert np.all(knn.indices >= 0)
        assert np.all(knn.distances >= -1e-6)
        for i in range(60):
            assert i not in knn.indices[i]

    def test_hnsw_cosine_distances_nonneg(self) -> None:
        index = self._make_index(mode="hnsw")
        data = _clustered_dense_data(n_samples=500, n_features=32)
        index.build_from_vectors(data, metric="cosine")

        knn = index.query_knn(k=K)
        assert np.all(knn.distances >= -1e-6), (
            f"Negative distances found: min={knn.distances.min()}"
        )

    def test_hnsw_knn_contract(self) -> None:
        """HNSW: shape, dtype, valid indices, no self-match."""
        index = self._make_index(mode="hnsw")
        data = _clustered_dense_data(n_samples=1000, n_features=32)
        index.build_from_vectors(data, metric="euclidean")
        knn = index.query_knn(k=K)

        assert knn.indices.dtype == np.int32
        assert knn.distances.dtype == np.float32
        assert knn.indices.shape == (1000, K)
        assert np.all(knn.indices >= 0)
        assert np.all(knn.indices < 1000)
        for i in range(100):
            assert i not in knn.indices[i]

    def test_metric_property(self) -> None:
        data = _clustered_dense_data()
        index = self._make_index()
        assert index.metric is None
        index.build_from_vectors(data, metric="euclidean")
        assert index.metric == "euclidean"

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="mode must be"):
            USearchIndex(mode="bad")

    def test_euclidean_sqrt_correctness(self) -> None:
        """Verify that euclidean distances are actual L2 (not squared)."""
        rng = np.random.default_rng(99)
        data = rng.standard_normal((20, 4)).astype(np.float32)
        index = self._make_index(mode="exact")
        index.build_from_vectors(data, metric="euclidean")
        knn = index.query_knn(k=3)

        # Verify against manual L2
        for i in range(20):
            for j_pos in range(3):
                nb = knn.indices[i, j_pos]
                expected = np.sqrt(np.sum((data[i] - data[nb]) ** 2))
                np.testing.assert_allclose(knn.distances[i, j_pos], expected, rtol=1e-4)

    def test_duplicate_vectors_self_exclusion(self) -> None:
        """When vectors are duplicated, self-exclusion still works."""
        rng = np.random.default_rng(42)
        base = rng.standard_normal((10, 8)).astype(np.float32)
        # Duplicate each vector so self might not be first match
        data = np.repeat(base, 2, axis=0)  # shape (20, 8)
        index = self._make_index(mode="exact")
        index.build_from_vectors(data, metric="euclidean")
        knn = index.query_knn(k=3)

        for i in range(20):
            assert i not in knn.indices[i]

    def test_pickle_roundtrip(self) -> None:
        """USearchIndex survives pickle dump/load (for TMAP.save)."""
        data = _clustered_dense_data(n_samples=40)
        index = self._make_index(mode="hnsw")
        index.build_from_vectors(data, metric="cosine")

        buf = pickle.dumps(index)
        restored = pickle.loads(buf)

        # Restored index can query new points
        idx_r, dist_r = restored.query_point(data[0], k=K)
        assert idx_r.shape == (K,)
        assert np.all(dist_r >= -1e-6)

    def test_exact_save_load_roundtrip(self, tmp_path) -> None:
        """Exact mode save/load persists raw vectors and remains queryable."""
        data = _clustered_dense_data(n_samples=40)
        index = self._make_index(mode="exact")
        index.build_from_vectors(data, metric="euclidean")

        path = tmp_path / "exact.usearch"
        index.save(path)
        restored = USearchIndex.load(path)

        knn = restored.query_knn(k=K)
        assert knn.indices.shape == (40, K)
        idx_r, dist_r = restored.query_point(data[0], k=K)
        assert idx_r.shape == (K,)
        assert np.all(dist_r >= -1e-6)

    def test_hnsw_save_load_roundtrip(self, tmp_path) -> None:
        """HNSW save/load restores a queryable ANN index."""
        data = _clustered_dense_data(n_samples=200, n_features=16)
        index = self._make_index(mode="hnsw")
        index.build_from_vectors(data, metric="cosine")

        path = tmp_path / "hnsw.usearch"
        index.save(path)
        restored = USearchIndex.load(path)

        idx_r, dist_r = restored.query_point(data[0], k=K)
        assert idx_r.shape == (K,)
        assert np.all(dist_r >= -1e-6)

    @pytest.mark.parametrize("mode", ["exact", "hnsw"])
    def test_add_makes_new_vectors_queryable(self, mode: str) -> None:
        base = np.full((20, 4), 10.0, dtype=np.float32)
        added = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        query = np.array([0.01, 0.01, 0.01, 0.01], dtype=np.float32)

        index = self._make_index(mode=mode)
        index.build_from_vectors(base, metric="euclidean")
        keys = index.add(added)

        assert keys.tolist() == [20]
        assert index.n_nodes == 21

        indices, _ = index.query_point(query, k=1)
        assert indices[0] == 20


# ---------------------------------------------------------------------------
# Estimator integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
class TestEstimatorDenseMetrics:
    def test_cosine_fit_produces_valid_embedding(self) -> None:
        from tmap import TMAP

        data = _clustered_dense_data(n_samples=40, n_features=16)
        model = TMAP(metric="cosine", n_neighbors=5, seed=42).fit(data)

        assert model.embedding_.shape == (40, 2)
        assert model.embedding_.dtype == np.float32
        assert np.all(np.isfinite(model.embedding_))
        assert model.graph_.indices.shape == (40, 5)

    def test_euclidean_fit_produces_valid_embedding(self) -> None:
        from tmap import TMAP

        data = _clustered_dense_data(n_samples=40, n_features=16)
        model = TMAP(metric="euclidean", n_neighbors=5, seed=42).fit(data)

        assert model.embedding_.shape == (40, 2)
        assert model.embedding_.dtype == np.float32
        assert np.all(np.isfinite(model.embedding_))

    def test_cosine_fit_transform(self) -> None:
        from tmap import TMAP

        data = _clustered_dense_data(n_samples=40, n_features=16)
        x, y, s, t = TMAP(metric="cosine", n_neighbors=5, seed=42).fit_transform(data)

        assert x.shape == (40,)
        assert y.shape == (40,)
        assert len(s) > 0
        assert len(t) > 0

    def test_n_neighbors_too_large_raises(self) -> None:
        from tmap import TMAP

        data = _clustered_dense_data(n_samples=10, n_features=8)
        with pytest.raises(ValueError, match="n_neighbors.*must be < n_samples"):
            TMAP(metric="cosine", n_neighbors=10, seed=42).fit(data)

    def test_lsh_forest_is_none_for_dense_metric(self) -> None:
        from tmap import TMAP

        data = _clustered_dense_data(n_samples=40, n_features=16)
        model = TMAP(metric="cosine", n_neighbors=5, seed=42).fit(data)

        with pytest.raises(RuntimeError, match="No fitted LSHForest"):
            _ = model.lsh_forest_

    def test_store_index(self) -> None:
        from tmap import TMAP

        data = _clustered_dense_data(n_samples=40, n_features=16)
        model = TMAP(metric="cosine", n_neighbors=5, seed=42, store_index=True).fit(data)
        idx = model.index_
        assert idx.is_built

        indices, distances = idx.query_point(data[0], k=5)
        assert indices.shape == (5,)

    def test_index_not_stored_by_default(self) -> None:
        from tmap import TMAP

        data = _clustered_dense_data(n_samples=40, n_features=16)
        model = TMAP(metric="cosine", n_neighbors=5, seed=42).fit(data)

        with pytest.raises(RuntimeError, match="No index stored"):
            _ = model.index_

    def test_save_load_roundtrip_with_store_index(self, tmp_path) -> None:
        """TMAP.save/load roundtrip with store_index=True and USearch backend."""
        from tmap import TMAP

        data = _clustered_dense_data(n_samples=40, n_features=16)
        model = TMAP(metric="cosine", n_neighbors=5, seed=42, store_index=True).fit(data)
        emb_before = model.embedding_.copy()

        save_path = tmp_path / "model.tmap"
        model.save(str(save_path))
        loaded = TMAP.load(str(save_path))

        np.testing.assert_array_equal(loaded.embedding_, emb_before)
        # Loaded model's stored index can still query
        idx_l, dist_l = loaded.index_.query_point(data[0], k=5)
        assert idx_l.shape == (5,)
        assert np.all(dist_l >= -1e-6)
