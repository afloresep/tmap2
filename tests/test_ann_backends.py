"""Tests for FaissIndex and estimator cosine/euclidean integration."""

from __future__ import annotations

import numpy as np
import pytest

from tmap.index.types import KNNGraph
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
# FaissIndex tests
# ---------------------------------------------------------------------------


class TestFaissIndex:
    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        pytest.importorskip("faiss")

    def _make_index(self, **kwargs):
        from tmap.index.faiss_index import FaissIndex

        kwargs.setdefault("seed", 42)
        return FaissIndex(**kwargs)

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
        assert np.all(knn.distances >= 0)
        for i in range(data.shape[0]):
            assert i not in knn.indices[i]

    def test_query_single(self) -> None:
        data = _clustered_dense_data()
        index = self._make_index()
        index.build_from_vectors(data, metric="euclidean")

        indices, distances = index.query_point(data[0], k=K)
        assert indices.shape == (K,)
        assert distances.shape == (K,)

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
        from tmap.index.faiss_index import FaissIndex

        index = FaissIndex()
        with pytest.raises(ValueError, match="does not support metric"):
            index.build_from_vectors(_clustered_dense_data(), metric="jaccard")

    def test_auto_mode_picks_flat_for_small(self) -> None:
        """Auto mode should select flat index for small datasets."""
        index = self._make_index(mode="auto")
        data = _clustered_dense_data(n_samples=100)
        index.build_from_vectors(data, metric="euclidean")
        assert index.effective_mode == "flat"

    def test_hnsw_cosine_distances_nonneg(self) -> None:
        """HNSW with cosine metric returns non-negative distances."""
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
        from tmap.index.faiss_index import FaissIndex

        with pytest.raises(ValueError, match="mode must be"):
            FaissIndex(mode="bad")


# ---------------------------------------------------------------------------
# Estimator integration
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
class TestEstimatorDenseMetrics:
    @pytest.fixture(autouse=True)
    def _skip_if_no_backend(self):
        pytest.importorskip("faiss")

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
