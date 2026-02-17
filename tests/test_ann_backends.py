"""Tests for NNDescentIndex, FaissIndex, and estimator cosine/euclidean integration."""

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
# NNDescentIndex tests
# ---------------------------------------------------------------------------

class TestNNDescentIndex:
    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        pytest.importorskip("pynndescent")

    def _make_index(self):
        from tmap.index.nndescent import NNDescentIndex
        return NNDescentIndex(seed=42)

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

        # Indices in valid range
        assert np.all(knn.indices >= 0)
        assert np.all(knn.indices < data.shape[0])
        # Non-negative distances
        assert np.all(knn.distances >= 0)
        # No self-matches
        for i in range(data.shape[0]):
            assert i not in knn.indices[i]

    def test_query_single_works_without_query_knn(self) -> None:
        """query_point must work immediately after build — no query_knn prerequisite."""
        data = _clustered_dense_data()
        index = self._make_index()
        index.build_from_vectors(data, metric="euclidean")

        # Do NOT call query_knn first
        indices, distances = index.query_point(data[0], k=K)
        assert indices.shape == (K,)
        assert distances.shape == (K,)
        assert indices.dtype == np.int32
        assert distances.dtype == np.float32

    def test_recall_above_90_percent(self) -> None:
        """NNDescent with n_trees=32 should achieve >90% recall vs brute-force."""
        from tmap.index.nndescent import NNDescentIndex

        rng = np.random.default_rng(123)
        data = rng.standard_normal((2000, 32)).astype(np.float32)
        k = 10

        # Brute-force ground truth
        from scipy.spatial.distance import cdist
        dists = cdist(data, data, metric="euclidean").astype(np.float32)
        np.fill_diagonal(dists, np.inf)
        gt_indices = np.argsort(dists, axis=1)[:, :k]

        # NNDescent
        index = NNDescentIndex(seed=42)
        index.build_from_vectors(data, metric="euclidean")
        knn = index.query_knn(k=k)

        # Compute recall
        hits = 0
        for i in range(data.shape[0]):
            hits += len(set(knn.indices[i].tolist()) & set(gt_indices[i].tolist()))
        recall = hits / (data.shape[0] * k)
        assert recall > 0.90, f"Recall {recall:.3f} too low (expected > 0.90)"

    def test_duplicate_vectors(self) -> None:
        """Duplicate vectors should return distance=0 neighbors."""
        data = np.ones((20, 8), dtype=np.float32)
        # Make first 10 identical, last 10 different
        data[10:] *= 5.0

        index = self._make_index()
        index.build_from_vectors(data, metric="euclidean")
        knn = index.query_knn(k=3)

        # First point's neighbors should all be from the 0..9 cluster
        for j in knn.indices[0]:
            assert j < 10
        # Distances to identical points should be ~0
        assert np.allclose(knn.distances[0], 0.0, atol=1e-5)

    def test_metric_property(self) -> None:
        data = _clustered_dense_data()
        index = self._make_index()
        assert index.metric is None
        index.build_from_vectors(data, metric="cosine")
        assert index.metric == "cosine"


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

    def test_explicit_ivf_mode(self) -> None:
        """Explicit IVF mode builds and queries correctly."""
        index = self._make_index(mode="ivf")
        data = _clustered_dense_data(n_samples=500, n_features=32)
        index.build_from_vectors(data, metric="euclidean")
        assert index.effective_mode == "ivf"

        knn = index.query_knn(k=K)
        assert knn.indices.shape == (500, K)
        assert np.all(knn.indices >= 0)
        assert np.all(knn.indices < 500)
        assert np.all(knn.distances >= 0)
        # No self-matches
        for i in range(500):
            assert i not in knn.indices[i]

    def test_explicit_ivfpq_mode(self) -> None:
        """Explicit IVFPQ mode builds and queries correctly at n=1000."""
        index = self._make_index(mode="ivfpq")
        data = _clustered_dense_data(n_samples=1000, n_features=32)
        index.build_from_vectors(data, metric="euclidean")
        assert index.effective_mode == "ivfpq"

        knn = index.query_knn(k=K)
        assert knn.indices.shape == (1000, K)
        assert np.all(knn.indices >= 0)
        assert np.all(knn.indices < 1000)

    def test_ivf_cosine_distances_nonneg(self) -> None:
        """IVF with cosine metric returns non-negative distances."""
        index = self._make_index(mode="ivf")
        data = _clustered_dense_data(n_samples=500, n_features=32)
        index.build_from_vectors(data, metric="cosine")

        knn = index.query_knn(k=K)
        assert np.all(knn.distances >= -1e-6), (
            f"Negative distances found: min={knn.distances.min()}"
        )

    def test_ivfpq_knn_contract(self) -> None:
        """IVFPQ: shape, dtype, valid indices, no self-match."""
        index = self._make_index(mode="ivfpq")
        data = _clustered_dense_data(n_samples=1000, n_features=32)
        index.build_from_vectors(data, metric="euclidean")
        knn = index.query_knn(k=K)

        assert knn.indices.dtype == np.int32
        assert knn.distances.dtype == np.float32
        assert knn.indices.shape == (1000, K)
        assert np.all(knn.indices >= 0)
        assert np.all(knn.indices < 1000)
        # Spot-check no self-match for first 100 rows
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
# Cross-backend consistency
# ---------------------------------------------------------------------------

def test_backends_produce_similar_knn() -> None:
    pytest.importorskip("pynndescent")
    pytest.importorskip("faiss")

    from tmap.index.faiss_index import FaissIndex
    from tmap.index.nndescent import NNDescentIndex

    data = _clustered_dense_data(n_samples=60, seed=7)
    k = 5

    nnd = NNDescentIndex(seed=42)
    nnd.build_from_vectors(data, metric="euclidean")
    knn_nnd = nnd.query_knn(k=k)

    fi = FaissIndex(seed=42)
    fi.build_from_vectors(data, metric="euclidean")
    knn_faiss = fi.query_knn(k=k)

    # Both should find similar (not necessarily identical) neighborhoods.
    # Check that at least 60% of neighbors overlap on average.
    overlap = 0
    for i in range(data.shape[0]):
        s1 = set(knn_nnd.indices[i].tolist())
        s2 = set(knn_faiss.indices[i].tolist())
        overlap += len(s1 & s2)
    avg_overlap = overlap / (data.shape[0] * k)
    assert avg_overlap > 0.6, f"Average neighbor overlap {avg_overlap:.2f} is too low"


# ---------------------------------------------------------------------------
# Estimator integration
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
class TestEstimatorDenseMetrics:
    @pytest.fixture(autouse=True)
    def _skip_if_no_backend(self):
        # Need at least one ANN backend
        try:
            import pynndescent  # noqa: F401
        except ImportError:
            try:
                import faiss  # noqa: F401
            except ImportError:
                pytest.skip("Neither pynndescent nor faiss installed")

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

        # query_point should work
        indices, distances = idx.query_point(data[0], k=5)
        assert indices.shape == (5,)

    def test_index_not_stored_by_default(self) -> None:
        from tmap import TMAP

        data = _clustered_dense_data(n_samples=40, n_features=16)
        model = TMAP(metric="cosine", n_neighbors=5, seed=42).fit(data)

        with pytest.raises(RuntimeError, match="No index stored"):
            _ = model.index_
