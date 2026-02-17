import numpy as np
import pytest

from tmap import TMAP
from tmap.index.types import KNNGraph
from tmap.layout import OGDF_AVAILABLE


def _clustered_binary_data(
    n_samples: int = 40,
    n_features: int = 128,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = (rng.random(n_features) < 0.3).astype(np.uint8)
    data = np.zeros((n_samples, n_features), dtype=np.uint8)

    for i in range(n_samples):
        row = base.copy()
        flip_mask = rng.random(n_features) < 0.08
        row[flip_mask] = 1 - row[flip_mask]
        data[i] = row

    return data


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
def test_fit_transform_returns_embedding_shape_and_dtype() -> None:
    data = _clustered_binary_data()
    model = TMAP(n_neighbors=5, n_permutations=64, seed=123)

    x, y, s, t = model.fit_transform(data)

    assert x.shape == (data.shape[0],)
    assert y.shape == (data.shape[0],)
    assert s.shape == (len(model.tree_.edges),)
    assert t.shape == (len(model.tree_.edges),)
    assert x.dtype == np.float32
    assert y.dtype == np.float32
    np.testing.assert_array_equal(x, model.embedding_[:, 0])
    np.testing.assert_array_equal(y, model.embedding_[:, 1])
    np.testing.assert_array_equal(s, model.tree_.edges[:, 0])
    np.testing.assert_array_equal(t, model.tree_.edges[:, 1])


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
def test_list_and_array_inputs_produce_same_result() -> None:
    data = _clustered_binary_data(seed=123)
    model_array = TMAP(n_neighbors=5, n_permutations=64, seed=7).fit(data)
    model_list = TMAP(n_neighbors=5, n_permutations=64, seed=7).fit(data.tolist())

    np.testing.assert_array_equal(model_array.graph_.indices, model_list.graph_.indices)
    np.testing.assert_allclose(model_array.graph_.distances, model_list.graph_.distances)
    np.testing.assert_allclose(model_array.embedding_, model_list.embedding_)


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
def test_precomputed_metric_builds_embedding() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)

    model = TMAP(metric="precomputed", n_neighbors=2, seed=11).fit(distances)

    assert model.graph_.indices.shape == (4, 2)
    assert model.graph_.distances.shape == (4, 2)
    assert model.embedding_.shape == (4, 2)


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
def test_fit_accepts_precomputed_knn_graph() -> None:
    knn = KNNGraph.from_arrays(
        indices=[[1, 2], [0, 2], [1, 3], [2, 0]],
        distances=[[0.1, 0.2], [0.1, 0.3], [0.3, 0.1], [0.1, 0.4]],
    )

    model = TMAP(seed=13).fit(knn_graph=knn)

    assert model.graph_ is knn
    assert model.embedding_.shape == (4, 2)


def test_jaccard_rejects_non_binary_input() -> None:
    data = np.array([[0, 1, 2], [1, 0, 1]], dtype=np.int32)

    with pytest.raises(ValueError, match="0/1"):
        TMAP(metric="jaccard", n_neighbors=1).fit(data)


def test_dense_metric_requires_ann_backend() -> None:
    """Cosine/euclidean should either work (if backend installed) or give ImportError."""
    data = np.random.default_rng(9).random((8, 4), dtype=np.float32)

    try:
        import pynndescent  # noqa: F401

        has_backend = True
    except ImportError:
        try:
            import faiss  # noqa: F401

            has_backend = True
        except ImportError:
            has_backend = False

    if has_backend:
        # Should not raise NotImplementedError anymore
        model = TMAP(metric="cosine", n_neighbors=3, seed=42).fit(data)
        assert model.embedding_.shape == (8, 2)
    else:
        with pytest.raises(ImportError, match="pynndescent or faiss"):
            TMAP(metric="cosine", n_neighbors=3).fit(data)


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
def test_graph_layout_builds_valid_embedding_and_tree() -> None:
    data = _clustered_binary_data()
    model_tree = TMAP(n_neighbors=5, n_permutations=64, seed=1, layout="tree").fit(data)
    model_graph = TMAP(n_neighbors=5, n_permutations=64, seed=1, layout="graph").fit(data)

    # Same KNN graph and MST
    np.testing.assert_array_equal(model_tree.graph_.indices, model_graph.graph_.indices)
    np.testing.assert_array_equal(model_tree.tree_.edges, model_graph.tree_.edges)

    # Graph mode should produce a different embedding from tree mode.
    assert model_tree.embedding_.shape == model_graph.embedding_.shape
    assert not np.allclose(model_tree.embedding_, model_graph.embedding_)


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
def test_graph_layout_with_precomputed_knn() -> None:
    knn = KNNGraph.from_arrays(
        indices=[[1, 2], [0, 2], [1, 3], [2, 0]],
        distances=[[0.1, 0.2], [0.1, 0.3], [0.3, 0.1], [0.1, 0.4]],
    )

    model = TMAP(seed=13, layout="graph").fit(knn_graph=knn)

    assert model.embedding_.shape == (4, 2)
    assert model.tree_ is not None


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
def test_jaccard_knn_is_stable_across_layout_seeds_by_default() -> None:
    data = _clustered_binary_data(n_samples=80, n_features=256, seed=321)

    model_seed_1 = TMAP(n_neighbors=8, n_permutations=128, seed=1).fit(data)
    model_seed_42 = TMAP(n_neighbors=8, n_permutations=128, seed=42).fit(data)

    np.testing.assert_array_equal(model_seed_1.graph_.indices, model_seed_42.graph_.indices)
    np.testing.assert_allclose(model_seed_1.graph_.distances, model_seed_42.graph_.distances)


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
def test_jaccard_knn_changes_when_minhash_seed_changes() -> None:
    data = _clustered_binary_data(n_samples=80, n_features=256, seed=321)

    model_seed_1 = TMAP(n_neighbors=8, n_permutations=128, seed=1, minhash_seed=1).fit(data)
    model_seed_42 = TMAP(n_neighbors=8, n_permutations=128, seed=42, minhash_seed=42).fit(data)

    same_indices = np.array_equal(model_seed_1.graph_.indices, model_seed_42.graph_.indices)
    same_distances = np.allclose(model_seed_1.graph_.distances, model_seed_42.graph_.distances)

    assert not (same_indices and same_distances)


def test_invalid_layout_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported layout"):
        TMAP(layout="invalid")


@pytest.mark.parametrize("kw", [{"l": 8}, {"lsh_num_trees": 8}])
def test_legacy_lsh_keywords_raise_type_error(kw: dict[str, int]) -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        TMAP(**kw)


# =============================================================================
# Tree exploration wrapper tests
# =============================================================================


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
def test_path_delegates_to_tree() -> None:
    data = _clustered_binary_data()
    model = TMAP(n_neighbors=5, n_permutations=64, seed=42).fit(data)
    assert model.path(0, 5) == model.tree_.path(0, 5)


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
def test_distance_delegates_to_tree() -> None:
    data = _clustered_binary_data()
    model = TMAP(n_neighbors=5, n_permutations=64, seed=42).fit(data)
    assert model.distance(0, 5) == model.tree_.distance(0, 5)


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
def test_distances_from_returns_array() -> None:
    data = _clustered_binary_data()
    model = TMAP(n_neighbors=5, n_permutations=64, seed=42).fit(data)
    dists = model.distances_from(0)
    assert dists.shape == (data.shape[0],)
    assert dists.dtype == np.float32
    assert dists[0] == 0.0


def test_path_before_fit_raises() -> None:
    model = TMAP()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.path(0, 1)
