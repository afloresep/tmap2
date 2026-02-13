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
    model = TMAP(n_neighbors=5, n_permutations=64, l=8, seed=123)

    coords = model.fit_transform(data)

    assert coords.shape == (data.shape[0], 2)
    assert coords.dtype == np.float32
    np.testing.assert_array_equal(coords, model.embedding_)


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
def test_list_and_array_inputs_produce_same_result() -> None:
    data = _clustered_binary_data(seed=123)
    model_array = TMAP(n_neighbors=5, n_permutations=64, l=8, seed=7).fit(data)
    model_list = TMAP(n_neighbors=5, n_permutations=64, l=8, seed=7).fit(data.tolist())

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


def test_dense_metric_not_implemented() -> None:
    data = np.random.default_rng(9).random((8, 4), dtype=np.float32)

    with pytest.raises(NotImplementedError, match="not implemented"):
        TMAP(metric="cosine", n_neighbors=3).fit(data)


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
def test_graph_layout_produces_different_embedding() -> None:
    data = _clustered_binary_data()
    model_tree = TMAP(n_neighbors=5, n_permutations=64, l=8, seed=1, layout="tree").fit(data)
    model_graph = TMAP(n_neighbors=5, n_permutations=64, l=8, seed=1, layout="graph").fit(data)

    # Same KNN graph and MST
    np.testing.assert_array_equal(model_tree.graph_.indices, model_graph.graph_.indices)
    np.testing.assert_array_equal(model_tree.tree_.edges, model_graph.tree_.edges)

    # Different coordinates (different layout input)
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


def test_invalid_layout_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported layout"):
        TMAP(layout="invalid")
