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


# =============================================================================
# add_points() tests
# =============================================================================


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
def test_add_points_jaccard_shape() -> None:
    data = _clustered_binary_data(n_samples=40, n_features=128)
    model = TMAP(n_neighbors=5, n_permutations=64, seed=42).fit(data)

    new_data = _clustered_binary_data(n_samples=5, n_features=128, seed=99)
    coords = model.add_points(new_data)

    assert coords.shape == (5, 2)
    assert coords.dtype == np.float32
    assert model.embedding_.shape == (45, 2)
    assert model.graph_.indices.shape == (45, 5)
    assert model.graph_.distances.shape == (45, 5)


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
def test_add_points_jaccard_extends_tree() -> None:
    data = _clustered_binary_data(n_samples=40, n_features=128)
    model = TMAP(n_neighbors=5, n_permutations=64, seed=42).fit(data)

    new_data = _clustered_binary_data(n_samples=5, n_features=128, seed=99)
    model.add_points(new_data)

    assert model.tree_.n_nodes == 45
    # Original tree has n-1 edges, add_points adds 1 edge per new point
    assert len(model.tree_.edges) == 39 + 5


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
def test_add_points_existing_coords_unchanged() -> None:
    data = _clustered_binary_data(n_samples=40, n_features=128)
    model = TMAP(n_neighbors=5, n_permutations=64, seed=42).fit(data)
    original_coords = model.embedding_.copy()

    new_data = _clustered_binary_data(n_samples=5, n_features=128, seed=99)
    model.add_points(new_data)

    np.testing.assert_array_equal(model.embedding_[:40], original_coords)


def test_add_points_cosine_requires_store_index() -> None:
    try:
        import pynndescent  # noqa: F401
        has_backend = True
    except ImportError:
        try:
            import faiss  # noqa: F401
            has_backend = True
        except ImportError:
            has_backend = False

    if not has_backend:
        pytest.skip("No ANN backend available")

    data = np.random.default_rng(9).random((20, 8), dtype=np.float32)
    model = TMAP(metric="cosine", n_neighbors=3, store_index=False, seed=42).fit(data)

    new_data = np.random.default_rng(10).random((3, 8), dtype=np.float32)
    with pytest.raises(RuntimeError, match="store_index"):
        model.add_points(new_data)


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
def test_add_points_cosine_with_store_index() -> None:
    try:
        import pynndescent  # noqa: F401
        has_backend = True
    except ImportError:
        try:
            import faiss  # noqa: F401
            has_backend = True
        except ImportError:
            has_backend = False

    if not has_backend:
        pytest.skip("No ANN backend available")

    data = np.random.default_rng(9).random((20, 8), dtype=np.float32)
    model = TMAP(metric="cosine", n_neighbors=3, store_index=True, seed=42).fit(data)

    new_data = np.random.default_rng(10).random((3, 8), dtype=np.float32)
    coords = model.add_points(new_data)

    assert coords.shape == (3, 2)
    assert model.embedding_.shape == (23, 2)
    assert model.tree_.n_nodes == 23


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
def test_add_points_precomputed() -> None:
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    dist_full = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    model = TMAP(metric="precomputed", n_neighbors=2, seed=11).fit(dist_full)

    # New point at (0.5, 0.5) — distances to existing 4 points
    new_pt = np.array([[0.5, 0.5]], dtype=np.float32)
    new_dists = np.linalg.norm(new_pt[:, None, :] - points[None, :, :], axis=2).astype(
        np.float32
    )

    coords = model.add_points(new_dists)
    assert coords.shape == (1, 2)
    assert model.embedding_.shape == (5, 2)
    assert model.tree_.n_nodes == 5


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
def test_add_points_precomputed_wrong_shape_raises() -> None:
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    dist_full = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    model = TMAP(metric="precomputed", n_neighbors=2, seed=11).fit(dist_full)

    # Wrong number of columns (3 instead of 4)
    bad_dists = np.random.default_rng(0).random((2, 3)).astype(np.float32)
    with pytest.raises(ValueError, match="n_existing"):
        model.add_points(bad_dists)


def test_add_points_before_fit_raises() -> None:
    model = TMAP()
    with pytest.raises(RuntimeError, match="fit"):
        model.add_points(np.zeros((3, 10), dtype=np.uint8))


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
def test_add_points_empty_input() -> None:
    data = _clustered_binary_data(n_samples=40, n_features=128)
    model = TMAP(n_neighbors=5, n_permutations=64, seed=42).fit(data)
    original_embedding = model.embedding_.copy()
    original_n_nodes = model.tree_.n_nodes

    empty = np.zeros((0, 128), dtype=np.uint8)
    coords = model.add_points(empty)

    assert coords.shape == (0, 2)
    np.testing.assert_array_equal(model.embedding_, original_embedding)
    assert model.tree_.n_nodes == original_n_nodes


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
def test_add_points_multiple_batches() -> None:
    data = _clustered_binary_data(n_samples=40, n_features=128)
    model = TMAP(n_neighbors=5, n_permutations=64, seed=42).fit(data)

    batch1 = _clustered_binary_data(n_samples=3, n_features=128, seed=80)
    model.add_points(batch1)
    assert model.embedding_.shape == (43, 2)
    assert model.tree_.n_nodes == 43

    batch2 = _clustered_binary_data(n_samples=4, n_features=128, seed=81)
    model.add_points(batch2)
    assert model.embedding_.shape == (47, 2)
    assert model.tree_.n_nodes == 47
    assert len(model.tree_.edges) == 39 + 3 + 4


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
def test_add_points_tree_traversal_works() -> None:
    """Path/distance across old→new boundary should work."""
    data = _clustered_binary_data(n_samples=40, n_features=128)
    model = TMAP(n_neighbors=5, n_permutations=64, seed=42).fit(data)

    new_data = _clustered_binary_data(n_samples=3, n_features=128, seed=99)
    model.add_points(new_data)

    # Path from first original node to first new node
    p = model.path(0, 40)
    assert p[0] == 0
    assert p[-1] == 40
    assert len(p) >= 2

    d = model.distance(0, 40)
    assert d > 0


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
def test_add_points_duplicate_near_original() -> None:
    """A point identical to an existing one should be placed nearby."""
    data = _clustered_binary_data(n_samples=40, n_features=128)
    model = TMAP(n_neighbors=5, n_permutations=64, seed=42).fit(data)

    original_coord = model.embedding_[0].copy()

    # Insert row identical to the first sample
    duplicate = data[0:1].copy()
    coords = model.add_points(duplicate)

    # Should be within a small radius of the original
    dist = np.linalg.norm(coords[0] - original_coord)
    embedding_range = np.linalg.norm(
        model.embedding_.max(axis=0) - model.embedding_.min(axis=0)
    )
    assert dist < 0.05 * embedding_range


# =============================================================================
# save / load tests
# =============================================================================


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
def test_save_load_roundtrip(tmp_path: object) -> None:
    """Save, load, and verify all state is preserved."""
    import pathlib
    tmp = pathlib.Path(str(tmp_path))

    data = _clustered_binary_data(n_samples=40, n_features=128)
    model = TMAP(n_neighbors=5, n_permutations=64, seed=42).fit(data)

    saved_path = model.save(tmp / "model.tmap")
    assert saved_path.exists()

    loaded = TMAP.load(saved_path)

    np.testing.assert_array_equal(loaded.embedding_, model.embedding_)
    np.testing.assert_array_equal(loaded.tree_.edges, model.tree_.edges)
    np.testing.assert_array_equal(loaded.tree_.weights, model.tree_.weights)
    np.testing.assert_array_equal(loaded.graph_.indices, model.graph_.indices)
    np.testing.assert_array_equal(loaded.graph_.distances, model.graph_.distances)
    assert loaded.tree_.n_nodes == model.tree_.n_nodes
    assert loaded.n_neighbors == model.n_neighbors
    assert loaded.metric == model.metric
    assert loaded.lsh_forest_.size == model.lsh_forest_.size


@pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")
def test_save_load_then_add_points(tmp_path: object) -> None:
    """Load a saved model and add_points should work."""
    import pathlib
    tmp = pathlib.Path(str(tmp_path))

    data = _clustered_binary_data(n_samples=40, n_features=128)
    model = TMAP(n_neighbors=5, n_permutations=64, seed=42).fit(data)
    model.save(tmp / "model.tmap")

    loaded = TMAP.load(tmp / "model.tmap")
    new_data = _clustered_binary_data(n_samples=5, n_features=128, seed=99)
    coords = loaded.add_points(new_data)

    assert coords.shape == (5, 2)
    assert loaded.embedding_.shape == (45, 2)
    assert loaded.tree_.n_nodes == 45


def test_save_before_fit_raises(tmp_path: object) -> None:
    import pathlib
    tmp = pathlib.Path(str(tmp_path))

    model = TMAP()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.save(tmp / "model.tmap")
