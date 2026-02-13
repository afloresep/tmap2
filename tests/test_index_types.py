import numpy as np
import pytest

from tmap.index.types import KNNGraph


def test_knn_graph_from_arrays() -> None:
    knn = KNNGraph.from_arrays(
        indices=[[1, 2], [0, 2], [1, 0]],
        distances=[[0.1, 0.2], [0.1, 0.3], [0.3, 0.2]],
    )

    assert knn.indices.shape == (3, 2)
    assert knn.distances.shape == (3, 2)
    assert knn.indices.dtype == np.int32
    assert knn.distances.dtype == np.float32


def test_knn_graph_from_arrays_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        KNNGraph.from_arrays(
            indices=[[1, 2], [0, 2]],
            distances=[[0.1], [0.2]],
        )


def test_knn_graph_from_distance_matrix_neighbors() -> None:
    distance_matrix = np.array(
        [
            [0.0, 1.0, 3.0, 2.0],
            [1.0, 0.0, 4.0, 5.0],
            [3.0, 4.0, 0.0, 1.0],
            [2.0, 5.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    knn = KNNGraph.from_distance_matrix(distance_matrix, k=2)

    np.testing.assert_array_equal(knn.indices[0], np.array([1, 3], dtype=np.int32))
    np.testing.assert_allclose(knn.distances[0], np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(knn.indices[2], np.array([3, 0], dtype=np.int32))
    np.testing.assert_allclose(knn.distances[2], np.array([1.0, 3.0], dtype=np.float32))


def test_knn_graph_from_distance_matrix_invalid_k_raises() -> None:
    distance_matrix = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="1 <= k < n_samples"):
        KNNGraph.from_distance_matrix(distance_matrix, k=3)


def test_knn_graph_from_distance_matrix_non_square_raises() -> None:
    distance_matrix = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 3.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="must be square"):
        KNNGraph.from_distance_matrix(distance_matrix, k=1)

