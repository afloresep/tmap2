"""
Tests for LSH Forest implementation.

Tests cover:
- Initialization and parameter validation
- Add operations (single and batch)
- Query methods (LSH-only and with linear scan)
- k-NN graph construction
- Distance computation
- Persistence (save/load)
- Weighted MinHash support
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from tmap.index import LSHForest
from tmap.index.types import KNNGraph


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def random_signatures():
    """Generate random MinHash signatures for testing."""
    rng = np.random.default_rng(42)
    # 100 signatures with d=128
    return rng.integers(0, 2**63, size=(100, 128), dtype=np.uint64)


@pytest.fixture
def small_signatures():
    """Generate a small set of signatures for detailed testing."""
    rng = np.random.default_rng(123)
    return rng.integers(0, 2**63, size=(10, 64), dtype=np.uint64)


@pytest.fixture
def weighted_signatures():
    """Generate weighted MinHash signatures (shape: n, d, 2)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 2**63, size=(50, 128, 2), dtype=np.uint64)


@pytest.fixture
def indexed_forest(random_signatures):
    """Create an indexed LSH forest for query tests."""
    lsh = LSHForest(d=128, l=8)
    lsh.batch_add(random_signatures)
    lsh.index()
    return lsh


# =============================================================================
# Initialization Tests
# =============================================================================


class TestLSHForestInit:
    """Tests for LSHForest initialization."""

    def test_default_parameters(self):
        """Test default parameter values."""
        lsh = LSHForest()
        assert lsh.d == 128
        assert lsh.l == 8
        assert lsh.size == 0
        assert not lsh.is_clean

    def test_custom_parameters(self):
        """Test custom parameter values."""
        lsh = LSHForest(d=256, l=16)
        assert lsh.d == 256
        assert lsh.l == 16

    def test_invalid_d_raises_error(self):
        """Test that d <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="d must be positive"):
            LSHForest(d=0)
        with pytest.raises(ValueError, match="d must be positive"):
            LSHForest(d=-1)

    def test_invalid_l_raises_error(self):
        """Test that l <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="l must be positive"):
            LSHForest(l=0)
        with pytest.raises(ValueError, match="l must be positive"):
            LSHForest(l=-1)

    def test_l_greater_than_d_raises_error(self):
        """Test that l > d raises ValueError."""
        with pytest.raises(ValueError, match="l cannot be greater than d"):
            LSHForest(d=64, l=128)

    def test_weighted_initialization(self):
        """Test weighted MinHash mode initialization."""
        lsh = LSHForest(d=128, weighted=True)
        assert lsh.d == 128


# =============================================================================
# Add Methods Tests
# =============================================================================


class TestLSHForestAdd:
    """Tests for add and batch_add methods."""

    def test_add_single_signature(self, small_signatures):
        """Test adding a single signature."""
        lsh = LSHForest(d=64)
        lsh.add(small_signatures[0])
        lsh.index()
        assert lsh.size == 1
        assert lsh.is_clean

    def test_batch_add_signatures(self, random_signatures):
        """Test batch adding multiple signatures."""
        lsh = LSHForest(d=128)
        lsh.batch_add(random_signatures)
        lsh.index()
        assert lsh.size == 100
        assert lsh.is_clean

    def test_add_incremental(self, small_signatures):
        """Test incremental adding of signatures."""
        lsh = LSHForest(d=64)
        lsh.add(small_signatures[0])
        lsh.add(small_signatures[1])
        lsh.index()
        assert lsh.size == 2

    def test_add_invalid_shape_raises_error(self):
        """Test that invalid signature shape raises ValueError."""
        lsh = LSHForest(d=128)
        wrong_shape = np.zeros((64,), dtype=np.uint64)
        with pytest.raises(ValueError, match="Expected shape"):
            lsh.add(wrong_shape)

    def test_batch_add_invalid_shape_raises_error(self):
        """Test that invalid batch signature shape raises ValueError."""
        lsh = LSHForest(d=128)
        wrong_shape = np.zeros((10, 64), dtype=np.uint64)
        with pytest.raises(ValueError, match="Expected shape"):
            lsh.batch_add(wrong_shape)

    def test_needs_reindex_after_add(self, small_signatures):
        """Test that is_clean becomes False after adding without reindexing."""
        lsh = LSHForest(d=64)
        lsh.batch_add(small_signatures[:5])
        lsh.index()
        assert lsh.is_clean

        # Add more without reindexing
        lsh.add(small_signatures[5])
        assert not lsh.is_clean


# =============================================================================
# Index Method Tests
# =============================================================================


class TestLSHForestIndex:
    """Tests for index method."""

    def test_index_creates_contiguous_array(self, random_signatures):
        """Test that index() creates a contiguous signature array."""
        lsh = LSHForest(d=128)
        lsh.batch_add(random_signatures)
        lsh.index()

        # Internal signatures should be contiguous
        assert lsh._signatures is not None
        assert lsh._signatures.flags["C_CONTIGUOUS"]

    def test_index_sets_is_clean(self, small_signatures):
        """Test that index() sets is_clean to True."""
        lsh = LSHForest(d=64)
        lsh.batch_add(small_signatures)
        assert not lsh.is_clean
        lsh.index()
        assert lsh.is_clean


# =============================================================================
# Query Methods Tests
# =============================================================================


class TestLSHForestQuery:
    """Tests for query methods."""

    def test_query_returns_candidates(self, indexed_forest, random_signatures):
        """Test that query returns candidate indices."""
        candidates = indexed_forest.query(random_signatures[0], k=10)
        assert isinstance(candidates, np.ndarray)
        assert candidates.dtype == np.int32
        assert len(candidates) <= 10
        # Should include self
        assert 0 in candidates

    def test_query_before_index_raises_error(self, random_signatures):
        """Test that query before index() raises RuntimeError."""
        lsh = LSHForest(d=128)
        lsh.batch_add(random_signatures)
        # Don't call index()
        with pytest.raises(RuntimeError, match="Must call index"):
            lsh.query(random_signatures[0], k=10)

    def test_query_by_id(self, indexed_forest):
        """Test query_by_id method."""
        candidates = indexed_forest.query_by_id(0, k=10)
        assert isinstance(candidates, np.ndarray)
        assert len(candidates) <= 10

    def test_query_by_id_invalid_raises_error(self, indexed_forest):
        """Test that query_by_id with invalid ID raises IndexError."""
        with pytest.raises(IndexError):
            indexed_forest.query_by_id(999, k=10)
        with pytest.raises(IndexError):
            indexed_forest.query_by_id(-1, k=10)


# =============================================================================
# Linear Scan Tests
# =============================================================================


class TestLSHForestLinearScan:
    """Tests for linear scan methods."""

    def test_linear_scan_returns_sorted(self, indexed_forest, random_signatures):
        """Test that linear_scan returns results sorted by distance."""
        candidates = indexed_forest.query(random_signatures[0], k=20)
        results = indexed_forest.linear_scan(random_signatures[0], candidates, k=5)

        # May get fewer results if not enough valid neighbors (self is excluded)
        assert len(results) <= 5
        # Check sorted by distance
        if len(results) > 0:
            distances = [r[0] for r in results]
            assert distances == sorted(distances)

    def test_linear_scan_empty_indices(self, indexed_forest, random_signatures):
        """Test linear_scan with empty indices."""
        results = indexed_forest.linear_scan(random_signatures[0], [], k=5)
        assert results == []

    def test_query_linear_scan(self, indexed_forest, random_signatures):
        """Test query_linear_scan method."""
        results = indexed_forest.query_linear_scan(random_signatures[0], k=5, kc=10)

        assert len(results) <= 5
        # Check returns (distance, index) tuples
        for dist, idx in results:
            assert 0.0 <= dist <= 1.0
            assert 0 <= idx < 100

    def test_query_linear_scan_by_id_excludes_self(self, indexed_forest):
        """Test that query_linear_scan_by_id excludes the query point itself."""
        results = indexed_forest.query_linear_scan_by_id(0, k=5, kc=10)

        # Self (index 0) should not be in results
        result_indices = [idx for _, idx in results]
        assert 0 not in result_indices


# =============================================================================
# k-NN Graph Tests
# =============================================================================


class TestLSHForestKNNGraph:
    """Tests for k-NN graph construction."""

    def test_get_knn_graph_returns_correct_type(self, indexed_forest):
        """Test that get_knn_graph returns a KNNGraph."""
        knn = indexed_forest.get_knn_graph(k=5, kc=10)
        assert isinstance(knn, KNNGraph)

    def test_get_knn_graph_shape(self, indexed_forest):
        """Test that get_knn_graph returns correct shapes."""
        knn = indexed_forest.get_knn_graph(k=5, kc=10)

        assert knn.indices.shape == (100, 5)
        assert knn.distances.shape == (100, 5)

    def test_get_knn_graph_distances_in_range(self, indexed_forest):
        """Test that valid distances are in range [0, 1]."""
        knn = indexed_forest.get_knn_graph(k=5, kc=10)

        # Valid entries have indices >= 0 and distances < 2.0 (2.0 is invalid marker)
        valid_mask = (knn.indices >= 0) & (knn.distances < 2.0)
        valid_distances = knn.distances[valid_mask]
        if len(valid_distances) > 0:
            assert np.all(valid_distances >= 0.0)
            assert np.all(valid_distances <= 1.0)

    def test_get_knn_graph_excludes_self(self, indexed_forest):
        """Test that k-NN graph excludes self-loops."""
        knn = indexed_forest.get_knn_graph(k=5, kc=10)

        for i in range(100):
            # Valid neighbors (indices >= 0) should not include self
            valid_neighbors = knn.indices[i][knn.indices[i] >= 0]
            assert i not in valid_neighbors

    def test_get_all_nearest_neighbors(self, indexed_forest):
        """Test get_all_nearest_neighbors method."""
        neighbors = indexed_forest.get_all_nearest_neighbors(k=5, kc=10)

        assert neighbors.shape == (500,)  # 100 * 5
        assert neighbors.dtype == np.int32

    def test_get_knn_graph_before_index_raises_error(self, random_signatures):
        """Test that get_knn_graph before index() raises RuntimeError."""
        lsh = LSHForest(d=128)
        lsh.batch_add(random_signatures)
        with pytest.raises(RuntimeError):
            lsh.get_knn_graph(k=5)


# =============================================================================
# Distance Methods Tests
# =============================================================================


class TestLSHForestDistance:
    """Tests for distance computation methods."""

    def test_get_distance_static(self):
        """Test static get_distance method."""
        # Create two identical signatures
        sig = np.ones(128, dtype=np.uint64)
        dist = LSHForest.get_distance(sig, sig)
        assert dist == 0.0

        # Different signatures
        sig_a = np.zeros(128, dtype=np.uint64)
        sig_b = np.ones(128, dtype=np.uint64)
        dist = LSHForest.get_distance(sig_a, sig_b)
        assert dist == 1.0  # Completely different

    def test_get_distance_by_id(self, indexed_forest):
        """Test get_distance_by_id method."""
        # Same signature should have distance 0
        dist = indexed_forest.get_distance_by_id(0, 0)
        assert dist == 0.0

        # Different signatures should have distance > 0
        dist = indexed_forest.get_distance_by_id(0, 1)
        assert 0.0 <= dist <= 1.0

    def test_get_all_distances(self, indexed_forest, random_signatures):
        """Test get_all_distances method."""
        distances = indexed_forest.get_all_distances(random_signatures[0])

        assert distances.shape == (100,)
        # Self distance should be 0
        assert distances[0] == 0.0
        # All distances in valid range
        assert np.all(distances >= 0.0)
        assert np.all(distances <= 1.0)


# =============================================================================
# Storage Tests
# =============================================================================


class TestLSHForestStorage:
    """Tests for signature storage and retrieval."""

    def test_get_hash(self, indexed_forest, random_signatures):
        """Test get_hash retrieves correct signature."""
        retrieved = indexed_forest.get_hash(0)
        np.testing.assert_array_equal(retrieved, random_signatures[0])

    def test_get_hash_invalid_id_raises_error(self, indexed_forest):
        """Test that get_hash with invalid ID raises IndexError."""
        with pytest.raises(IndexError):
            indexed_forest.get_hash(999)
        with pytest.raises(IndexError):
            indexed_forest.get_hash(-1)

    def test_store_false_disables_storage(self, random_signatures):
        """Test that store=False disables signature storage."""
        lsh = LSHForest(d=128, store=False)
        lsh.batch_add(random_signatures)
        lsh.index()

        # These methods should raise ValueError
        with pytest.raises(ValueError, match="requires store=True"):
            lsh.get_hash(0)
        with pytest.raises(ValueError, match="requires store=True"):
            lsh.linear_scan(random_signatures[0], [0, 1], k=2)
        with pytest.raises(ValueError, match="requires store=True"):
            lsh.get_knn_graph(k=5)


# =============================================================================
# Persistence Tests
# =============================================================================


class TestLSHForestPersistence:
    """Tests for save/load functionality."""

    def test_save_load_roundtrip(self, indexed_forest, random_signatures):
        """Test that save/load preserves state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lsh_forest.pkl"
            indexed_forest.save(str(path))

            loaded = LSHForest.load(str(path))

            # Check properties preserved
            assert loaded.d == indexed_forest.d
            assert loaded.l == indexed_forest.l
            assert loaded.size == indexed_forest.size
            assert loaded.is_clean == indexed_forest.is_clean

            # Check signatures preserved
            np.testing.assert_array_equal(
                loaded.get_hash(0), indexed_forest.get_hash(0)
            )

            # Check queries work
            result_orig = indexed_forest.query(random_signatures[0], k=5)
            result_loaded = loaded.query(random_signatures[0], k=5)
            np.testing.assert_array_equal(result_orig, result_loaded)


# =============================================================================
# Weighted MinHash Tests
# =============================================================================


class TestLSHForestWeighted:
    """Tests for weighted MinHash support."""

    def test_weighted_add_and_query(self, weighted_signatures):
        """Test adding and querying weighted signatures."""
        lsh = LSHForest(d=128, weighted=True)
        lsh.batch_add(weighted_signatures)
        lsh.index()

        assert lsh.size == 50

        # Query should work
        candidates = lsh.query(weighted_signatures[0], k=10)
        assert len(candidates) > 0

    def test_weighted_knn_graph(self, weighted_signatures):
        """Test k-NN graph construction with weighted signatures."""
        lsh = LSHForest(d=128, weighted=True)
        lsh.batch_add(weighted_signatures)
        lsh.index()

        knn = lsh.get_knn_graph(k=5, kc=10)

        assert knn.indices.shape == (50, 5)
        assert knn.distances.shape == (50, 5)

    def test_weighted_signature_validation(self):
        """Test that wrong shape for weighted signatures raises error."""
        lsh = LSHForest(d=128, weighted=True)

        # Non-weighted signature shape should fail
        wrong_shape = np.zeros((128,), dtype=np.uint64)
        with pytest.raises(ValueError, match="Expected shape"):
            lsh.add(wrong_shape)

    def test_get_weighted_distance_static(self):
        """Test static get_weighted_distance method."""
        # Identical signatures
        sig = np.ones((128, 2), dtype=np.uint64)
        dist = LSHForest.get_weighted_distance(sig, sig)
        assert dist == 0.0

        # Different signatures
        sig_a = np.zeros((128, 2), dtype=np.uint64)
        sig_b = np.ones((128, 2), dtype=np.uint64)
        dist = LSHForest.get_weighted_distance(sig_a, sig_b)
        assert dist == 1.0


# =============================================================================
# State Methods Tests
# =============================================================================


class TestLSHForestState:
    """Tests for state management methods."""

    def test_clear(self, indexed_forest):
        """Test that clear() resets state."""
        indexed_forest.clear()

        assert indexed_forest.size == 0
        assert not indexed_forest.is_clean
        assert indexed_forest._signatures is None
        assert indexed_forest._signatures_list == []


# =============================================================================
# Integration Tests
# =============================================================================


class TestLSHForestIntegration:
    """Integration tests for full workflow."""

    def test_full_workflow(self):
        """Test complete workflow: add -> index -> query -> knn_graph."""
        rng = np.random.default_rng(42)
        signatures = rng.integers(0, 2**63, size=(200, 128), dtype=np.uint64)

        # Build forest
        lsh = LSHForest(d=128, l=8)
        lsh.batch_add(signatures[:100])
        lsh.index()

        # Query
        neighbors = lsh.query_linear_scan(signatures[0], k=10, kc=10)
        assert len(neighbors) > 0

        # Build k-NN graph
        knn = lsh.get_knn_graph(k=10, kc=10)
        assert knn.indices.shape == (100, 10)

        # Add more and reindex
        lsh.batch_add(signatures[100:])
        lsh.index()
        assert lsh.size == 200

        # Query again
        knn = lsh.get_knn_graph(k=10, kc=10)
        assert knn.indices.shape == (200, 10)

    def test_similar_signatures_are_neighbors(self):
        """Test that similar signatures are found as neighbors."""
        rng = np.random.default_rng(42)

        # Create base signature
        base = rng.integers(0, 2**63, size=(128,), dtype=np.uint64)

        # Create similar signatures (50% overlap)
        signatures = []
        for i in range(10):
            sig = base.copy()
            # Modify half the values
            mask = rng.choice(128, size=64, replace=False)
            sig[mask] = rng.integers(0, 2**63, size=64, dtype=np.uint64)
            signatures.append(sig)

        signatures = np.array(signatures, dtype=np.uint64)

        lsh = LSHForest(d=128, l=8)
        lsh.batch_add(signatures)
        lsh.index()

        # All signatures should be somewhat similar
        for i in range(10):
            neighbors = lsh.query_linear_scan_by_id(i, k=5, kc=10)
            # Should find some neighbors with reasonable distance
            if neighbors:
                min_dist = neighbors[0][0]
                assert min_dist < 0.8  # Not completely random
