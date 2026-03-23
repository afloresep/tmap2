"""
Test suite for MinHash and WeightedMinHash encoders.

Tests verify:
- Correct output shapes and dtypes
- Deterministic behavior with fixed seeds
- Jaccard estimate quality (statistical properties)
- Error handling for invalid inputs
- API compatibility (from_* methods)
"""

import builtins
import sys

import numpy as np
import pytest
from datasketch.weighted_minhash import WeightedMinHashGenerator

import tmap.index.encoders.minhash as minhash_module
from tmap.index.encoders.minhash import MinHash, WeightedMinHash

# =============================================================================
# MinHash Tests
# =============================================================================


class TestMinHashInit:
    """Test MinHash initialization."""

    def test_default_parameters(self):
        mh = MinHash()
        assert mh._num_perm == 128
        assert mh._seed == 1

    def test_custom_parameters(self):
        mh = MinHash(num_perm=64, seed=42)
        assert mh._num_perm == 64
        assert mh._seed == 42


class TestMinHashEncode:
    """Test MinHash.encode() method."""

    def test_encode_binary_array_shape(self):
        mh = MinHash(num_perm=32)
        data = np.array([[1, 0, 1, 1, 0], [0, 1, 1, 0, 1]], dtype=np.uint8)
        sigs = mh.encode(data)
        assert sigs.shape == (2, 32)
        assert sigs.dtype == np.uint64

    def test_encode_list_of_sets(self):
        mh = MinHash(num_perm=32)
        data = [{1, 2, 3}, {2, 3, 4}]
        sigs = mh.encode(data)  # type: ignore
        assert sigs.shape == (2, 32)
        assert sigs.dtype == np.uint64

    def test_encode_list_of_lists(self):
        """encode() should handle list of lists by converting to sets."""
        mh = MinHash(num_perm=32)
        data = [[1, 2, 3], [2, 3, 4]]
        sigs = mh.encode(data)  # type: ignore
        assert sigs.shape == (2, 32)

    def test_encode_string_sets(self):
        mh = MinHash(num_perm=32)
        data = [{"hello", "world"}, {"hello", "there"}]
        sigs = mh.encode(data)  # type: ignore
        assert sigs.shape == (2, 32)

    def test_encode_empty_set(self):
        """Empty sets should produce valid signatures (all max values)."""
        mh = MinHash(num_perm=8)
        sigs = mh.encode([set()])
        assert sigs.shape == (1, 8)

    def test_encode_single_element(self):
        mh = MinHash(num_perm=8)
        sigs = mh.encode([{42}])
        assert sigs.shape == (1, 8)


class TestMinHashDeterminism:
    """Test that MinHash produces deterministic results with same seed."""

    def test_same_seed_same_result(self):
        mh1 = MinHash(num_perm=32, seed=42)
        mh2 = MinHash(num_perm=32, seed=42)
        data = [{1, 2, 3, 4, 5}]

        sig1 = mh1.encode(data)  # type:ignore
        sig2 = mh2.encode(data)  # type:ignore

        np.testing.assert_array_equal(sig1, sig2)

    def test_different_seed_different_result(self):
        mh1 = MinHash(num_perm=32, seed=1)
        mh2 = MinHash(num_perm=32, seed=2)
        data = [{1, 2, 3, 4, 5}]

        sig1 = mh1.encode(data)  # type: ignore
        sig2 = mh2.encode(data)  # type: ignore

        assert not np.array_equal(sig1, sig2)


class TestMinHashJaccardQuality:
    """Test that MinHash produces accurate Jaccard estimates for sets."""

    def test_integer_sets_jaccard_quality(self):
        """Integer sets should give accurate Jaccard estimates."""
        mh = MinHash(num_perm=512, seed=42)
        set1 = {1, 2, 3, 4, 5, 6, 7, 8}
        set2 = {5, 6, 7, 8, 9, 10, 11, 12}

        sig1 = mh.encode([set1])[0]
        sig2 = mh.encode([set2])[0]

        estimated = 1 - mh.get_distance(sig1, sig2)
        true_jaccard = len(set1 & set2) / len(set1 | set2)  # 4/12 = 0.333

        assert abs(estimated - true_jaccard) < 0.1

    def test_string_sets_jaccard_quality(self):
        """String sets should give accurate Jaccard estimates."""
        mh = MinHash(num_perm=512, seed=42)
        set1 = {"apple", "banana", "cherry", "date", "elderberry"}
        set2 = {"cherry", "date", "elderberry", "fig", "grape"}

        sig1 = mh.encode([set1])[0]
        sig2 = mh.encode([set2])[0]

        estimated = 1 - mh.get_distance(sig1, sig2)
        true_jaccard = len(set1 & set2) / len(set1 | set2)  # 3/7 ≈ 0.429

        assert abs(estimated - true_jaccard) < 0.1


class TestMinHashDistance:
    """Test MinHash distance calculations."""

    def test_identical_sets_zero_distance(self):
        mh = MinHash(num_perm=128)
        sig = mh.encode([{1, 2, 3}])[0]
        assert mh.get_distance(sig, sig) == 0.0

    def test_disjoint_sets_high_distance(self):
        """Completely disjoint sets should have distance close to 1.0."""
        mh = MinHash(num_perm=256, seed=42)
        sig1 = mh.encode([{1, 2, 3, 4, 5}])[0]
        sig2 = mh.encode([{100, 200, 300, 400, 500}])[0]
        distance = mh.get_distance(sig1, sig2)
        # With disjoint sets, distance should be close to 1.0
        assert distance > 0.8

    def test_distance_bounds(self):
        """Distance should always be between 0 and 1."""
        mh = MinHash(num_perm=64)
        for _ in range(10):
            set1 = set(np.random.randint(0, 100, size=20))
            set2 = set(np.random.randint(0, 100, size=20))
            sig1 = mh.encode([set1])[0]
            sig2 = mh.encode([set2])[0]
            distance = mh.get_distance(sig1, sig2)
            assert 0.0 <= distance <= 1.0

    def test_distance_symmetry(self):
        """Distance should be symmetric: d(a,b) == d(b,a)."""
        mh = MinHash(num_perm=64)
        sig1 = mh.encode([{1, 2, 3}])[0]
        sig2 = mh.encode([{2, 3, 4}])[0]
        assert mh.get_distance(sig1, sig2) == mh.get_distance(sig2, sig1)

    def test_jaccard_distance_alias(self):
        """jaccard_distance should be an alias for get_distance."""
        mh = MinHash(num_perm=32)
        sig1 = mh.encode([{1, 2, 3}])[0]
        sig2 = mh.encode([{2, 3, 4}])[0]
        assert mh.get_distance(sig1, sig2) == mh.jaccard_distance(sig1, sig2)

    def test_distance_approximates_true_jaccard(self):
        """MinHash distance should approximate true Jaccard distance."""
        mh = MinHash(num_perm=512, seed=42)  # High num_perm for accuracy
        set1 = {0, 1, 2, 3, 5}
        set2 = {0, 1, 2, 3, 4}

        sig1 = mh.encode([set1])[0]
        sig2 = mh.encode([set2])[0]

        estimated_distance = mh.get_distance(sig1, sig2)

        # True Jaccard: intersection=4, union=6, similarity=4/6, distance=2/6≈0.333
        true_distance = 1 - len(set1 & set2) / len(set1 | set2)

        # Should be within 0.1 of true value with 512 permutations
        assert abs(estimated_distance - true_distance) < 0.1


class TestMinHashDistanceErrors:
    """Test error handling in distance calculations."""

    def test_mismatched_lengths_raises(self):
        mh = MinHash(num_perm=32)
        sig1 = mh.encode([{1, 2}])[0]

        mh2 = MinHash(num_perm=64)
        sig2 = mh2.encode([{1, 2}])[0]

        with pytest.raises(ValueError, match="different.*permutations"):
            mh.get_distance(sig1, sig2)


class TestMinHashFromMethods:
    """Test MinHash from_* convenience methods."""

    def test_from_binary_array_shape(self):
        mh = MinHash(num_perm=16)
        arr = np.array([1, 0, 1, 1, 0, 1], dtype=np.uint8)
        sig = mh.from_binary_array(arr)
        assert sig.shape == (16,)
        assert sig.dtype == np.uint64

    def test_from_binary_array_consistency(self):
        """from_binary_array should match encode for equivalent input."""
        mh = MinHash(num_perm=32, seed=1)
        arr = np.array([1, 0, 1, 1, 0, 1], dtype=np.uint8)

        sig1 = mh.from_binary_array(arr)
        sig2 = mh.encode(arr.reshape(1, -1))[0]

        np.testing.assert_array_equal(sig1, sig2)

    def test_from_sparse_binary_array_shape(self):
        mh = MinHash(num_perm=16)
        indices = [0, 5, 10, 15, 20]
        sig = mh.from_sparse_binary_array(indices)
        assert sig.shape == (16,)

    def test_from_sparse_binary_array_consistency(self):
        """Sparse and dense representations should produce same signature."""
        mh = MinHash(num_perm=64, seed=42)

        # Sparse: list of indices
        sparse = [0, 2, 5, 10]
        sig_sparse = mh.from_sparse_binary_array(sparse)

        # Dense: binary array with 1s at those indices
        dense = np.zeros(20, dtype=np.uint8)
        dense[sparse] = 1
        sig_dense = mh.from_binary_array(dense)

        np.testing.assert_array_equal(sig_sparse, sig_dense)

    def test_from_string_array_shape(self):
        mh = MinHash(num_perm=16)
        strings = ["hello", "world", "test"]
        sig = mh.from_string_array(strings)
        assert sig.shape == (16,)

    def test_from_string_array_consistency(self):
        """from_string_array should match encode for equivalent input."""
        mh = MinHash(num_perm=32, seed=1)
        strings = ["foo", "bar", "baz"]

        sig1 = mh.from_string_array(strings)
        sig2 = mh.encode([set(strings)])[0]

        np.testing.assert_array_equal(sig1, sig2)


class TestMinHashOptionalDependencies:
    """Test that optional dependencies are only loaded on relevant paths."""

    def test_binary_and_integer_set_paths_skip_optional_deps(self, monkeypatch):
        def _unexpected() -> None:
            raise AssertionError("Optional dependency loader should not be called")

        monkeypatch.setattr(minhash_module, "_get_xxhash", _unexpected)
        monkeypatch.setattr(minhash_module, "_get_weighted_minhash_generator", _unexpected)

        mh = MinHash(num_perm=16, seed=42)

        binary = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
        sigs_binary = mh.encode(binary)
        assert sigs_binary.shape == (2, 16)

        sigs_sets = mh.encode([{1, 2, 3}, {2, 3, 4}])  # type: ignore[arg-type]
        assert sigs_sets.shape == (2, 16)

    def test_string_path_missing_xxhash_raises_clear_error(self, monkeypatch):
        real_import = builtins.__import__

        def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "xxhash":
                raise ModuleNotFoundError("No module named 'xxhash'")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.delitem(sys.modules, "xxhash", raising=False)
        monkeypatch.setattr(builtins, "__import__", _fake_import)

        mh = MinHash(num_perm=16, seed=42)
        with pytest.raises(ModuleNotFoundError, match="optional dependency 'xxhash'"):
            mh.from_string_array(["hello", "world"])

    def test_weighted_path_missing_datasketch_raises_clear_error(self, monkeypatch):
        real_import = builtins.__import__

        def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "datasketch.weighted_minhash":
                raise ModuleNotFoundError("No module named 'datasketch'")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.delitem(sys.modules, "datasketch.weighted_minhash", raising=False)
        monkeypatch.delitem(sys.modules, "datasketch", raising=False)
        monkeypatch.setattr(builtins, "__import__", _fake_import)

        with pytest.raises(ModuleNotFoundError, match="optional dependency 'datasketch'"):
            WeightedMinHash(dim=4, num_perm=16, seed=42)


# =============================================================================
# WeightedMinHash Tests
# =============================================================================


class TestWeightedMinHashInit:
    """Test WeightedMinHash initialization."""

    def test_default_parameters(self):
        wmh = WeightedMinHash(dim=10)
        assert wmh._dim == 10
        assert wmh._num_perm == 128
        assert wmh._seed == 1

    def test_custom_parameters(self):
        wmh = WeightedMinHash(dim=20, num_perm=64, seed=42)
        assert wmh._dim == 20
        assert wmh._num_perm == 64
        assert wmh._seed == 42


class TestWeightedMinHashEncode:
    """Test WeightedMinHash.encode() method."""

    def test_encode_shape(self):
        wmh = WeightedMinHash(dim=5, num_perm=32)
        data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]])
        sigs = wmh.encode(data)
        # Weighted MinHash has shape (n_samples, num_perm, 2)
        assert sigs.shape == (2, 32, 2)
        assert sigs.dtype == np.uint64

    def test_encode_single_sample(self):
        wmh = WeightedMinHash(dim=3, num_perm=16)
        data = np.array([[1.0, 2.0, 3.0]])
        sigs = wmh.encode(data)
        assert sigs.shape == (1, 16, 2)


class TestWeightedMinHashDeterminism:
    """Test WeightedMinHash determinism."""

    def test_same_seed_same_result(self):
        wmh1 = WeightedMinHash(dim=5, num_perm=32, seed=42)
        wmh2 = WeightedMinHash(dim=5, num_perm=32, seed=42)
        data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

        sig1 = wmh1.encode(data)
        sig2 = wmh2.encode(data)

        np.testing.assert_array_equal(sig1, sig2)

    def test_different_seed_different_result(self):
        wmh1 = WeightedMinHash(dim=5, num_perm=32, seed=1)
        wmh2 = WeightedMinHash(dim=5, num_perm=32, seed=2)
        data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

        sig1 = wmh1.encode(data)
        sig2 = wmh2.encode(data)

        assert not np.array_equal(sig1, sig2)


class TestWeightedMinHashDatasketchConsistency:
    """Test consistency with datasketch WeightedMinHash."""

    def test_consistency_with_datasketch(self):
        """Our implementation should produce identical hashes to datasketch."""
        dim = 5
        num_perm = 32
        seed = 42
        test_vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Our implementation
        wmh = WeightedMinHash(dim=dim, num_perm=num_perm, seed=seed)
        our_sig = wmh.from_weight_array(test_vec)

        # Datasketch implementation
        gen = WeightedMinHashGenerator(dim=dim, sample_size=num_perm, seed=seed)
        ds_wm = gen.minhash(test_vec)
        ds_sig = ds_wm.hashvalues

        np.testing.assert_array_equal(our_sig, ds_sig)


class TestWeightedMinHashDistance:
    """Test WeightedMinHash distance calculations."""

    def test_identical_vectors_zero_distance(self):
        wmh = WeightedMinHash(dim=5, num_perm=64)
        sig = wmh.from_weight_array(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert wmh.get_weighted_distance(sig, sig) == 0.0

    def test_distance_bounds(self):
        """Distance should always be between 0 and 1."""
        wmh = WeightedMinHash(dim=10, num_perm=64)
        for _ in range(10):
            v1 = np.random.uniform(0.1, 10, size=10)
            v2 = np.random.uniform(0.1, 10, size=10)
            sig1 = wmh.from_weight_array(v1)
            sig2 = wmh.from_weight_array(v2)
            distance = wmh.get_weighted_distance(sig1, sig2)
            assert 0.0 <= distance <= 1.0

    def test_distance_symmetry(self):
        """Distance should be symmetric."""
        wmh = WeightedMinHash(dim=5, num_perm=64)
        sig1 = wmh.from_weight_array(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        sig2 = wmh.from_weight_array(np.array([5.0, 4.0, 3.0, 2.0, 1.0]))
        assert wmh.get_weighted_distance(sig1, sig2) == wmh.get_weighted_distance(sig2, sig1)

    def test_similar_vectors_low_distance(self):
        """Very similar vectors should have low distance."""
        wmh = WeightedMinHash(dim=5, num_perm=128, seed=42)
        v1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v2 = np.array([1.0, 2.0, 3.0, 4.0, 5.1])  # Small change

        sig1 = wmh.from_weight_array(v1)
        sig2 = wmh.from_weight_array(v2)
        distance = wmh.get_weighted_distance(sig1, sig2)

        assert distance < 0.3  # Should be relatively low


class TestWeightedMinHashDistanceErrors:
    """Test error handling in weighted distance calculations."""

    def test_shape_mismatch_raises(self):
        wmh = WeightedMinHash(dim=5, num_perm=32)
        sig1 = wmh.from_weight_array(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

        wmh2 = WeightedMinHash(dim=5, num_perm=64)
        sig2 = wmh2.from_weight_array(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

        with pytest.raises(ValueError, match="Shape mismatch"):
            wmh.get_weighted_distance(sig1, sig2)

    def test_wrong_ndim_raises(self):
        wmh = WeightedMinHash(dim=5, num_perm=32)
        # Create a 1D array instead of 2D
        bad_sig = np.array([1, 2, 3, 4], dtype=np.uint64)

        with pytest.raises(ValueError, match="Expected shape"):
            wmh.get_weighted_distance(bad_sig, bad_sig)


class TestWeightedMinHashEncodeErrors:
    """Test error handling in encode method."""

    def test_dimension_mismatch_raises(self):
        wmh = WeightedMinHash(dim=5, num_perm=32)
        wrong_dim_data = np.array([[1.0, 2.0, 3.0]])  # 3 features, expected 5

        with pytest.raises(ValueError, match="Expected 5 features"):
            wmh.encode(wrong_dim_data)

    def test_negative_values_raises(self):
        wmh = WeightedMinHash(dim=3, num_perm=32)
        negative_data = np.array([[1.0, -2.0, 3.0]])

        with pytest.raises(ValueError, match="positive"):
            wmh.encode(negative_data)

    def test_zero_values_raises(self):
        """Values must be strictly positive (>0), not just non-negative."""
        wmh = WeightedMinHash(dim=3, num_perm=32)
        zero_data = np.array([[1.0, 0.0, 3.0]])

        with pytest.raises(ValueError, match="positive"):
            wmh.encode(zero_data)


class TestWeightedMinHashFromMethods:
    """Test WeightedMinHash from_* convenience methods."""

    def test_from_weight_array_shape(self):
        wmh = WeightedMinHash(dim=5, num_perm=16)
        vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sig = wmh.from_weight_array(vec)
        assert sig.shape == (16, 2)
        assert sig.dtype == np.uint64

    def test_from_weight_array_consistency(self):
        """from_weight_array should match encode for equivalent input."""
        wmh = WeightedMinHash(dim=5, num_perm=32, seed=1)
        vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        sig1 = wmh.from_weight_array(vec)
        sig2 = wmh.encode(vec.reshape(1, -1))[0]

        np.testing.assert_array_equal(sig1, sig2)


# =============================================================================
# Integration Tests
# =============================================================================


class TestMinHashIntegration:
    """Integration tests for real-world usage patterns."""

    def test_molecular_fingerprint_workflow(self):
        """Simulate a molecular fingerprint comparison workflow."""
        mh = MinHash(num_perm=128, seed=42)

        # Simulate two molecules with similar fingerprints
        fp1 = np.zeros(1024, dtype=np.uint8)
        fp2 = np.zeros(1024, dtype=np.uint8)

        # Set some bits
        fp1[[10, 50, 100, 200, 500, 800]] = 1
        fp2[[10, 50, 100, 200, 501, 801]] = 1  # 4/6 overlap

        sig1 = mh.from_binary_array(fp1)
        sig2 = mh.from_binary_array(fp2)

        distance = mh.get_distance(sig1, sig2)

        # True Jaccard distance = 1 - 4/8 = 0.5
        assert 0.3 < distance < 0.7  # Reasonable range for estimate

    def test_batch_processing(self):
        """Test processing many samples at once."""
        mh = MinHash(num_perm=64)
        n_samples = 100
        n_features = 50

        # Generate random binary data
        data = (np.random.rand(n_samples, n_features) > 0.7).astype(np.uint8)

        sigs = mh.encode(data)

        assert sigs.shape == (n_samples, 64)
        assert sigs.dtype == np.uint64


class TestWeightedMinHashIntegration:
    """Integration tests for WeightedMinHash."""

    def test_count_vector_workflow(self):
        """Simulate comparing count vectors (e.g., word counts)."""
        wmh = WeightedMinHash(dim=100, num_perm=128, seed=42)

        # Simulate two documents with word counts
        doc1 = np.random.uniform(0.1, 10, size=100)
        doc2 = doc1.copy()
        doc2[:10] = np.random.uniform(0.1, 10, size=10)  # Modify first 10 words

        sig1 = wmh.from_weight_array(doc1)
        sig2 = wmh.from_weight_array(doc2)

        distance = wmh.get_weighted_distance(sig1, sig2)

        # Should have moderate distance since 90% is the same
        assert 0.0 < distance < 0.5


# =============================================================================
# Numba Backend Tests
# =============================================================================


class TestMinHashNumbaBackend:
    """Test MinHash with Numba JIT backend."""

    def test_numba_encode_shape(self):
        """Test that Numba backend produces correct output shape."""
        mh = MinHash(num_perm=64, seed=42)
        data = np.array([[1, 0, 1, 1, 0], [0, 1, 1, 0, 1]], dtype=np.uint8)
        sigs = mh.encode(data)

        assert sigs.shape == (2, 64)
        assert sigs.dtype == np.uint64

    def test_numba_determinism(self):
        """Test that Numba backend is deterministic with same seed."""
        mh1 = MinHash(num_perm=64, seed=42)
        mh2 = MinHash(num_perm=64, seed=42)

        data = np.array([[1, 0, 1, 1, 0, 0, 1, 0, 1, 1]], dtype=np.uint8)

        sig1 = mh1.encode(data)
        sig2 = mh2.encode(data)

        np.testing.assert_array_equal(sig1, sig2)

    def test_numba_distance_accuracy(self):
        """Test that Numba backend produces accurate Jaccard estimates."""
        mh = MinHash(num_perm=512, seed=42)

        # Create two fingerprints with known overlap
        # fp1: bits 0-99 on
        # fp2: bits 50-149 on
        # Intersection: 50-99 (50 bits), Union: 0-149 (150 bits)
        # True Jaccard = 50/150 = 0.333, Distance = 0.667
        fp1 = np.zeros(200, dtype=np.uint8)
        fp2 = np.zeros(200, dtype=np.uint8)
        fp1[:100] = 1
        fp2[50:150] = 1

        sig1 = mh.from_binary_array(fp1)
        sig2 = mh.from_binary_array(fp2)

        distance = mh.get_distance(sig1, sig2)
        true_distance = 1 - 50 / 150  # ~0.667

        # With 512 permutations, should be within 0.1 of true value
        assert abs(distance - true_distance) < 0.1

    def test_numba_sparse_consistency(self):
        """Test that sparse and dense paths produce consistent results."""
        mh = MinHash(num_perm=128, seed=42)

        # Create same data in sparse and dense format
        indices = [0, 5, 10, 15, 20, 100, 200]
        dense = np.zeros(300, dtype=np.uint8)
        dense[indices] = 1

        sig_sparse = mh.from_sparse_binary_array(indices)
        sig_dense = mh.from_binary_array(dense)

        np.testing.assert_array_equal(sig_sparse, sig_dense)

    def test_numba_batch_consistency(self):
        """Test that batch and single encoding produce same results."""
        mh = MinHash(num_perm=64, seed=42)

        # Single fingerprints
        fp1 = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        fp2 = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)

        sig1_single = mh.from_binary_array(fp1)
        sig2_single = mh.from_binary_array(fp2)

        # Batch
        batch = np.stack([fp1, fp2])
        sigs_batch = mh.batch_from_binary_array(batch)

        np.testing.assert_array_equal(sig1_single, sigs_batch[0])
        np.testing.assert_array_equal(sig2_single, sigs_batch[1])

    def test_numba_large_batch(self):
        """Test Numba backend with large batch (performance sanity check)."""
        mh = MinHash(num_perm=128, seed=42)

        # 10k fingerprints, 2048 bits each
        np.random.seed(42)
        data = (np.random.rand(10_000, 2048) < 0.1).astype(np.uint8)

        sigs = mh.batch_from_binary_array(data)

        assert sigs.shape == (10_000, 128)
        assert sigs.dtype == np.uint64

        # Sanity check: identical fingerprints should have distance 0
        assert mh.get_distance(sigs[0], sigs[0]) == 0.0

    def test_identical_inputs_zero_distance_numba(self):
        """Test that identical inputs produce zero distance with Numba."""
        mh = MinHash(num_perm=128, seed=42)

        fp = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1], dtype=np.uint8)
        sig = mh.from_binary_array(fp)

        assert mh.get_distance(sig, sig) == 0.0


# =============================================================================
# Additional Tests for Documentation Coverage
# =============================================================================


class TestMinHashInputValidation:
    """Test input validation and edge cases for MinHash methods.

    These tests ensure proper error handling and document expected behavior
    for edge cases and invalid inputs.
    """

    def test_from_binary_array_rejects_2d_input(self):
        """from_binary_array should reject 2D arrays.

        Covers: Input validation for from_binary_array.
        Users should use batch_from_binary_array for 2D input.
        """
        mh = MinHash(num_perm=32)
        arr_2d = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)

        with pytest.raises(ValueError, match="must be 1D"):
            mh.from_binary_array(arr_2d)

    def test_from_sparse_rejects_nested_sequences(self):
        """from_sparse_binary_array should reject nested sequences.

        Covers: Input validation preventing accidental nested input.
        """
        mh = MinHash(num_perm=32)
        nested = [[1, 2], [3, 4]]  # Nested - should be [1, 2, 3, 4]

        with pytest.raises(ValueError, match="nested sequence"):
            mh.from_sparse_binary_array(nested)

    def test_from_string_array_rejects_non_strings(self):
        """from_string_array should reject non-string elements.

        Covers: Input validation for string method.
        """
        mh = MinHash(num_perm=32)
        mixed = ["hello", 123, "world"]  # Contains int

        with pytest.raises(ValueError, match="must be strings"):
            mh.from_string_array(mixed)

    def test_from_string_array_rejects_nested(self):
        """from_string_array should reject nested sequences.

        Covers: Input validation for nested string lists.
        """
        mh = MinHash(num_perm=32)
        nested = [["hello"], ["world"]]

        with pytest.raises(ValueError, match="nested sequence"):
            mh.from_string_array(nested)

    def test_empty_sparse_indices(self):
        """Empty indices list should produce valid signature.

        Covers: Edge case of empty set.
        """
        mh = MinHash(num_perm=32, seed=42)
        sig = mh.from_sparse_binary_array([])
        assert sig.shape == (32,)
        assert sig.dtype == np.uint64

    def test_very_large_sparse_indices(self):
        """Large index values should be handled correctly.

        Covers: Edge case of very large feature indices (e.g., from large vocabularies).
        """
        mh = MinHash(num_perm=32, seed=42)
        large_indices = [0, 1000000, 999999999]  # Very large indices
        sig = mh.from_sparse_binary_array(large_indices)
        assert sig.shape == (32,)
        # Should be deterministic
        sig2 = mh.from_sparse_binary_array(large_indices)
        np.testing.assert_array_equal(sig, sig2)


class TestMinHashBinaryStringIncompatibility:
    """Test and document that binary and string signatures are NOT comparable.

    Binary encoding uses column indices as element IDs. String encoding uses
    xxhash64 outputs as element IDs. Different ID spaces, incomparable signatures.
    """

    def test_binary_vs_string_produce_different_signatures(self):
        """Binary and string encoding produce different signatures for same data.

        Binary uses column indices as element IDs, strings use xxhash64 outputs.
        Same conceptual data produces DIFFERENT signatures.
        """
        mh = MinHash(num_perm=64, seed=42)

        # Same data represented two ways
        indices = [1, 2, 3]

        # Binary representation
        binary = np.zeros(10, dtype=np.uint8)
        binary[indices] = 1
        sig_binary = mh.from_binary_array(binary)

        # String representation (same indices as strings)
        sig_string = mh.from_string_array([str(i) for i in indices])

        # These should NOT be equal (different hash functions)
        assert not np.array_equal(sig_binary, sig_string), (
            "Binary and string signatures should differ due to different hash functions"
        )

    def test_binary_string_distance_is_meaningless(self):
        """Distance between binary and string signatures is not meaningful.

        get_distance() won't error (same shape), but the result is meaningless.
        """
        mh = MinHash(num_perm=64, seed=42)

        # Create similar data in both formats
        sig_binary = mh.from_binary_array(np.array([1, 1, 1, 0, 0], dtype=np.uint8))
        sig_string = mh.from_string_array(["0", "1", "2"])  # "similar" - first 3 elements

        # Distance can be computed but is meaningless
        distance = mh.get_distance(sig_binary, sig_string)

        # Distance will be high (close to 1) because hashes are unrelated
        # This documents the expected behavior rather than asserting exact value
        assert 0.0 <= distance <= 1.0


class TestMinHashBatchMethods:
    """Test batch method behavior.

    These tests verify that batch methods work correctly.
    """

    def test_batch_from_binary_array_accepts_list_of_arrays(self):
        """batch_from_binary_array should accept sequence of 1D arrays.

        Covers: Flexibility in input format - can pass list of arrays.
        """
        mh = MinHash(num_perm=32)
        arrays = [
            np.array([1, 0, 1], dtype=np.uint8),
            np.array([0, 1, 1], dtype=np.uint8),
        ]
        sigs = mh.batch_from_binary_array(arrays)
        assert sigs.shape == (2, 32)

    def test_batch_from_sparse_with_varied_lengths(self):
        """batch_from_sparse_binary_array handles varied-length index lists.

        Covers: Ragged input where different items have different cardinalities.
        """
        mh = MinHash(num_perm=32, seed=42)
        indices_list = [
            [0],  # 1 element
            [0, 1, 2, 3, 4, 5],  # 6 elements
            [],  # 0 elements
            [100],  # 1 element, large index
        ]
        sigs = mh.batch_from_sparse_binary_array(indices_list)
        assert sigs.shape == (4, 32)


class TestWeightedMinHashEdgeCases:
    """Test edge cases for WeightedMinHash.

    WeightedMinHash has stricter requirements than regular MinHash.
    """

    def test_dimension_must_be_specified(self):
        """WeightedMinHash requires dim to be specified upfront.

        Covers: Unlike MinHash, dim cannot be inferred from data.
        """
        # dim is required - no default
        wmh = WeightedMinHash(dim=10, num_perm=32)
        assert wmh._dim == 10

    def test_all_weights_must_be_positive(self):
        """All weight values must be strictly > 0, not just >= 0.

        Covers: WeightedMinHash algorithm requires positive weights.
        Zero values will cause errors.
        """
        wmh = WeightedMinHash(dim=3, num_perm=32)

        # Zero is not allowed
        with pytest.raises(ValueError, match="positive"):
            wmh.encode(np.array([[1.0, 0.0, 1.0]]))

        # Small positive is OK
        sig = wmh.encode(np.array([[1.0, 0.001, 1.0]]))
        assert sig.shape == (1, 32, 2)

    def test_weighted_signature_has_two_components(self):
        """Weighted MinHash signatures have shape (num_perm, 2).

        Covers: Output format difference - weighted has 2 values per hash.
        """
        wmh = WeightedMinHash(dim=5, num_perm=16)
        sig = wmh.from_weight_array(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

        # Shape is (num_perm, 2) not just (num_perm,)
        assert sig.shape == (16, 2)

    def test_batch_weighted_output_shape(self):
        """Batch weighted encoding has 3D output.

        Covers: Output shape is (n_samples, num_perm, 2).
        """
        wmh = WeightedMinHash(dim=5, num_perm=16)
        data = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [5.0, 4.0, 3.0, 2.0, 1.0],
            ]
        )
        sigs = wmh.batch_from_weight_array(data)
        assert sigs.shape == (2, 16, 2)
