from collections.abc import Collection, Sequence
import datasketch.minhash as _datasketch_minhash
import numpy as np
from datasketch.weighted_minhash import WeightedMinHashGenerator as _WeightedMinHashGenerator
from numpy.typing import NDArray

from .base import Encoder

__all__ = [
    "MinHash",
    "WeightedMinHash",
]


# for binary/set data (e.g molecular fingerprints)
class MinHash(Encoder):
    def __init__(self, num_perm: int = 128, seed: int = 1):
        self._num_perm = num_perm
        self._seed = seed

    def encode(
        self, data: NDArray[np.uint8] | Sequence[Collection[int | str]]
    ) -> NDArray[np.uint64]:
        """
        Encode data into MinHash signatures.

        Args:
            data: EITHER
                - (n_samples, n_features) binary NumPy array (0/1 values)
                - sequence of collections (e.g., sets/lists of ints or strings)

        Returns:
            signatures: (n_samples, num_perm) uint64 array
        """
        if isinstance(data, np.ndarray):
            sets = [set(np.nonzero(row)[0].tolist()) for row in data]
        elif isinstance(data, list) and all(isinstance(s, set) for s in data):
            sets = data
        else:
            sets = [set(s) for s in data]

        signatures = np.zeros((len(sets), self._num_perm), dtype=np.uint64)

        for i, s in enumerate(sets):
            mh = _datasketch_minhash.MinHash(num_perm=self._num_perm, seed=self._seed)
            for item in s:
                mh.update(str(item).encode("utf-8"))
            signatures[i] = mh.hashvalues

        return signatures

    @staticmethod
    def get_distance(vec_a: NDArray[np.uint64], vec_b: NDArray[np.uint64]) -> float:
        """
        Calculate the Jaccard distance between two MinHash vectors.

        Jaccard distance = 1 - (number of matching hash values) / num_perm

        Args:
            vec_a: A MinHash vector
            vec_b: A MinHash vector

        Returns:
            The estimated Jaccard distance (0.0 to 1.0)
        """
        if len(vec_a) != len(vec_b):
            raise ValueError(
                "Cannot compute Jaccard distance: MinHash vectors have different "
                f"numbers of permutations ({len(vec_a)} vs {len(vec_b)})"
            )
        return float(1 - np.mean(vec_a == vec_b))

    # Alias for backward compatibility
    jaccard_distance = get_distance

    def _binary_to_sets(self, vectors: NDArray) -> list[set[int]]:
        """convert binary vectors to sets of 'on' indices."""
        return [set(np.where(row)[0]) for row in vectors]

    def from_binary_array(self, arr: NDArray[np.uint8] | list) -> NDArray[np.uint64]:
        """
        Create a MinHash signature from a single binary vector.

        Args:
            arr: 1D binary array (0/1 values) where 1s indicate set membership

        Returns:
            1D array of shape (num_perm,) containing the MinHash signature
        """
        if isinstance(arr, list): arr = np.array(arr) # Convert to np.array in case a list is provided  
        if len(arr.shape) > 1: 
            raise ValueError("vector must be 1D")
        return self.encode(arr.reshape(1, -1))[0]

    def from_sparse_binary_array(self, indices: Sequence[int]) -> NDArray[np.uint64]:
        """Create a MinHash signature from sparse representation (list of indices).

        Args:
            indices: 1D sequence of integers representing positions of 1s

        Returns:
            1D array of shape (num_perm,) containing the MinHash signature
        """
        # reject nested sequences/arrays like [[1, 2], [3]]
        if isinstance(indices, np.ndarray):
            if indices.ndim != 1:
                raise ValueError(f"indices must be 1D, got array with shape {indices.shape}")
        else:
            if any(isinstance(x, (list, tuple, set, dict, np.ndarray)) for x in indices):
                raise ValueError("indices must be a 1D sequence of ints, not a nested sequence")

        return self.encode([set(indices)])[0]

    def from_string_array(self, strings: Sequence[str]) -> NDArray[np.uint64]:
        """Create a MinHash signature from a list of strings.
        Args:
            strings: 1D sequence of strings to be treated as set elements
        Returns:
            1D array of shape (num_perm,) containing the MinHash signature
        """
        # reject nested sequences/arrays like [["a"], ["b"]]
        if isinstance(strings, np.ndarray):
            if strings.ndim != 1:
                raise ValueError(f"strings must be 1D, got array with shape {strings.shape}")
        else:
            if any(isinstance(x, (list, tuple, set, dict, np.ndarray)) for x in strings):
                raise ValueError("strings must be a 1D sequence of str, not a nested sequence")

        assert all(isinstance(x, float) for x in strings) 
        return self.encode([set(strings)])[0]


        #TODO: implement batch methods


# for integer/float data
class WeightedMinHash(Encoder): 
    def __init__(self, dim: int, num_perm: int = 128, seed: int = 1):
        """
        Weighted MinHash for float/integer vectors.

        Uses consistent weighted sampling to create MinHash signatures that
        estimate weighted Jaccard similarity.

        Args:
            dim: Number of features (must know upfront for generator)
            num_perm: Number of hash permutations
            seed: Random seed for reproducibility

        Note:
            The original TMAP supported ICWS and I2CWS methods. This implementation
            uses datasketch's consistent weighted sampling which is similar to ICWS.
        """
        self._num_perm = num_perm
        self._seed = seed
        self._dim = dim

        # generator must be created with known dimension
        self._generator = _WeightedMinHashGenerator(dim=dim, sample_size=num_perm, seed=seed)

    def encode(self, data: NDArray[np.float32]) -> NDArray[np.uint64]:
        """
        Encode weighted vectors into MinHash signatures.

        Args:
            data: (n_samples, n_features) array of non-negative weights

        Returns:
            signatures: (n_samples, num_perm, 2) uint64 array
                Each signature has 2 values per hash (k, y_k from the algorithm)
        """
        n_samples, n_features = data.shape
        self._validate_non_negative_vector(data)
        if n_features != self._dim:
            raise ValueError(f"Expected {self._dim} features, got {n_features}")

        signatures = np.zeros(shape=(n_samples, self._num_perm, 2), dtype=np.uint64)

        for i in range(n_samples):
            signatures[i] = self._generator.minhash(data[i]).hashvalues

        return signatures

    def from_weight_array(self, vec: NDArray[np.floating]) -> NDArray[np.uint64]:
        """
        Create a weighted MinHash vector from a float array.

        Args:
            vec: 1D array of non-negative float values

        Returns:
            2D array of shape (num_perm, 2) containing the weighted MinHash signature

        Note:
            The original TMAP supported 'method' parameter for ICWS or I2CWS.
            This implementation uses datasketch's consistent weighted sampling.
        """
        return self.encode(vec.reshape(1, -1))[0]

    @staticmethod
    def get_weighted_distance(vec_a: NDArray[np.uint64], vec_b: NDArray[np.uint64]) -> float:
        """
        Calculate the weighted Jaccard distance between two weighted MinHash vectors.

        For weighted MinHash, two hashes match only if both components (k, y_k) match.
        Distance = 1 - (number of matching rows) / num_perm

        Args:
            vec_a: A weighted MinHash vector of shape (num_perm, 2)
            vec_b: A weighted MinHash vector of shape (num_perm, 2)

        Returns:
            The estimated weighted Jaccard distance (0.0 to 1.0)
        """
        if vec_a.shape != vec_b.shape:
            raise ValueError(f"Shape mismatch: vec_a {vec_a.shape} vs vec_b {vec_b.shape}")
        if vec_a.ndim != 2 or vec_a.shape[1] != 2:
            raise ValueError(f"Expected shape (num_perm, 2), got {vec_a.shape}")
        # Both columns must match for a row to count as matching
        matches = np.all(vec_a == vec_b, axis=1)
        return float(1 - np.mean(matches))
