from collections.abc import Collection, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import datasketch.minhash as _datasketch_minhash
import numpy as np
from datasketch.weighted_minhash import WeightedMinHashGenerator as _WeightedMinHashGenerator
from numpy.typing import NDArray

from .base import Encoder


# Module-level helper functions for parallel processing (must be picklable)
def _encode_single_set(args: tuple[set, int, int]) -> NDArray[np.uint64]:
    """Encode a single set into a MinHash signature."""
    s, num_perm, seed = args
    mh = _datasketch_minhash.MinHash(num_perm=num_perm, seed=seed)
    for item in s:
        mh.update(str(item).encode("utf-8"))
    return mh.hashvalues


def _encode_weighted_chunk(
    args: tuple[NDArray, int, int, int],
) -> NDArray[np.uint64]:
    """Encode a chunk of weighted vectors (creates its own generator)."""
    data, dim, num_perm, seed = args
    generator = _WeightedMinHashGenerator(dim=dim, sample_size=num_perm, seed=seed)
    n_samples = data.shape[0]
    signatures = np.zeros((n_samples, num_perm, 2), dtype=np.uint64)
    for i in range(n_samples):
        signatures[i] = generator.minhash(data[i]).hashvalues
    return signatures

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

        if not all(isinstance(x, str) for x in strings):
            raise ValueError("All elements must be strings")
        return self.encode([set(strings)])[0]

    # -------------------------------------------------------------------------
    # Batch methods (parallelized)
    # -------------------------------------------------------------------------

    def batch_from_binary_array(
        self,
        arrays: Sequence[NDArray[np.uint8]] | NDArray[np.uint8],
        n_jobs: int | None = None,
    ) -> NDArray[np.uint64]:
        """
        Create MinHash signatures from multiple binary vectors (parallelized).

        Args:
            arrays: Either a 2D array of shape (n_samples, n_features) or
                    a sequence of 1D binary arrays
            n_jobs: Number of parallel workers. None = number of CPUs.

        Returns:
            2D array of shape (n_samples, num_perm) containing MinHash signatures
        """
        if isinstance(arrays, np.ndarray) and arrays.ndim == 2:
            # Already a 2D array, use encode directly with parallelization
            return self._parallel_encode_sets(
                [set(np.nonzero(row)[0].tolist()) for row in arrays],
                n_jobs=n_jobs,
            )
        else:
            # Sequence of 1D arrays
            sets = [set(np.nonzero(arr)[0].tolist()) for arr in arrays]
            return self._parallel_encode_sets(sets, n_jobs=n_jobs)

    def batch_from_sparse_binary_array(
        self,
        indices_list: Sequence[Sequence[int]],
        n_jobs: int | None = None,
    ) -> NDArray[np.uint64]:
        """
        Create MinHash signatures from multiple sparse representations (parallelized).

        Args:
            indices_list: Sequence of sequences, where each inner sequence contains
                          the indices of 1s in a sparse binary vector
            n_jobs: Number of parallel workers. None = number of CPUs.

        Returns:
            2D array of shape (n_samples, num_perm) containing MinHash signatures
        """
        sets = [set(indices) for indices in indices_list]
        return self._parallel_encode_sets(sets, n_jobs=n_jobs)

    def batch_from_string_array(
        self,
        string_lists: Sequence[Sequence[str]],
        n_jobs: int | None = None,
    ) -> NDArray[np.uint64]:
        """
        Create MinHash signatures from multiple string lists (parallelized).

        Args:
            string_lists: Sequence of string sequences, where each inner sequence
                          contains strings to be treated as set elements
            n_jobs: Number of parallel workers. None = number of CPUs.

        Returns:
            2D array of shape (n_samples, num_perm) containing MinHash signatures
        """
        sets = [set(strings) for strings in string_lists]
        return self._parallel_encode_sets(sets, n_jobs=n_jobs)

    def _parallel_encode_sets(
        self,
        sets: list[set],
        n_jobs: int | None = None,
    ) -> NDArray[np.uint64]:
        """
        Internal method to encode sets in parallel using ProcessPoolExecutor.

        Args:
            sets: List of sets to encode
            n_jobs: Number of parallel workers. None = number of CPUs.

        Returns:
            2D array of shape (n_samples, num_perm) containing MinHash signatures
        """
        if n_jobs is None:
            n_jobs = os.cpu_count() or 1

        n_samples = len(sets)

        # For small inputs, sequential is faster due to process spawn overhead
        if n_samples < 100 or n_jobs == 1:
            return self.encode(sets)

        # Prepare arguments for parallel processing
        args = [(s, self._num_perm, self._seed) for s in sets]

        signatures = np.zeros((n_samples, self._num_perm), dtype=np.uint64)

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all tasks and track their original indices
            future_to_idx = {
                executor.submit(_encode_single_set, arg): i for i, arg in enumerate(args)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                signatures[idx] = future.result()

        return signatures


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

    def batch_from_weight_array(
        self,
        vectors: Sequence[NDArray[np.floating]] | NDArray[np.floating],
        n_jobs: int | None = None,
    ) -> NDArray[np.uint64]:
        """
        Create weighted MinHash signatures from multiple weight vectors (parallelized).

        Args:
            vectors: Either a 2D array of shape (n_samples, dim) or
                     a sequence of 1D weight arrays
            n_jobs: Number of parallel workers. None = number of CPUs.
                    Note: Currently parallelization is limited because the
                    WeightedMinHashGenerator is not easily serializable.

        Returns:
            3D array of shape (n_samples, num_perm, 2) containing weighted MinHash signatures
        """
        if isinstance(vectors, np.ndarray) and vectors.ndim == 2:
            data = vectors
        else:
            data = np.stack(vectors)

        n_samples = data.shape[0]

        # For large datasets with n_jobs > 1, use chunk-based parallelism
        # Each worker gets its own generator to avoid serialization issues
        if n_jobs is None:
            n_jobs = os.cpu_count() or 1

        if n_samples < 100 or n_jobs == 1:
            return self.encode(data.astype(np.float32))

        # Chunk the data for parallel processing
        chunk_size = max(1, n_samples // n_jobs)
        chunks = [
            (data[i : i + chunk_size], self._dim, self._num_perm, self._seed)
            for i in range(0, n_samples, chunk_size)
        ]

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(_encode_weighted_chunk, chunks))

        return np.concatenate(results, axis=0)

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
