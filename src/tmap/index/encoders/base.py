from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Encoder(ABC):
    """Abstract base class for MinHash-style encoders."""
    def __init__(self, num_perm: int = 128, seed: int = 1) -> None:
        self.num_perm = num_perm
        self.seed = seed

    @property
    def n_hashes(self):
        """Return the number of hashes produced by the encoder."""
        return self.num_perm

    @abstractmethod
    def encode(self, data, num_perm: int= 128) -> NDArray[np.uint64]: # returns hash signatures
        """Encode input data into hash signatures."""
        ...

    # Helper functions
    def _validate_non_negative_vector(self, data):
        arr = np.asarray(data)
        if not np.all(arr > 0):
            raise ValueError("Vector must contain only non-negative values")

    def _validate_binary_vector(self,data):
        if not all(i in [0,1] for i in data):
            raise ValueError("Vector must be binary")



