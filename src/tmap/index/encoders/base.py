from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray


class Encoder(ABC):
    """Base class for MinHash-style encoders."""

    def __init__(self, num_perm: int = 128, seed: int = 1) -> None:
        """Initialize encoder configuration."""
        self.num_perm = num_perm
        self.seed = seed

    @property
    def n_hashes(self) -> int:
        """Return the number of hashes produced by the encoder."""
        return self.num_perm

    @abstractmethod
    def encode(self, data: Any) -> NDArray[np.uint64]:
        """Encode input data into hash signatures."""
        ...

    def _validate_non_negative_vector(self, data: Any) -> None:
        """Validate that vector values are strictly positive."""
        arr = np.asarray(data)
        if not np.all(arr > 0):
            raise ValueError("Vector must contain only positive values")
