import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from tmap.index.types import EdgeList, KNNGraph


class Index(ABC):
    """
    Abstract base class for nearest-neighbor search.
    """
    def __init__(self, seed: int | None = None) -> None:
        """
        Initialize the index.

        Args:
            seed: Random seed for reproducibility. If None, non-deterministic.
        """
        self._seed = seed
        self._is_built = False
        self._n_nodes: int = 0
        self._rng = np.random.default_rng(seed)

    # =========================================================================
    # PUBLIC API - These are what users call
    # =========================================================================

    def build_from_vectors(
        self,
        vectors: NDArray[np.float32],
        metric: str = "euclidean",
    ) -> Self:
        """
        Build index from raw vectors (embeddings, fingerprints, etc.).

        Args:
            vectors: Shape (n_samples, n_features)
            metric: Distance metric ("euclidean", "cosine", "jaccard")

        Returns:
            self (for method chaining)

        Example:
            index = FaissIndex(seed=42)
            index.build_from_vectors(embeddings, metric="cosine")
            knn = index.query_knn(k=10)
        """
        self._validate_vectors(vectors)
        self._n_nodes = vectors.shape[0]
        self._build_from_vectors(vectors, metric)
        self._is_built = True
        return self

    def build_from_edges(self, edges: EdgeList) -> Self:
        """
        Build index from pre-computed edges/distances.

        Args:
            edges: EdgeList with (source, target, distance) tuples

        Returns:
            self (for method chaining)

        Example:
            # Visualize arxiv papers by citation similarity
            edges = EdgeList(
                edges=[(0, 1, 0.9), (0, 2, 0.7), ...],
                n_nodes=1000,
            )
            index = AnnoyIndex(seed=42)
            index.build_from_edges(edges)
        """
        self._n_nodes = edges.n_nodes
        self._build_from_edges(edges)
        self._is_built = True
        return self

    def query_knn(self, k: int) -> KNNGraph:
        """
        Get k-nearest neighbors for ALL points in the index.

        Args:
            k: Number of neighbors per point

        Returns:
            KNNGraph with indices and distances arrays
        """
        self._check_is_built()
        if k >= self._n_nodes:
            raise ValueError(f"k={k} must be < n_nodes={self._n_nodes}")
        return self._query_all(k)

    def query_point(
        self,
        point: NDArray[np.float32],
        k: int,
    ) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        """
        Query k-nearest neighbors for a single NEW point.

        This is for INSERTION - finding where a new point belongs.

        Args:
            point: Shape (n_features,)
            k: Number of neighbors

        Returns:
            (neighbor_indices, distances)
        """
        self._check_is_built()
        return self._query_single(point, k)

    def save(self, path: str | Path) -> None:
        """
        Save index to disk.
        """
        self._check_is_built()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save metadata that all indices need
        metadata = {
            "class": self.__class__.__name__,
            "n_nodes": self._n_nodes,
            "seed": self._seed,
        }
        with open(path.with_suffix(".meta"), "wb") as f:
            pickle.dump(metadata, f)

        # Let subclass save the actual index data
        self._save_implementation(path)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """
        Load index from disk.

        Usage:
            index = FaissIndex.load("my_index.faiss")
        """
        path = Path(path)
        with open(path.with_suffix(".meta"), "rb") as f:
            metadata = pickle.load(f)

        instance = cls(seed=metadata["seed"])
        instance._n_nodes = metadata["n_nodes"]
        instance._load_implementation(path)
        instance._is_built = True
        return instance

    @property
    def is_built(self) -> bool:
        """Whether the index has been built and is ready for queries."""
        return self._is_built

    @property
    def n_nodes(self) -> int:
        """Number of nodes/points in the index."""
        return self._n_nodes

    @abstractmethod
    def _build_from_vectors(
        self,
        vectors: NDArray[np.float32],
        metric: str,
    ) -> None:
        """Implementation-specific vector indexing. Override this."""
        ...

    @abstractmethod
    def _build_from_edges(self, edges: EdgeList) -> None:
        """Implementation-specific edge-based building. Override this."""
        ...

    @abstractmethod
    def _query_all(self, k: int) -> KNNGraph:
        """Query k-NN for all points. Override this."""
        ...

    @abstractmethod
    def _query_single(
        self,
        point: NDArray[np.float32],
        k: int,
    ) -> tuple[NDArray[np.int32], NDArray[np.float32]]:
        """Query k-NN for single point. Override this."""
        ...

    @abstractmethod
    def _save_implementation(self, path: Path) -> None:
        """Save index-specific data. Override this."""
        ...

    @abstractmethod
    def _load_implementation(self, path: Path) -> None:
        """Load index-specific data. Override this."""
        ...

    def _validate_vectors(self, vectors: NDArray[Any]) -> None:
        """Validate input vectors. Called before building."""
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2D, got shape {vectors.shape}")
        if vectors.shape[0] < 2:
            raise ValueError("Need at least 2 vectors to build index")

    def _check_is_built(self) -> None:
        """Raise if index not built. Fail fast pattern."""
        if not self._is_built:
            raise RuntimeError(
                "Index not built. Call build_from_vectors() or build_from_edges() first."
            )
