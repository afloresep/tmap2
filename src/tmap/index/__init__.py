"""
Index module: Nearest-neighbor search data structures.

Defines an abstract base class (Index) and concrete backends for kNN search:
  - LSHForest: MinHash-based locality-sensitive hashing (Jaccard metric)
  - NNDescentIndex: PyNNDescent approximate NN (cosine/euclidean, pure Python)
  - FaissIndex: FAISS exact/approximate NN (cosine/euclidean, optional GPU)

Usage:
    index = FaissIndex(seed=42)       # or NNDescentIndex(...)
    index.build_from_vectors(data, metric="cosine")
    knn = index.query_knn(k=10)
"""

from tmap.index.base import Index
from tmap.index.lsh_forest import LSHForest
from tmap.index.types import EdgeList, KNNGraph

__all__ = ["Index", "KNNGraph", "EdgeList", "LSHForest", "NNDescentIndex", "FaissIndex"]

# Optional backends — imported lazily to avoid hard dependency on pynndescent/faiss.
try:
    from tmap.index.nndescent import NNDescentIndex
except ImportError:
    pass

try:
    from tmap.index.faiss_index import FaissIndex
except ImportError:
    pass
