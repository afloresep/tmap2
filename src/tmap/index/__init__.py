"""
Index module: Nearest-neighbor search data structures.

Backends:
  - LSHForest: MinHash-based locality-sensitive hashing (Jaccard metric)
  - FaissIndex: FAISS flat/HNSW (cosine/euclidean)

Usage:
    index = FaissIndex(seed=42)
    index.build_from_vectors(data, metric="cosine")
    knn = index.query_knn(k=10)
"""

from tmap.index.base import Index
from tmap.index.lsh_forest import LSHForest
from tmap.index.types import EdgeList, KNNGraph

__all__ = ["Index", "KNNGraph", "EdgeList", "LSHForest", "FaissIndex"]

try:
    from tmap.index.faiss_index import FaissIndex
except ImportError:
    pass
