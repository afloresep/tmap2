"""
Index module: Nearest-neighbor search data structures.

Backends:
  - LSHForest: MinHash-based locality-sensitive hashing (Jaccard metric)
  - FaissIndex: FAISS flat/HNSW (cosine/euclidean)
"""

from tmap.index.lsh_forest import LSHForest
from tmap.index.types import KNNGraph

__all__ = ["KNNGraph", "LSHForest", "FaissIndex"]

try:
    from tmap.index.faiss_index import FaissIndex
except ImportError:
    pass
