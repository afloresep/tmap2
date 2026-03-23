"""
Index module: Nearest-neighbor search data structures.

Backends:
  - LSHForest: MinHash-based locality-sensitive hashing (Jaccard metric)
  - USearchIndex: USearch exact/HNSW (cosine/euclidean)
"""

from tmap.index.lsh_forest import LSHForest
from tmap.index.types import KNNGraph
from tmap.index.usearch_index import USearchIndex

__all__ = ["KNNGraph", "LSHForest", "USearchIndex"]
