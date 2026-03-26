"""
Index module: Nearest-neighbor search data structures.

Backends:
  - USearchIndex: USearch exact/HNSW (cosine, euclidean, Jaccard on binary data)
  - LSHForest: MinHash-based locality-sensitive hashing (Jaccard on sets/strings)
"""

from tmap.index.lsh_forest import LSHForest
from tmap.index.types import KNNGraph
from tmap.index.usearch_index import USearchIndex

__all__ = ["KNNGraph", "LSHForest", "USearchIndex"]
