"""
TODO: ADd support for Annoy, FAISS etc.
Index module: Nearest-neighbor search data structures.
We define an abstract base class (Index) that specifies WHAT an index must do,
then concrete classes (FaissIndex, AnnoyIndex) that specify HOW.

This lets users swap implementations without changing their code:
    index = FaissIndex(...)   # or AnnoyIndex(...), or custom
    index.build(data)
    neighbors = index.query(point, k=10)
"""

from tmap.index.base import Index
from tmap.index.lsh_forest import LSHForest
from tmap.index.types import EdgeList, KNNGraph

__all__ = ["Index", "KNNGraph", "EdgeList", "LSHForest"]

# Concrete implementations
# from tmap.index.faiss_index import FaissIndex
# from tmap.index.annoy_index import AnnoyIndex
