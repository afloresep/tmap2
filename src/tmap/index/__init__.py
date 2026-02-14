"""
Index module: Nearest-neighbor search data structures.

DESIGN PATTERN: Strategy Pattern
--------------------------------
We define an abstract base class (Index) that specifies WHAT an index must do,
then concrete classes (FaissIndex, AnnoyIndex) that specify HOW.

This lets users swap implementations without changing their code:
    index = FaissIndex(...)   # or AnnoyIndex(...), or custom
    index.build(data)
    neighbors = index.query(point, k=10)

WHY Abstract Base Classes (ABC)?
--------------------------------
Python has two ways to define interfaces:

1. ABC (what we use here):
   - Enforces that subclasses implement required methods
   - Raises TypeError at instantiation time if methods missing
   - Use when: You have shared implementation (methods all subclasses use)

2. Protocol (typing.Protocol):
   - "Structural subtyping" - any class with matching methods works
   - No inheritance required
   - Use when: You want duck typing with type checking

We use ABC here because:
- We have shared methods (like save/load boilerplate)
- We want explicit "this IS an Index" relationship
- Subclasses benefit from shared validation logic
"""

from tmap.index.base import Index
from tmap.index.lsh_forest import LSHForest
from tmap.index.types import EdgeList, KNNGraph

__all__ = ["Index", "KNNGraph", "EdgeList", "LSHForest"]

# Concrete implementations - uncomment as you build them:
# from tmap.index.faiss_index import FaissIndex
# from tmap.index.annoy_index import AnnoyIndex
