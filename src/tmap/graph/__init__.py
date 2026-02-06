"""
Graph module: Convert k-NN graph to tree structure.

DESIGN PATTERN: Builder Pattern (simplified)
--------------------------------------------
The graph module takes a KNNGraph and builds a Tree (MST).

    knn_graph = index.query_knn(k=20)
    tree = MSTBuilder().build(knn_graph)

WHY A SEPARATE MODULE?
---------------------
The MST algorithm is independent of:
- How the k-NN was computed (FAISS, Annoy, precomputed edges)
- How the tree will be laid out (force-directed, radial, etc.)

Keeping it separate means:
- Easy to test in isolation
- Easy to swap algorithms (Kruskal vs Prim vs Boruvka)
- Clear data flow: Index -> Graph -> Layout -> Viz

MST BIAS FEATURE
----------------
One of your requirements was "bias MST toward close NNs ab initio".
This is implemented via edge weight modification before MST construction.
See MSTBuilder.build() for the bias_factor parameter.
"""

from tmap.graph.mst import MSTBuilder
from tmap.graph.types import Tree

__all__ = ["Tree", "MSTBuilder"]
