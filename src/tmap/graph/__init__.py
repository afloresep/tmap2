"""
Graph module: Convert k-NN graph to tree structure.

The graph module takes a KNNGraph and builds a Tree (MST).

    knn_graph = index.query_knn(k=20)
    tree = MSTBuilder().build(knn_graph)

MST BIAS FEATURE
----------------
TODO: "bias MST toward close NNs ab initio".
This is implemented via edge weight modification before MST construction.
MSTBuilder.build() for the bias_factor parameter.
But it is slow and the end result is not really good. This is going to be tricky
to implement...
"""

from tmap.graph.mst import MSTBuilder
from tmap.graph.types import Tree

__all__ = ["Tree", "MSTBuilder"]
