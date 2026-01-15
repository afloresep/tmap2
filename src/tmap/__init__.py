"""
TreeMap: Tree-based visualization for high-dimensional data.

This package is generic - it works with ANY data that can be represented
as edges/distances between points. Molecules, papers, images, embeddings, etc.

Architecture Overview
---------------------
The pipeline has 4 stages, each with a clear interface:

1. Index: Build a data structure for fast nearest-neighbor queries
   - Input: Distance matrix OR edge list OR raw vectors
   - Output: k-NN graph (sparse adjacency with distances)

2. Graph: Convert k-NN graph to a tree structure
   - Input: k-NN graph
   - Output: MST (minimum spanning tree)

3. Layout: Compute 2D coordinates for the tree
   - Input: MST
   - Output: (x, y) coordinates per node

4. Visualization: Render the layout
   - Input: Coordinates + metadata
   - Output: HTML/PNG/interactive plot

Design Principles
-----------------
- Each stage is independent and swappable (Strategy pattern)
- Base classes define the interface; you implement concrete strategies
- Composition over inheritance for the main TreeMap class
- Type hints everywhere for IDE support and self-documentation
"""

__version__ = "0.1.0"

# These will be your public API exports
# Uncomment as you implement them:

# from tmap.core import TreeMap
# from tmap.index import Index, FaissIndex, AnnoyIndex
# from tmap.graph import GraphBuilder, MSTBuilder
# from tmap.layout import Layout, ForceDirectedLayout
# from tmap.visualization import Visualizer, HTMLVisualizer

__all__ = [
    "__version__",
    # "TreeMap",
    # "Index", "FaissIndex", "AnnoyIndex",
    # "GraphBuilder", "MSTBuilder",
    # "Layout", "ForceDirectedLayout",
    # "Visualizer", "HTMLVisualizer",
]
