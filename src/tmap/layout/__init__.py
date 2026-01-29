"""
Layout module: Compute 2D coordinates for tree visualization.

Quick Start
-----------
>>> from tmap.layout import ForceDirectedLayout
>>> from tmap.graph.mst import MSTBuilder
>>>
>>> tree = MSTBuilder().build(knn_graph)
>>> layout = ForceDirectedLayout(seed=42)
>>> coords = layout.compute(tree)

Or use the convenience function (original TMAP API):

>>> from tmap.layout import layout_from_edge_list
>>> x, y, s, t = layout_from_edge_list(n_nodes, edges)
"""

# Always available (pure Python)
from tmap.layout.base import Layout
from tmap.layout.types import Coordinates

# OGDF availability (handles editable install quirks internally)
from tmap.layout._ogdf import _AVAILABLE as OGDF_AVAILABLE

__all__ = [
    "Layout",
    "Coordinates",
    "OGDF_AVAILABLE",
]

# Conditionally export OGDF-dependent items
if OGDF_AVAILABLE:
    from tmap.layout._ogdf import (
        LayoutConfig,
        Placer,
        Merger,
        ScalingType,
        layout_from_edge_list,
        layout_from_tree,
    )
    from tmap.layout.force_directed import ForceDirectedLayout

    __all__ += [
        "ForceDirectedLayout",
        "LayoutConfig",
        "Placer",
        "Merger",
        "ScalingType",
        "layout_from_edge_list",
        "layout_from_tree",
    ]
