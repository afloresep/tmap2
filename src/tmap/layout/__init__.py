"""
Layout module: Compute 2D coordinates for tree visualization.

Quick Start (recommended - matches original TMAP API)
-----------------------------------------------------
>>> from tmap import MinHash, LSHForest
>>> from tmap.layout import layout_from_lsh_forest, LayoutConfig
>>>
>>> # Build LSHForest
>>> mh = MinHash(num_perm=128)
>>> sigs = mh.batch_from_binary_array(fingerprints)
>>> lsh = LSHForest(d=128)
>>> lsh.batch_add(sigs)
>>> lsh.index()
>>>
>>> # Compute layout (original TMAP-style)
>>> cfg = LayoutConfig()
>>> cfg.k = 20
>>> cfg.kc = 50
>>> x, y, s, t = layout_from_lsh_forest(lsh, cfg)

Alternative: Step-by-step with ForceDirectedLayout
--------------------------------------------------
>>> from tmap.layout import ForceDirectedLayout
>>> from tmap.graph.mst import MSTBuilder
>>>
>>> tree = MSTBuilder().build(knn_graph)
>>> layout = ForceDirectedLayout(seed=42)
>>> coords = layout.compute(tree)

Low-level: Direct edge list
---------------------------
>>> from tmap.layout import layout_from_edge_list
>>> x, y, s, t = layout_from_edge_list(n_nodes, edges, create_mst=True)
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
        layout_from_lsh_forest,
        layout_from_knn_graph,
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
        "layout_from_lsh_forest",
        "layout_from_knn_graph",
    ]
