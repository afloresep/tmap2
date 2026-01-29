"""
Force-directed layout using OGDF.

This implements the Layout ABC using OGDF's FastMultipoleEmbedder
with ModularMultilevelMixer, matching the original TMAP algorithm.
"""

from __future__ import annotations

import numpy as np

from tmap.graph.types import Tree
from tmap.layout.base import Layout
from tmap.layout.types import Coordinates
from tmap.layout._ogdf import (
    require_ogdf,
    layout_from_tree,
    LayoutConfig,
    Placer,
    Merger,
    ScalingType,
)


class ForceDirectedLayout(Layout):
    """
    Force-directed layout using OGDF's FastMultipoleEmbedder.

    This is the recommended layout for TMAP-style visualizations.
    It produces organic layouts that respect edge weights as distances.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility. When set, enables deterministic mode.
    max_iterations : int, default 1000
        Maximum iterations for the force simulation.
    placer : Placer, default Placer.Barycenter
        Initial placement strategy during multilevel uncoarsening.
    merger : Merger, default Merger.LocalBiconnected
        Graph coarsening strategy for multilevel algorithm.

    Example
    -------
    >>> from tmap.layout import ForceDirectedLayout
    >>> from tmap.graph.mst import MSTBuilder
    >>>
    >>> # Build MST from k-NN graph
    >>> tree = MSTBuilder().build(knn_graph)
    >>>
    >>> # Compute layout
    >>> layout = ForceDirectedLayout(seed=42)
    >>> coords = layout.compute(tree)
    >>>
    >>> # Use coordinates
    >>> print(coords.x, coords.y)
    """

    def __init__(
        self,
        seed: int | None = None,
        max_iterations: int = 1000,
        placer: Placer | None = None,
        merger: Merger | None = None,
    ) -> None:
        # Check OGDF availability early
        require_ogdf()

        super().__init__(seed=seed, max_iterations=max_iterations)

        # Store OGDF-specific config
        self._placer = placer if placer is not None else Placer.Barycenter
        self._merger = merger if merger is not None else Merger.LocalBiconnected

    def _make_config(self) -> LayoutConfig:
        """Create OGDF LayoutConfig from our parameters."""
        config = LayoutConfig()
        config.fme_iterations = self.max_iterations
        config.placer = self._placer
        config.merger = self._merger

        # Determinism: if seed is set, enable deterministic mode
        if self._seed is not None:
            config.deterministic = True
            config.seed = self._seed
        else:
            config.deterministic = False

        return config

    def _compute_initial(self, tree: Tree) -> Coordinates:
        """
        Compute layout from scratch using OGDF.

        This is the main layout computation. It:
        1. Converts Tree to edge list
        2. Calls OGDF FastMultipoleEmbedder via C++ extension
        3. Returns normalized coordinates
        """
        config = self._make_config()
        x, y = layout_from_tree(tree, config)

        return Coordinates(x=x, y=y)

    def _compute_incremental(
        self,
        tree: Tree,
        existing: Coordinates,
        new_nodes: list[int],
    ) -> Coordinates:
        """
        Update layout for new nodes.

        For now, we just recompute the full layout.
        A smarter implementation could:
        1. Fix existing node positions
        2. Only optimize new nodes
        3. Run fewer iterations

        This is a TODO for future optimization when adding new nodes
        """
        # Simple approach: recompute everything
        # The existing layout provides a good starting point conceptually,
        # but OGDF doesn't support warm-starting easily.
        return self._compute_initial(tree)


# Re-export config types for convenience
__all__ = [
    "ForceDirectedLayout",
    "Placer",
    "Merger",
    "ScalingType",
    "LayoutConfig",
]
