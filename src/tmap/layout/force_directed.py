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
        Maximum iterations for the force simulation (fme_iterations).
    placer : Placer, default Placer.Barycenter
        Initial placement strategy during multilevel uncoarsening.
    merger : Merger, default Merger.LocalBiconnected
        Graph coarsening strategy for multilevel algorithm.
    node_size : float, default 1/65
        Node size for repulsion calculation. Larger values = more spread out.
    mmm_repeats : int, default 1
        Number of layout repeats at each multilevel.
    sl_extra_scaling_steps : int, default 2
        Extra scaling refinement steps.
    sl_scaling_type : ScalingType, default ScalingType.RelativeToDrawing
        How to scale the layout during refinement.
    merger_factor : float, default 2.0
        Controls coarsening aggressiveness.
    config : LayoutConfig, optional
        Full configuration object. If provided, overrides individual parameters.

    Example
    -------
    >>> from tmap.layout import ForceDirectedLayout
    >>> from tmap.graph.mst import MSTBuilder
    >>>
    >>> # Build MST from k-NN graph
    >>> tree = MSTBuilder().build(knn_graph)
    >>>
    >>> # Compute layout with custom parameters
    >>> layout = ForceDirectedLayout(
    ...     seed=42,
    ...     node_size=1/30,
    ...     mmm_repeats=2,
    ...     sl_extra_scaling_steps=10,
    ... )
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
        node_size: float = 1.0 / 65.0,
        mmm_repeats: int = 1,
        sl_extra_scaling_steps: int = 2,
        sl_scaling_type: ScalingType | None = None,
        merger_factor: float = 2.0,
        config: LayoutConfig | None = None,
    ) -> None:
        # Check OGDF availability early
        require_ogdf()

        super().__init__(seed=seed, max_iterations=max_iterations)

        # If full config provided, use it directly
        if config is not None:
            self._config = config
            # Override seed/deterministic from config if seed is also set
            if seed is not None:
                self._config.deterministic = True
                self._config.seed = seed
            return

        # Store individual parameters
        self._placer = placer if placer is not None else Placer.Barycenter
        self._merger = merger if merger is not None else Merger.LocalBiconnected
        self._node_size = node_size
        self._mmm_repeats = mmm_repeats
        self._sl_extra_scaling_steps = sl_extra_scaling_steps
        self._sl_scaling_type = sl_scaling_type if sl_scaling_type is not None else ScalingType.RelativeToDrawing
        self._merger_factor = merger_factor
        self._config = None  # Will be built in _make_config()

    def _make_config(self) -> LayoutConfig:
        """Create OGDF LayoutConfig from our parameters."""
        # If a full config was provided, use it
        if self._config is not None:
            return self._config

        config = LayoutConfig()
        config.fme_iterations = self.max_iterations
        config.placer = self._placer
        config.merger = self._merger
        config.node_size = self._node_size
        config.mmm_repeats = self._mmm_repeats
        config.sl_extra_scaling_steps = self._sl_extra_scaling_steps
        config.sl_scaling_type = self._sl_scaling_type
        config.merger_factor = self._merger_factor

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
