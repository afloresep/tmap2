"""
Abstract base class for layout algorithms.

STRATEGY PATTERN IN ACTION
--------------------------
The Layout ABC defines WHAT a layout must do:
    - compute(tree) -> coordinates
    - support incremental updates (for insertion)
    - be deterministic given a seed

Concrete classes define HOW:
    - ForceDirectedLayout: Iterative force simulation
    - RadialLayout: Concentric circles from root
    - HierarchicalLayout: Top-down levels

CHOOSING A LAYOUT ALGORITHM
---------------------------
1. Force-directed (default recommendation):
   + Natural-looking, respects distances
   + Handles any tree structure
   - Slower for large trees
   - Needs tuning (iterations, forces)

2. Radial:
   + Fast (O(n) single pass)
   + Clear hierarchy visualization
   - Doesn't respect actual distances
   - Wastes space for unbalanced trees

3. Hierarchical:
   + Very fast
   + Clear parent-child relationships
   - Very sparse for deep trees
   - Doesn't show similarity at all

For TMAP-like visualization of high-dimensional data,
force-directed is usually best because it preserves
distance relationships.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self
import pickle

import numpy as np

from tmap.graph.types import Tree
from tmap.layout.types import Coordinates


class Layout(ABC):
    """
    Abstract base class for tree layout algorithms.

    IMPLEMENTATION GUIDE:
    1. Override _compute_initial() - full layout from scratch
    2. Override _compute_incremental() - update layout for new nodes
    3. Use self._rng for ALL randomness (ensures reproducibility)

    Example implementation:

        class MyLayout(Layout):
            def _compute_initial(self, tree: Tree) -> Coordinates:
                # Your algorithm here
                x = self._rng.random(tree.n_nodes)  # Use _rng!
                y = self._rng.random(tree.n_nodes)
                return Coordinates(x=x, y=y)

            def _compute_incremental(
                self,
                tree: Tree,
                existing: Coordinates,
                new_nodes: list[int],
            ) -> Coordinates:
                # Update existing layout with new nodes
                ...
    """

    def __init__(
        self,
        seed: int | None = None,
        max_iterations: int = 1000,
    ) -> None:
        """
        Initialize layout algorithm.

        Args:
            seed: Random seed for reproducibility
            max_iterations: Max iterations for iterative algorithms

        DESIGN NOTE: Common parameters in base class
        All layout algorithms benefit from seed and iteration control.
        Algorithm-specific parameters go in subclass __init__.
        """
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self.max_iterations = max_iterations

        # Cache for incremental updates
        self._last_tree: Tree | None = None
        self._last_coords: Coordinates | None = None

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def compute(self, tree: Tree) -> Coordinates:
        """
        Compute layout for a tree.

        This is the main entry point. It handles:
        1. Checking if we can do incremental update
        2. Resetting RNG for reproducibility
        3. Caching results for future incremental updates

        Args:
            tree: Tree structure to layout

        Returns:
            Coordinates with x, y positions for each node
        """
        # Reset RNG to ensure same seed -> same result
        self._rng = np.random.default_rng(self._seed)

        coords = self._compute_initial(tree)

        # Cache for potential incremental updates
        self._last_tree = tree
        self._last_coords = coords

        return coords

    def compute_incremental(
        self,
        tree: Tree,
        existing_coords: Coordinates,
        new_nodes: list[int],
    ) -> Coordinates:
        """
        Update layout after inserting new nodes.

        This is for the INSERTION feature - adding new data points
        to an existing visualization.

        Args:
            tree: Updated tree with new nodes
            existing_coords: Layout of original nodes
            new_nodes: Indices of newly added nodes

        Returns:
            Updated Coordinates for all nodes

        DESIGN NOTE: Two modes of insertion
        -----------------------------------
        1. Local insertion (fast):
           - Only move new nodes and immediate neighbors
           - Existing layout mostly unchanged
           - Good for real-time/interactive use

        2. Global re-layout (thorough):
           - Use existing coords as initial positions
           - Run full optimization
           - Better final result but slower

        The implementation chooses based on len(new_nodes).
        """
        # Reset RNG for reproducibility
        self._rng = np.random.default_rng(self._seed)

        return self._compute_incremental(tree, existing_coords, new_nodes)

    # =========================================================================
    # ABSTRACT METHODS - You must implement these
    # =========================================================================

    @abstractmethod
    def _compute_initial(self, tree: Tree) -> Coordinates:
        """
        Compute layout from scratch.

        Override this with your layout algorithm.
        Use self._rng for all randomness.
        """
        ...

    @abstractmethod
    def _compute_incremental(
        self,
        tree: Tree,
        existing: Coordinates,
        new_nodes: list[int],
    ) -> Coordinates:
        """
        Update layout for new nodes.

        Override this. Options:
        1. Simple: Place new nodes, run local optimization
        2. Full: Use existing as initial, run global optimization
        """
        ...

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def save(self, path: str | Path) -> None:
        """Save layout parameters and cached state."""
        path = Path(path)
        state = {
            "class": self.__class__.__name__,
            "seed": self._seed,
            "max_iterations": self.max_iterations,
            "last_coords": self._last_coords,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load layout from file."""
        with open(path, "rb") as f:
            state = pickle.load(f)

        instance = cls(
            seed=state["seed"],
            max_iterations=state["max_iterations"],
        )
        instance._last_coords = state["last_coords"]
        return instance
