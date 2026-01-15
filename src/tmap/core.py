"""
TreeMap: Main orchestrator class.

DESIGN PATTERN: Composition over Inheritance
--------------------------------------------
TreeMap doesn't inherit from Index, Graph, or Layout.
Instead, it USES them (composition).

    class TreeMap:
        def __init__(self, index: Index, layout: Layout):
            self.index = index   # HAS-A index
            self.layout = layout  # HAS-A layout

Why composition?
1. Flexibility: Mix any Index with any Layout
2. Testability: Mock individual components
3. Clarity: TreeMap orchestrates, doesn't implement

Compare to inheritance (DON'T do this):
    class TreeMap(Index, Layout):  # Multiple inheritance mess
        ...


DEPENDENCY INJECTION
--------------------
Components are passed in, not created internally.

    # Good: User controls which implementations
    tm = TreeMap(
        index=FaissIndex(seed=42),
        layout=ForceDirectedLayout(seed=42),
    )

    # Bad: Hardcoded inside TreeMap
    class TreeMap:
        def __init__(self):
            self.index = FaissIndex()  # Can't change this!

This makes TreeMap configurable and testable.


FLUENT API (Method Chaining)
----------------------------
Methods return `self` so you can chain calls:

    coords = (
        TreeMap(index, layout)
        .build_from_vectors(data)
        .query_knn(k=20)
        .build_mst()
        .compute_layout()
        .get_coordinates()
    )

Each method returns `self`, enabling this pattern.
"""

from pathlib import Path
from typing import Self
import pickle

import numpy as np
from numpy.typing import NDArray

from tmap.index.base import Index
from tmap.index.types import KNNGraph, EdgeList
from tmap.graph.types import Tree
from tmap.graph.mst import MSTBuilder
from tmap.layout.base import Layout
from tmap.layout.types import Coordinates


class TreeMap:
    """
    Main TreeMap class - orchestrates the full pipeline.

    Pipeline:
        Input -> Index -> KNN Graph -> MST -> Layout -> Coordinates

    Example usage:
        # From vectors
        tm = TreeMap(index=FaissIndex(), layout=ForceDirectedLayout())
        tm.build_from_vectors(embeddings)
        tm.compute(k=20)
        coords = tm.coordinates

        # From edge list (generic)
        tm = TreeMap(index=AnnoyIndex(), layout=ForceDirectedLayout())
        tm.build_from_edges(edge_list)
        tm.compute(k=20)
        coords = tm.coordinates

        # Insertion (your most requested feature!)
        new_coords = tm.insert(new_vectors)
    """

    def __init__(
        self,
        index: Index,
        layout: Layout,
        mst_builder: MSTBuilder | None = None,
    ) -> None:
        """
        Initialize TreeMap with components.

        Args:
            index: Nearest-neighbor index (FaissIndex, AnnoyIndex, etc.)
            layout: Layout algorithm (ForceDirectedLayout, etc.)
            mst_builder: MST construction (default: MSTBuilder with no bias)

        DESIGN NOTE:
        All components are injected. TreeMap just orchestrates.
        """
        self.index = index
        self.layout = layout
        self.mst_builder = mst_builder or MSTBuilder(bias_factor=0.0)

        # State - populated as pipeline runs
        self._knn_graph: KNNGraph | None = None
        self._tree: Tree | None = None
        self._coordinates: Coordinates | None = None
        self._k: int | None = None

    # =========================================================================
    # BUILD METHODS - Entry points
    # =========================================================================

    def build_from_vectors(
        self,
        vectors: NDArray[np.float32],
        metric: str = "euclidean",
    ) -> Self:
        """
        Build from raw vectors (embeddings, fingerprints, etc.).

        Args:
            vectors: Shape (n_samples, n_features)
            metric: Distance metric

        Returns:
            self (for chaining)
        """
        self.index.build_from_vectors(vectors, metric)
        return self

    def build_from_edges(self, edges: EdgeList) -> Self:
        """
        Build from pre-computed edges (generic entry point).

        Args:
            edges: EdgeList with (source, target, distance) tuples

        Returns:
            self (for chaining)

        Example:
            # Visualize any similarity data
            edges = EdgeList(
                edges=[(0, 1, 0.9), (0, 2, 0.7), ...],
                n_nodes=n,
            )
            tm.build_from_edges(edges)
        """
        self.index.build_from_edges(edges)
        return self

    # =========================================================================
    # COMPUTE METHODS - Run the pipeline
    # =========================================================================

    def compute(self, k: int) -> Self:
        """
        Run full pipeline: k-NN -> MST -> Layout.

        Args:
            k: Number of nearest neighbors

        Returns:
            self (for chaining)

        After calling, access results via:
            tm.knn_graph  # KNNGraph object
            tm.tree       # Tree (MST) object
            tm.coordinates  # Coordinates object
        """
        self._k = k

        # Step 1: Query k-NN graph
        self._knn_graph = self.index.query_knn(k)

        # Step 2: Build MST
        self._tree = self.mst_builder.build(self._knn_graph)

        # Step 3: Compute layout
        self._coordinates = self.layout.compute(self._tree)

        return self

    # =========================================================================
    # INSERTION - Most requested feature!
    # =========================================================================

    def insert(
        self,
        new_vectors: NDArray[np.float32],
        mode: str = "local",
    ) -> Coordinates:
        """
        Insert new points into existing visualization.

        Args:
            new_vectors: Shape (n_new, n_features)
            mode: "local" (fast) or "global" (thorough)

        Returns:
            Updated Coordinates with new nodes

        INSERTION MODES:
        ----------------
        "local" (fast):
        - Query k-NN for new points
        - Add edges to existing MST
        - Only move new nodes and neighbors
        - Good for: Real-time/interactive, few insertions

        "global" (thorough):
        - Rebuild k-NN including new points
        - Rebuild MST
        - Re-run layout using existing as initial positions
        - Good for: Batch insertions, final visualization

        Example:
            # Start with 10k points
            tm.build_from_vectors(initial_data)
            tm.compute(k=20)

            # Later, add 100 more
            new_coords = tm.insert(new_data, mode="local")
        """
        if self._coordinates is None:
            raise RuntimeError("Must call compute() before insert()")

        if mode == "local":
            return self._insert_local(new_vectors)
        elif mode == "global":
            return self._insert_global(new_vectors)
        else:
            raise ValueError(f"mode must be 'local' or 'global', got {mode}")

    def _insert_local(self, new_vectors: NDArray[np.float32]) -> Coordinates:
        """
        Fast local insertion.

        Steps:
        1. Query k-NN for new points against existing index
        2. Connect new nodes to their nearest existing neighbors
        3. Run incremental layout (only move new nodes + neighbors)
        """
        # TODO: Implement
        # This is where you'll add your insertion logic
        raise NotImplementedError("Local insertion not yet implemented")

    def _insert_global(self, new_vectors: NDArray[np.float32]) -> Coordinates:
        """
        Thorough global insertion.

        Steps:
        1. Add new vectors to index
        2. Rebuild k-NN graph for all points
        3. Rebuild MST
        4. Run layout with existing coords as initial positions
        """
        # TODO: Implement
        raise NotImplementedError("Global insertion not yet implemented")

    # =========================================================================
    # PROPERTIES - Access results
    # =========================================================================

    @property
    def knn_graph(self) -> KNNGraph | None:
        """k-NN graph (after compute())."""
        return self._knn_graph

    @property
    def tree(self) -> Tree | None:
        """MST (after compute())."""
        return self._tree

    @property
    def coordinates(self) -> Coordinates | None:
        """Layout coordinates (after compute())."""
        return self._coordinates

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the visualization."""
        return self.index.n_nodes

    # =========================================================================
    # PERSISTENCE - Save/load entire TreeMap state
    # =========================================================================

    def save(self, path: str | Path) -> None:
        """
        Save TreeMap state to disk.

        Saves:
        - Index state (for insertion queries)
        - Tree structure
        - Coordinates
        - Configuration

        Usage:
            tm.save("my_tmap.pkl")
            # Later:
            tm = TreeMap.load("my_tmap.pkl")
        """
        path = Path(path)

        state = {
            "k": self._k,
            "knn_graph": self._knn_graph,
            "tree": self._tree,
            "coordinates": self._coordinates,
            # Note: Index and Layout are saved separately
            # because they may have their own binary formats
        }

        # Save TreeMap state
        with open(path, "wb") as f:
            pickle.dump(state, f)

        # Save index (may have its own format)
        self.index.save(path.with_suffix(".index"))

        # Save layout
        self.layout.save(path.with_suffix(".layout"))

    @classmethod
    def load(
        cls,
        path: str | Path,
        index_cls: type[Index] | None = None,
        layout_cls: type[Layout] | None = None,
    ) -> "TreeMap":
        """
        Load TreeMap from disk.

        Args:
            path: Path to saved TreeMap
            index_cls: Index class to use (must match saved type)
            layout_cls: Layout class to use (must match saved type)

        DESIGN NOTE:
        We need the classes because Python can't automatically
        instantiate the right subclass from saved data.
        """
        path = Path(path)

        # Load main state
        with open(path, "rb") as f:
            state = pickle.load(f)

        # Load index
        if index_cls is None:
            raise ValueError("Must provide index_cls to load TreeMap")
        index = index_cls.load(path.with_suffix(".index"))

        # Load layout
        if layout_cls is None:
            raise ValueError("Must provide layout_cls to load TreeMap")
        layout = layout_cls.load(path.with_suffix(".layout"))

        # Create TreeMap and restore state
        tm = cls(index=index, layout=layout)
        tm._k = state["k"]
        tm._knn_graph = state["knn_graph"]
        tm._tree = state["tree"]
        tm._coordinates = state["coordinates"]

        return tm
