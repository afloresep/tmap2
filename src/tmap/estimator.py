from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, Tuple

import numpy as np
from numpy.typing import NDArray

from tmap.graph.types import Tree
from tmap.index.encoders.minhash import MinHash
from tmap.index.lsh_forest import LSHForest
from tmap.index.types import KNNGraph
from tmap.layout._ogdf import (
    LayoutConfig,
    layout_from_edge_list,
    layout_from_knn_graph,
    require_ogdf,
)

if TYPE_CHECKING:
    from tmap.visualization import TmapViz

# Experimental graph-mode sparsification defaults.
# these are internal until i figure out the graph mode and is fully stabilized.
def _resolve_ann_backend(n_samples: int = 0, seed: int | None = None) -> Any:
    """Auto-detect and return the best available ANN index.

    For large datasets (n >= 50k), prefers FaissIndex which supports IVF/IVFPQ
    auto-selection for sub-linear query time.  For small datasets, prefers
    NNDescentIndex (simpler, no training step).

    Raises ImportError with install instructions if neither is available.
    """
    _have_nnd = False
    _have_faiss = False

    try:
        import pynndescent  # noqa: F401

        _have_nnd = True
    except ImportError:
        pass

    try:
        import faiss  # noqa: F401

        _have_faiss = True
    except ImportError:
        pass

    if not _have_nnd and not _have_faiss:
        raise ImportError(
            "metric='cosine' and metric='euclidean' require either pynndescent or faiss. "
            "Install one with:\n"
            "  pip install pynndescent\n"
            "  pip install faiss-cpu"
        )

    # Large data: prefer FAISS (IVF/IVFPQ auto-selection handles scale)
    if n_samples >= 50_000 and _have_faiss:
        from tmap.index.faiss_index import FaissIndex

        return FaissIndex(seed=seed)

    # Small data: prefer NNDescent (simpler, no training)
    if _have_nnd:
        from tmap.index.nndescent import NNDescentIndex

        return NNDescentIndex(seed=seed)

    # Fallback: whatever is available
    from tmap.index.faiss_index import FaissIndex

    return FaissIndex(seed=seed)


_GRAPH_LAYOUT_REQUIRE_MUTUAL_DEFAULT = True
_GRAPH_LAYOUT_MAX_DEGREE_DEFAULT = 8


class TMAP:
    """High-level estimator API for the TMAP pipeline.

    Provides a scikit-learn-style interface for computing TMAP embeddings.
    Supports two layout modes that share the same underlying k-NN graph:

    - ``layout='tree'`` (default): Computes a minimum spanning tree from the
      k-NN graph and uses force-directed layout on the tree edges. This
      produces the classic TMAP visualization where paths between any two
      points can be traced through the tree structure. Best for exploring
      local neighborhoods and relationships between specific data points.

    - ``layout='graph'``: Experimental overview mode for cluster inspection.
      It layouts a sparsified k-NN graph using reciprocal neighbor filtering
      plus per-node top-edge capping (defaults: ``mutual=True``,
      ``max_degree=8``) to improve runtime and quality versus dense full-graph
      layout.

    Both modes compute the same k-NN graph and expose an MST via ``tree_``.

    Parameters
    ----------
    n_neighbors : int, default=10
        Number of nearest neighbors per point for the k-NN graph.
    metric : str, default='jaccard'
        Distance metric. Options:

        - ``'jaccard'``: For binary feature matrices (e.g. molecular
          fingerprints). Uses MinHash encoding + LSH Forest.
        - ``'precomputed'``: X is a square distance matrix. Suitable for
          small datasets (< ~50k points due to O(n^2) memory). For larger
          datasets, compute k-NN externally and pass via ``knn_graph=``.
        - ``'cosine'``, ``'euclidean'``: For dense float vectors (e.g. protein
          embeddings). Requires ``pynndescent`` or ``faiss-cpu``.
    n_permutations : int, default=512
        Number of MinHash permutations. Only used with ``metric='jaccard'``.
    kc : int, default=10
        Candidate multiplier for LSH queries. The forest retrieves
        ``k * kc`` candidates before linear scan. Higher values improve
        recall at the cost of speed.
    seed : int, default=42
        Random seed for OGDF layout reproducibility.
    minhash_seed : int, default=42
        Random seed for MinHash permutation generation when
        ``metric='jaccard'``. Keep this fixed to make k-NN graph topology
        stable while varying ``seed`` for layout initialization.
    mst_bias : float, default=0.0
        Reserved for low-level MST tuning. The high-level ``TMAP`` estimator
        currently ignores this value.
    layout : str, default='tree'
        Layout mode. ``'tree'`` uses the stable OGDF MST path for detailed
        path exploration. ``'graph'`` is experimental and uses a sparsified
        non-MST graph layout for quick cluster-oriented overview.
    layout_iterations : int, default=1000
        Number of iterations for the force-directed layout algorithm.
    store_index : bool, default=False
        If True, keep the ANN index in memory after ``fit()``. Access it
        via ``index_``. Useful for later ``query_point`` calls (e.g.
        embedding new data). Only applies to ``metric='cosine'`` and
        ``metric='euclidean'``.
    layout_config : LayoutConfig or None, default=None
        Advanced OGDF layout configuration. Overrides ``layout_iterations``
        and ``seed`` when provided.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, 2)
        2D coordinates after fitting.
    tree_ : Tree
        Minimum spanning tree. Available in both layout modes.
    graph_ : KNNGraph
        The k-NN graph computed during fitting.
    lsh_forest_ : LSHForest
        The LSH Forest index. Only available when ``metric='jaccard'``.

    Examples
    --------
    Basic usage with binary fingerprints:

    >>> model = TMAP(n_neighbors=20).fit(binary_matrix)
    >>> coords = model.embedding_

    Cluster-oriented overview:

    >>> model = TMAP(n_neighbors=20, layout='graph').fit(binary_matrix)

    Precomputed distance matrix (small datasets):

    >>> model = TMAP(metric='precomputed', n_neighbors=10).fit(dist_matrix)

    External k-NN graph (large datasets, any metric):

    >>> from tmap.index.types import KNNGraph
    >>> knn = KNNGraph.from_arrays(faiss_indices, faiss_distances)
    >>> model = TMAP().fit(knn_graph=knn)
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        metric: str = "jaccard",
        n_permutations: int = 512,
        kc: int = 10,
        seed: int = 42,
        minhash_seed: int = 42,
        mst_bias: float = 0.0,
        layout: str = "tree",
        layout_iterations: int = 1000,
        layout_config: Any | None = None,
        store_index: bool = False,
    ) -> None:
        if n_neighbors <= 0:
            raise ValueError(f"n_neighbors must be > 0, got {n_neighbors}")
        if n_permutations <= 0:
            raise ValueError(f"n_permutations must be > 0, got {n_permutations}")
        if kc <= 0:
            raise ValueError(f"kc must be > 0, got {kc}")
        if not 0.0 <= mst_bias <= 1.0:
            raise ValueError(f"mst_bias must be in [0, 1], got {mst_bias}")

        metric = metric.lower()
        valid_metrics = {"jaccard", "precomputed", "cosine", "euclidean"}
        if metric not in valid_metrics:
            valid_list = ", ".join(sorted(valid_metrics))
            raise ValueError(f"Unsupported metric {metric!r}. Supported metrics: {valid_list}")

        layout = layout.lower()
        valid_layouts = {"tree", "graph"}
        if layout not in valid_layouts:
            valid_list = ", ".join(sorted(valid_layouts))
            raise ValueError(f"Unsupported layout {layout!r}. Supported layouts: {valid_list}")

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_permutations = n_permutations
        self.kc = kc
        self.seed = seed
        self.minhash_seed = minhash_seed
        self.mst_bias = mst_bias
        self.layout = layout
        self.layout_iterations = layout_iterations
        self.layout_config = layout_config
        self.store_index = store_index

        self._embedding: NDArray[np.float32] | None = None
        self._index: Any | None = None
        self._tree: Tree | None = None
        self._graph: KNNGraph | None = None
        self._lsh_forest: LSHForest | None = None

        # Graph mode defaults are module-level constants by design.
        self._graph_mode_mutual = _GRAPH_LAYOUT_REQUIRE_MUTUAL_DEFAULT
        self._graph_mode_max_degree = _GRAPH_LAYOUT_MAX_DEGREE_DEFAULT

    def fit(
        self,
        X: Any | None = None,
        *,
        knn_graph: KNNGraph | None = None,
    ) -> Self:
        """Fit the estimator and compute embedding coordinates."""
        if knn_graph is not None and X is not None:
            raise ValueError("Pass either X or knn_graph, not both.")
        if knn_graph is None and X is None:
            raise ValueError("Either X or knn_graph must be provided.")

        if knn_graph is not None:
            knn = knn_graph
            self._lsh_forest = None
        elif self.metric == "jaccard":
            binary_matrix = self._coerce_binary_matrix(X)
            if self.n_neighbors >= binary_matrix.shape[0]:
                raise ValueError(
                    f"n_neighbors={self.n_neighbors} must be < n_samples={binary_matrix.shape[0]}"
                )

            encoder = MinHash(num_perm=self.n_permutations, seed=self.minhash_seed)
            signatures = encoder.batch_from_binary_array(binary_matrix)

            forest = LSHForest(d=self.n_permutations)
            forest.batch_add(signatures)
            forest.index()
            knn = forest.get_knn_graph(k=self.n_neighbors, kc=self.kc)
            self._lsh_forest = forest
        elif self.metric == "precomputed":
            distance_matrix = self._coerce_distance_matrix(X)
            knn = KNNGraph.from_distance_matrix(distance_matrix, k=self.n_neighbors)
            self._lsh_forest = None
        elif self.metric in {"cosine", "euclidean"}:
            X_dense = self._coerce_dense_matrix(X)
            if self.n_neighbors >= X_dense.shape[0]:
                raise ValueError(
                    f"n_neighbors={self.n_neighbors} must be < n_samples={X_dense.shape[0]}"
                )
            index = _resolve_ann_backend(
                n_samples=X_dense.shape[0], seed=self.seed
            )
            index.build_from_vectors(X_dense, metric=self.metric)
            knn = index.query_knn(k=self.n_neighbors)
            self._lsh_forest = None
            self._index = index if self.store_index else None
        else:
            # Defensive fallback; __init__ already validates metrics.
            raise ValueError(f"Unsupported metric {self.metric!r}")

        self._graph = knn

        require_ogdf()
        config = self._make_layout_config()

        if self.layout == "tree":
            x, y, s, t = layout_from_knn_graph(knn, config=config, create_mst=True)
            self._tree = self._tree_from_ogdf_edges(knn, s, t)
        else:
            edges = self._graph_edges_from_knn(
                knn,
                max_degree=min(self._graph_mode_max_degree, knn.k),
                mutual=self._graph_mode_mutual,
            )
            x, y, s, t = layout_from_edge_list(
                knn.n_nodes,
                edges,
                config=config,
                create_mst=False,
            )
            # Defer MST extraction until tree_ is requested.
            self._tree = None

        self._embedding = np.column_stack([x, y]).astype(np.float32, copy=False)
        return self

    def fit_transform(
        self,
        X: Any | None = None,
        *,
        knn_graph: KNNGraph | None = None,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int32], NDArray[np.int32]]:
        """Fit and return 2D coordinates with shape (n_samples, 2)."""
        self.fit(X, knn_graph=knn_graph)
        # Return x,y coordinates  + s,t edges
        return (
            self.embedding_[:, 0],
            self.embedding_[:, 1],
            self.tree_.edges[:, 0],
            self.tree_.edges[:, 1],
        )

    def transform(self, X: Any) -> NDArray[np.float32]:
        """Embed new points into an existing embedding."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support transform(new_X) yet."
        )

    @property
    def embedding_(self) -> NDArray[np.float32]:
        """Coordinates with shape (n_samples, 2)."""
        if self._embedding is None:
            raise RuntimeError("Estimator is not fitted. Call fit() first.")
        return self._embedding

    @property
    def tree_(self) -> Tree:
        """Minimum spanning tree produced during fit."""
        if self._tree is None:
            if self._graph is None:
                raise RuntimeError("Estimator is not fitted. Call fit() first.")
            self._tree = self._extract_tree_from_knn(self._graph)
        return self._tree

    @property
    def graph_(self) -> KNNGraph:
        """k-NN graph produced during fit."""
        if self._graph is None:
            raise RuntimeError("Estimator is not fitted. Call fit() first.")
        return self._graph

    @property
    def lsh_forest_(self) -> LSHForest:
        """LSHForest used for metric='jaccard' fits."""
        if self._lsh_forest is None:
            raise RuntimeError(
                "No fitted LSHForest available. This estimator was not fitted with metric='jaccard'."
            )
        return self._lsh_forest

    @property
    def index_(self) -> Any:
        """ANN index retained after fit (only when ``store_index=True``)."""
        if self._index is None:
            raise RuntimeError(
                "No index stored. Use store_index=True when constructing TMAP "
                "to retain the ANN index after fit()."
            )
        return self._index

    def to_tmapviz(self, include_edges: bool = True) -> TmapViz:
        """Create a TmapViz object preloaded with fitted coordinates."""
        from tmap.visualization import TmapViz

        embedding = self.embedding_
        tree = self.tree_

        viz = TmapViz()
        viz.set_points(embedding[:, 0], embedding[:, 1])

        if include_edges and len(tree.edges) > 0:
            viz.set_edges(
                tree.edges[:, 0].astype(np.uint32, copy=False),
                tree.edges[:, 1].astype(np.uint32, copy=False),
            )

        return viz

    def to_html(
        self,
        path: str | Path,
        *,
        title: str | None = None,
        include_edges: bool = True,
    ) -> Path:
        """Write a default HTML visualization to disk."""
        viz = self.to_tmapviz(include_edges=include_edges)
        if title is not None:
            viz.title = title
        return viz.write_html(path)

    def plot(
        self,
        *,
        color_by: Any | None = None,
        color_map: str | list[str] | dict[str, str] | None = None,
        data: Any | None = None,
        tooltip_properties: list[str] | None = None,
        point_size: float = 3,
        opacity: float = 0.8,
        width: int | str = 800,
        height: int = 420,
        show: bool = True,
        controls: bool = False,
    ) -> Any:
        """Show an interactive scatter plot in a Jupyter notebook.

        Requires the ``jupyter-scatter`` package
        (``pip install tmap[notebook]``).

        Parameters
        ----------
        color_by : str, array-like, or None
            Column name in *data*, or a raw array of values.
        color_map : str, list, dict, or None
            Colormap override (e.g. ``'plasma'``, ``'tab20'``).
        data : DataFrame or None
            Metadata DataFrame whose columns can be referenced by
            *color_by* and *tooltip_properties*.
        tooltip_properties : list of str or None
            Column names to show on hover.
        point_size : float, default 3
            Uniform point size.
        opacity : float, default 0.8
            Uniform point opacity.
        width : int or "auto", default 800
            Widget width. Use ``"auto"`` to follow notebook cell width.
        height : int, default 420
            Widget height in pixels.
        show : bool, default True
            If True, display the widget in notebook environments.
        controls : bool, default False
            If True and ``show`` is enabled, display jscatter's control toolbar.

        Returns
        -------
        jscatter.Scatter
            The configured widget.
        """
        from tmap.visualization.jupyter import _display_scatter, to_jscatter

        scatter = to_jscatter(
            self.embedding_,
            color_by=color_by,
            color_map=color_map,
            data=data,
            tooltip_properties=tooltip_properties,
            point_size=point_size,
            opacity=opacity,
            width=width,
            height=height,
        )

        if show:
            _display_scatter(scatter, controls=controls)

        return scatter

    def plot_static(
        self,
        *,
        color_by: Any | None = None,
        color_map: str | None = None,
        data: Any | None = None,
        edges: bool = True,
        edge_color: str = "#cccccc",
        edge_alpha: float = 0.3,
        edge_linewidth: float = 0.3,
        point_size: float = 1.0,
        alpha: float = 0.8,
        ax: Any | None = None,
        figsize: tuple[float, float] = (8, 8),
    ) -> Any:
        """Render the embedding as a static matplotlib scatter plot.

        Parameters
        ----------
        color_by : str, array-like, or None
            Column name in *data*, or a raw array of values.
        color_map : str or None
            Matplotlib colormap name.
        data : DataFrame or None
            Metadata DataFrame.
        edges : bool, default True
            If True, draw tree edges behind the points.
        edge_color : str, default '#cccccc'
            Color for edge lines.
        edge_alpha : float, default 0.3
            Opacity for edge lines.
        edge_linewidth : float, default 0.3
            Line width for edges.
        point_size : float, default 1.0
            Marker size.
        alpha : float, default 0.8
            Point opacity.
        ax : matplotlib Axes or None
            Draw into an existing axes.
        figsize : tuple, default (8, 8)
            Figure size when *ax* is None.

        Returns
        -------
        matplotlib.axes.Axes
        """
        from tmap.visualization.static import plot_static

        edge_arr = self.tree_.edges if edges else None
        return plot_static(
            self.embedding_,
            color_by=color_by,
            color_map=color_map,
            data=data,
            edges=edge_arr,
            edge_color=edge_color,
            edge_alpha=edge_alpha,
            edge_linewidth=edge_linewidth,
            point_size=point_size,
            alpha=alpha,
            ax=ax,
            figsize=figsize,
        )

    # ------------------------------------------------------------------
    # Tree exploration convenience methods
    # ------------------------------------------------------------------

    def path(self, from_idx: int, to_idx: int) -> list[int]:
        """Shortest path in the tree between two points.

        Delegates to :meth:`Tree.path`.

        Parameters
        ----------
        from_idx : int
            Source point index.
        to_idx : int
            Target point index.

        Returns
        -------
        list[int]
            Ordered node indices from source to target (inclusive).
        """
        return self.tree_.path(from_idx, to_idx)

    def distance(self, from_idx: int, to_idx: int) -> float:
        """Sum of edge weights along the tree path between two points.

        Delegates to :meth:`Tree.distance`.
        """
        return self.tree_.distance(from_idx, to_idx)

    def distances_from(self, source: int) -> NDArray[np.float32]:
        """Tree distance from *source* to every other point (pseudotime).

        Delegates to :meth:`Tree.distances_from`.

        Parameters
        ----------
        source : int
            Source point index.

        Returns
        -------
        NDArray[np.float32]
            Array of shape ``(n_samples,)`` with tree distances.
        """
        return self.tree_.distances_from(source)

    def _tree_from_ogdf_edges(
        self,
        knn: KNNGraph,
        s: NDArray[np.uint32],
        t: NDArray[np.uint32],
    ) -> Tree:
        """Build a Tree object from OGDF edge topology output."""
        edge_count = len(s)
        if edge_count == 0:
            return Tree(
                n_nodes=knn.n_nodes,
                edges=np.empty((0, 2), dtype=np.int32),
                weights=np.empty(0, dtype=np.float32),
                root=0,
            )

        edges = np.column_stack(
            [
                s.astype(np.int32, copy=False),
                t.astype(np.int32, copy=False),
            ]
        )

        # Recover edge weights from the k-NN table where possible.
        edge_weights: dict[tuple[int, int], float] = {}
        for i in range(knn.n_nodes):
            for j_idx in range(knn.k):
                j = int(knn.indices[i, j_idx])
                if j < 0 or j == i:
                    continue
                w = float(knn.distances[i, j_idx])
                if not np.isfinite(w):
                    continue
                key = (min(i, j), max(i, j))
                prev = edge_weights.get(key)
                if prev is None or w < prev:
                    edge_weights[key] = w

        weights = np.ones(edge_count, dtype=np.float32)
        for idx, (src, tgt) in enumerate(edges):
            key = (min(int(src), int(tgt)), max(int(src), int(tgt)))
            w_opt = edge_weights.get(key)
            if w_opt is not None:
                weights[idx] = np.float32(w_opt)

        degree = np.zeros(knn.n_nodes, dtype=np.int32)
        np.add.at(degree, edges[:, 0], 1)
        np.add.at(degree, edges[:, 1], 1)
        root = int(np.argmax(degree))

        return Tree(
            n_nodes=knn.n_nodes,
            edges=edges,
            weights=weights,
            root=root,
        )

    def _extract_tree_from_knn(self, knn: KNNGraph) -> Tree:
        """Extract MST topology via OGDF and convert to Tree."""
        config = self._make_tree_extraction_config()
        _, _, s, t = layout_from_knn_graph(knn, config=config, create_mst=True)
        return self._tree_from_ogdf_edges(knn, s, t)

    def _make_tree_extraction_config(self) -> Any | None:
        """Create a lightweight layout config used only for MST extraction."""
        if LayoutConfig is None:
            return None
        config = LayoutConfig()
        if hasattr(config, "fme_iterations"):
            config.fme_iterations = 1
        if hasattr(config, "deterministic"):
            config.deterministic = True
        if hasattr(config, "seed"):
            config.seed = self.seed
        return config

    def _graph_edges_from_knn(
        self,
        knn: KNNGraph,
        *,
        max_degree: int,
        mutual: bool,
    ) -> list[tuple[int, int, float]]:
        """Build a sparsified undirected edge list from a k-NN graph.

        Parameters
        ----------
        knn : KNNGraph
            Input k-NN graph.
        max_degree : int
            Keep at most this many ranked neighbors per node when proposing
            edges. Lower values reduce density and runtime.
        mutual : bool
            If True, keep an edge only when the neighbor relation is reciprocal
            (i in N(j) and j in N(i)).
        """
        n = knn.n_nodes
        k = knn.k
        if max_degree < 1:
            raise ValueError(f"max_degree must be >= 1, got {max_degree}")

        neighbor_sets: list[set[int]] = [set() for _ in range(n)]
        for i in range(n):
            for j_idx in range(k):
                j = int(knn.indices[i, j_idx])
                if j < 0 or j == i:
                    continue
                neighbor_sets[i].add(j)

        edges: list[tuple[int, int, float]] = []
        seen: set[tuple[int, int]] = set()

        for i in range(n):
            degree_limit = min(max_degree, k)
            for j_idx in range(degree_limit):
                j = int(knn.indices[i, j_idx])
                if j < 0 or j == i:
                    continue
                w = float(knn.distances[i, j_idx])
                if not np.isfinite(w):
                    continue
                if mutual and i not in neighbor_sets[j]:
                    continue
                a, b = (i, j) if i < j else (j, i)
                key = (a, b)
                if key in seen:
                    continue
                seen.add(key)
                edges.append((i, j, w))

        return edges

    def _make_layout_config(self) -> Any | None:
        if self.layout_config is not None:
            return self.layout_config
        if LayoutConfig is None:
            return None

        config = LayoutConfig()
        if hasattr(config, "fme_iterations"):
            config.fme_iterations = self.layout_iterations
        if hasattr(config, "deterministic"):
            config.deterministic = True
        if hasattr(config, "seed"):
            config.seed = self.seed
        return config

    def _coerce_binary_matrix(self, X: Any | None) -> NDArray[np.uint8]:
        if X is None:
            raise ValueError("X cannot be None for metric='jaccard'.")

        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError(
                "metric='jaccard' expects a 2D binary matrix of shape (n_samples, n_features)."
            )
        if arr.shape[0] < 2:
            raise ValueError("Need at least 2 samples.")

        if arr.dtype != np.bool_ and not np.issubdtype(arr.dtype, np.number):
            raise ValueError("Binary matrix must contain numeric/boolean values.")

        if not np.all((arr == 0) | (arr == 1)):
            raise ValueError("Binary matrix must contain only 0/1 values.")

        return arr.astype(np.uint8, copy=False)

    def _coerce_distance_matrix(self, X: Any | None) -> NDArray[np.float32]:
        if X is None:
            raise ValueError("X cannot be None for metric='precomputed'.")

        distances = np.asarray(X, dtype=np.float32)
        if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
            raise ValueError(
                "metric='precomputed' expects a square distance matrix with shape (n_samples, n_samples)."
            )
        if distances.shape[0] < 2:
            raise ValueError("Distance matrix must contain at least 2 samples.")
        if not np.all(np.isfinite(distances)):
            raise ValueError("Distance matrix must contain only finite values.")

        return distances

    def _coerce_dense_matrix(self, X: Any | None) -> NDArray[np.float32]:
        if X is None:
            raise ValueError(f"X cannot be None for metric={self.metric!r}.")

        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(
                f"metric={self.metric!r} expects a 2D matrix of shape (n_samples, n_features)."
            )
        if arr.shape[0] < 2:
            raise ValueError("Need at least 2 samples.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Input matrix must contain only finite values.")

        return arr
