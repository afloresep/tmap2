from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np
from numpy.typing import NDArray

from tmap.graph.mst import MSTBuilder
from tmap.graph.types import Tree
from tmap.index.encoders.minhash import MinHash
from tmap.index.lsh_forest import LSHForest
from tmap.index.types import KNNGraph
from tmap.layout._ogdf import (
    LayoutConfig,
    layout_from_knn_graph,
    layout_from_tree,
    require_ogdf,
)

if TYPE_CHECKING:
    from tmap.visualization import TmapViz


class TMAP:
    """High-level estimator API for the TMAP pipeline.

    Provides a scikit-learn-style interface for computing TMAP embeddings.
    Supports two layout modes that share the same underlying k-NN graph:

    - ``layout='tree'`` (default): Computes a minimum spanning tree from the
      k-NN graph and uses force-directed layout on the tree edges. This
      produces the classic TMAP visualization where paths between any two
      points can be traced through the tree structure. Best for exploring
      local neighborhoods and relationships between specific data points.

    - ``layout='graph'``: Uses the full k-NN graph for force-directed layout
      without reducing to a tree first. Dense regions with many mutual
      neighbors are pulled together, producing cluster-like groupings similar
      to UMAP or t-SNE. The MST is still computed and accessible via
      ``tree_`` for programmatic exploration, but point positions reflect
      global density structure rather than tree topology.

    Both modes compute the same k-NN graph and MST internally. The only
    difference is which graph the layout algorithm operates on.

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
        - ``'cosine'``, ``'euclidean'``: Not yet implemented.
    n_permutations : int, default=128
        Number of MinHash permutations. Only used with ``metric='jaccard'``.
    kc : int, default=10
        Candidate multiplier for LSH queries. The forest retrieves
        ``k * kc`` candidates before linear scan. Higher values improve
        recall at the cost of speed.
    seed : int, default=1
        Random seed for reproducibility.
    mst_bias : float, default=0.0
        Bias factor for MST construction. Not yet implemented.
    layout : str, default='tree'
        Layout mode. ``'tree'`` layouts the MST for detailed path
        exploration. ``'graph'`` layouts the full k-NN graph for a
        cluster-oriented overview.
    layout_iterations : int, default=1000
        Number of iterations for the force-directed layout algorithm.
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
        n_neighbors: int = 10,
        metric: str = "jaccard",
        n_permutations: int = 128,
        kc: int = 10,
        seed: int = 1,
        mst_bias: float = 0.0,
        layout: str = "tree",
        layout_iterations: int = 1000,
        layout_config: Any | None = None,
        **kwargs: Any,
    ) -> None:
        # Backward compatibility with early prototype keywords.
        legacy_num_trees = kwargs.pop("l", None)
        legacy_num_trees_2 = kwargs.pop("lsh_num_trees", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")
        if legacy_num_trees is not None or legacy_num_trees_2 is not None:
            import warnings

            warnings.warn(
                "The 'l' and 'lsh_num_trees' parameters are deprecated in TMAP. "
                "The number of LSH bands is now auto-selected from n_permutations. "
                "This parameter will be ignored.",
                DeprecationWarning,
                stacklevel=2,
            )

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
        self.mst_bias = mst_bias
        self.layout = layout
        self.layout_iterations = layout_iterations
        self.layout_config = layout_config

        self._embedding: NDArray[np.float32] | None = None
        self._tree: Tree | None = None
        self._graph: KNNGraph | None = None
        self._lsh_forest: LSHForest | None = None

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

            encoder = MinHash(num_perm=self.n_permutations, seed=self.seed)
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
            raise NotImplementedError(
                f"metric={self.metric!r} is not implemented yet. "
                "Use metric='jaccard' or metric='precomputed'."
            )
        else:
            # Defensive fallback; __init__ already validates metrics.
            raise ValueError(f"Unsupported metric {self.metric!r}")

        self._graph = knn
        self._tree = MSTBuilder(bias_factor=self.mst_bias).build(knn)

        require_ogdf()
        config = self._make_layout_config()

        if self.layout == "tree":
            x, y = layout_from_tree(self._tree, config=config)
        else:
            # Layout the full k-NN graph; points in dense regions cluster.
            x, y, _, _ = layout_from_knn_graph(
                self._graph,
                config=config,
                create_mst=False,
            )

        self._embedding = np.column_stack([x, y]).astype(np.float32, copy=False)
        return self

    def fit_transform(
        self,
        X: Any | None = None,
        *,
        knn_graph: KNNGraph | None = None,
    ) -> NDArray[np.float32]:
        """Fit and return 2D coordinates with shape (n_samples, 2)."""
        self.fit(X, knn_graph=knn_graph)
        return self.embedding_

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
            raise RuntimeError("Estimator is not fitted. Call fit() first.")
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
        return viz.save(path)

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

        Requires the ``jupyter-scatter`` package (installed by default with
        ``pip install -e .``).

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
