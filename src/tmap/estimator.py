from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, Tuple

import numpy as np
from numpy.typing import NDArray

from tmap.graph.mst import _tree_from_ogdf_edges, tree_from_knn_graph
from tmap.graph.types import Tree
from tmap.index.encoders.minhash import MinHash
from tmap.index.lsh_forest import LSHForest
from tmap.index.types import KNNGraph
from tmap.layout._ogdf import (
    LayoutConfig,
    layout_from_knn_graph,
    require_ogdf,
)

if TYPE_CHECKING:
    from tmap.visualization import TmapViz


def _resolve_ann_backend(seed: int | None = None) -> Any:
    """Return a FaissIndex for cosine/euclidean kNN search.

    Raises ImportError with install instructions if faiss is not available.
    """
    try:
        import faiss  # noqa: F401
    except ImportError:
        raise ImportError(
            "metric='cosine' and metric='euclidean' require faiss. "
            "Install with:\n  pip install faiss or \
                conda install faiss (recommended for MacOS (arm64)"
        )

    from tmap.index.faiss_index import FaissIndex

    return FaissIndex(seed=seed)


def _select_lsh_l(d: int, n_samples: int) -> int:
    """Auto-select number of LSH prefix trees based on d and dataset size.

    The LSH band width (k_band = d // l) controls discrimination:
    - Short bands (small k_band): high recall but many false positives.
      At large N, the candidate budget (k*kc) overflows with random matches.
    - Long bands (large k_band): precise candidates but low recall.
      At small N, too few collisions to find any neighbors.

    We use k_band=4 up to ~1M points (tested optimal for 100k molecules),
    then gradually increase for larger datasets where the collision pool
    would otherwise overwhelm the candidate budget.
    """
    if n_samples <= 1_000_000:
        k_band = 4
    else:
        # Scale up gently: k_band=5 at ~2M, 6 at ~4M, 7 at ~8M
        k_band = max(4, 4 + round(math.log2(n_samples / 1_000_000)))
    n_trees = max(8, d // k_band)
    return min(n_trees, d)


class TMAP:
    """High-level estimator API for the TMAP pipeline.

    Provides a scikit-learn-style interface for computing TMAP embeddings.
    Computes a minimum spanning tree from the k-NN graph and uses
    force-directed layout on the tree edges. This produces the classic TMAP
    visualization where paths between any two points can be traced through
    the tree structure.

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
          embeddings). Requires ``faiss-cpu``.
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
        Minimum spanning tree.
    graph_ : KNNGraph
        The k-NN graph computed during fitting.
    lsh_forest_ : LSHForest
        The LSH Forest index. Only available when ``metric='jaccard'``.

    Examples
    --------
    Basic usage with binary fingerprints:

    >>> model = TMAP(n_neighbors=20).fit(binary_matrix)
    >>> coords = model.embedding_

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

        metric = metric.lower()
        valid_metrics = {"jaccard", "precomputed", "cosine", "euclidean"}
        if metric not in valid_metrics:
            valid_list = ", ".join(sorted(valid_metrics))
            raise ValueError(f"Unsupported metric {metric!r}. Supported metrics: {valid_list}")

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_permutations = n_permutations
        self.kc = kc
        self.seed = seed
        self.minhash_seed = minhash_seed
        self.layout_iterations = layout_iterations
        self.layout_config = layout_config
        self.store_index = store_index

        self._embedding: NDArray[np.float32] | None = None
        self._index: Any | None = None
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
            signatures, n_samples = self._encode_jaccard(X)

            lsh_l = _select_lsh_l(self.n_permutations, n_samples)
            forest = LSHForest(d=self.n_permutations, l=lsh_l)
            forest.batch_add(signatures)
            del signatures  # forest has its own copy
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
            index = _resolve_ann_backend(seed=self.seed)
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
        x, y, s, t = layout_from_knn_graph(knn, config=config, create_mst=True)
        self._tree = _tree_from_ogdf_edges(knn, s, t)

        self._embedding = np.column_stack([x, y]).astype(np.float32, copy=False)
        return self

    def fit_transform(
        self,
        X: Any | None = None,
        *,
        knn_graph: KNNGraph | None = None,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int32], NDArray[np.int32]]:
        """Fit the estimator and return ``(x, y, s, t)`` arrays."""
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
        # TODO: Decide whether transform() should wrap add_points().
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
            self._tree = tree_from_knn_graph(self._graph)
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

    def serve(self, port: int = 8050, include_edges: bool = True, **kwargs: Any) -> None:
        """Serve the TMAP visualization on a local HTTP server.

        For large datasets (>1M points), this avoids embedding all data
        inline in a single HTML file.

        Parameters
        ----------
        port : int, default 8050
            TCP port for the local server.
        include_edges : bool, default True
            Include tree edges in the visualization.
        **kwargs
            Forwarded to ``TmapViz.serve()``.
        """
        viz = self.to_tmapviz(include_edges=include_edges)
        viz.serve(port=port, **kwargs)

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

    def save(self, path: str | Path) -> Path:
        """Save the fitted model to disk.

        The entire estimator state is serialised, including the embedding,
        tree, KNN graph, and, when present, the LSH Forest and ANN index.
        A saved model can later be loaded and extended with
        :meth:`add_points`.

        Parameters
        ----------
        path : str or Path
            Destination file path (e.g. ``"model.tmap"``).

        Returns
        -------
        Path
            The path the model was written to.

        Examples
        --------
        >>> model = TMAP(n_neighbors=10).fit(X)
        >>> model.save("my_model.tmap")
        >>> loaded = TMAP.load("my_model.tmap")
        >>> loaded.add_points(X_new)
        """
        if self._embedding is None:
            raise RuntimeError("Estimator is not fitted. Call fit() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path: str | Path) -> TMAP:
        """Load a previously saved model from disk.

        Parameters
        ----------
        path : str or Path
            File previously written by :meth:`save`.

        Returns
        -------
        TMAP
            The restored estimator, ready for queries or :meth:`add_points`.
        """
        path = Path(path)
        with open(path, "rb") as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise TypeError(f"Expected a TMAP instance, got {type(model).__name__}")
        return model

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

    def add_points(self, X: Any) -> NDArray[np.float32]:
        """Add new points to an existing embedding without re-fitting.

        Positions new points as inverse-distance weighted centroids of their
        nearest existing neighbors, then extends the tree and KNN graph.
        Existing coordinates are never modified.

        Parameters
        ----------
        X : array-like
            New data to insert. Interpretation depends on ``metric``:

            - ``'jaccard'``: ``(m, n_features)`` binary matrix
            - ``'cosine'``/``'euclidean'``: ``(m, n_features)`` float matrix
            - ``'precomputed'``: ``(m, n_existing)`` distance matrix (new->existing)

        Returns
        -------
        NDArray[np.float32]
            ``(m, 2)`` coordinates of the newly placed points.

        Raises
        ------
        RuntimeError
            If the estimator has not been fitted, or if ``metric`` is
            ``'cosine'``/``'euclidean'`` and ``store_index`` was ``False``.
        ValueError
            If ``metric='precomputed'`` and ``X.shape[1] != n_existing``.

        Notes
        -----
        - FAISS indices are not updated; subsequent ``add_points``
          calls only see the original fit data as neighbors.
        - For ``metric='jaccard'``, the LSH Forest *is* updated, so subsequent
          calls can discover previously added points.
        - For ``m > 0.2 * n``, quality degrades; consider re-fitting instead.
        - Positions are approximate (no force-directed optimisation).
        """
        if self._embedding is None or self._graph is None:
            raise RuntimeError("Estimator is not fitted. Call fit() first.")

        new_indices, new_distances, m = self._query_new_points(X)

        if m == 0:
            return np.empty((0, 2), dtype=np.float32)

        new_coords = self._position_new_points(new_indices)
        self._extend_tree(new_indices, new_distances, m)

        # Extend KNN graph
        self._graph = KNNGraph(
            indices=np.concatenate([self._graph.indices, new_indices]),
            distances=np.concatenate([self._graph.distances, new_distances]),
        )

        # Update embedding
        self._embedding = np.concatenate([self._embedding, new_coords])

        return new_coords

    def _query_new_points(self, X: Any) -> tuple[NDArray[np.int32], NDArray[np.float32], int]:
        """Dispatch neighbor queries based on metric.

        Returns ``(indices, distances, m)`` where shapes are ``(m, k)``.
        """
        k = self.n_neighbors

        if self.metric == "jaccard":
            binary = self._coerce_binary_matrix(X, min_samples=0)
            m = binary.shape[0]
            if m == 0:
                return (
                    np.empty((0, k), dtype=np.int32),
                    np.empty((0, k), dtype=np.float32),
                    0,
                )

            encoder = MinHash(num_perm=self.n_permutations, seed=self.minhash_seed)
            signatures = encoder.batch_from_binary_array(binary)

            forest = self._lsh_forest
            if forest is None:
                raise RuntimeError(
                    "No LSH Forest available. Cannot add_points without a jaccard-fitted estimator."
                )

            all_indices = np.empty((m, k), dtype=np.int32)
            all_distances = np.empty((m, k), dtype=np.float32)

            for i in range(m):
                results = forest.query_linear_scan(signatures[i], k, self.kc)
                for j, (dist, idx) in enumerate(results[:k]):
                    all_indices[i, j] = idx
                    all_distances[i, j] = dist
                # Pad with -1 if fewer than k results
                for j in range(len(results), k):
                    all_indices[i, j] = -1
                    all_distances[i, j] = np.inf

            # Update LSH Forest so future add_points sees these points
            forest.batch_add(signatures)
            forest.index()

            return all_indices, all_distances, m

        elif self.metric in {"cosine", "euclidean"}:
            if self._index is None:
                raise RuntimeError(
                    "No ANN index stored. Reconstruct with store_index=True "
                    "to use add_points() with metric={!r}.".format(self.metric)
                )
            X_dense = self._coerce_dense_matrix(X, min_samples=0)
            m = X_dense.shape[0]
            if m == 0:
                return (
                    np.empty((0, k), dtype=np.int32),
                    np.empty((0, k), dtype=np.float32),
                    0,
                )

            all_indices = np.empty((m, k), dtype=np.int32)
            all_distances = np.empty((m, k), dtype=np.float32)

            for i in range(m):
                idx_arr, dist_arr = self._index.query_point(X_dense[i], k)
                n_returned = min(len(idx_arr), k)
                all_indices[i, :n_returned] = idx_arr[:n_returned]
                all_distances[i, :n_returned] = dist_arr[:n_returned]
                for j in range(n_returned, k):
                    all_indices[i, j] = -1
                    all_distances[i, j] = np.inf

            return all_indices, all_distances, m

        elif self.metric == "precomputed":
            dist_matrix = np.asarray(X, dtype=np.float32)
            if dist_matrix.ndim != 2:
                raise ValueError(
                    "metric='precomputed' expects a 2D distance matrix (m_new, n_existing)."
                )
            n_existing = self._embedding.shape[0]
            if dist_matrix.shape[1] != n_existing:
                raise ValueError(
                    f"X.shape[1]={dist_matrix.shape[1]} must equal "
                    f"n_existing={n_existing} for metric='precomputed'."
                )
            m = dist_matrix.shape[0]
            if m == 0:
                return (
                    np.empty((0, k), dtype=np.int32),
                    np.empty((0, k), dtype=np.float32),
                    0,
                )

            actual_k = min(k, n_existing)
            sorted_idx = np.argsort(dist_matrix, axis=1)[:, :actual_k].astype(np.int32)
            sorted_dist = np.take_along_axis(dist_matrix, sorted_idx.astype(np.intp), axis=1)

            # Pad to k columns if n_existing < k
            if actual_k < k:
                pad_idx = np.full((m, k - actual_k), -1, dtype=np.int32)
                pad_dist = np.full((m, k - actual_k), np.inf, dtype=np.float32)
                sorted_idx = np.concatenate([sorted_idx, pad_idx], axis=1)
                sorted_dist = np.concatenate([sorted_dist, pad_dist], axis=1)

            return sorted_idx, sorted_dist, m

        else:
            raise RuntimeError(f"Unsupported metric {self.metric!r}")

    def _position_new_points(
        self,
        new_indices: NDArray[np.int32],
    ) -> NDArray[np.float32]:
        """Position new points near their nearest existing neighbor (tree parent).

        We place each new point at its parent plus a small offset pointing
        toward the local neighborhood centroid. We can't run a full
        force-directed layout here because that would push the new point
        far from its branch and create crossing edges that break the MST
        visual. A simple offset keeps edges short and looks clean.
        """
        m = new_indices.shape[0]
        existing = self._embedding  # (n, 2)
        new_coords = np.empty((m, 2), dtype=np.float32)

        centroid = existing.mean(axis=0)
        coord_range = existing.max(axis=0) - existing.min(axis=0)
        jitter_scale = coord_range * 0.001  # tiny jitter to avoid exact overlaps

        # We need a sense of how far apart connected nodes are in the
        # embedding so offset distances look proportional.
        # Measure the typical embedding distance between each node and its
        # nearest KNN neighbor, then take the median.
        local_scale = 1.0
        if self._graph is not None and self._graph.indices.shape[0] > 0:
            nn_idx = self._graph.indices[:, 0]
            valid = nn_idx >= 0
            if valid.any():
                diffs = existing[valid] - existing[nn_idx[valid]]
                nn_dists = np.linalg.norm(diffs, axis=1)
                local_scale = float(np.median(nn_dists)) or 1.0

        rng = np.random.default_rng(self.seed)

        for i in range(m):
            idxs = new_indices[i][new_indices[i] >= 0]

            if len(idxs) == 0:
                # No valid neighbors found, drop at the global centroid
                new_coords[i] = centroid
            else:
                # nearest neighbor becomes the tree parent
                parent_coord = existing[idxs[0]]

                if len(idxs) >= 2:
                    # Point the offset toward the centroid of the next
                    # few neighbors so the new point sits on the correct
                    # "side" of its parent branch.
                    nb_coords = existing[idxs[1 : min(5, len(idxs))]]
                    direction = nb_coords.mean(axis=0) - parent_coord
                    norm = np.linalg.norm(direction)
                    if norm > 1e-8:
                        direction /= norm
                    else:
                        # Neighbors are on top of each other then pick a random direction
                        direction = rng.normal(0, 1, size=2).astype(np.float32)
                        direction /= np.linalg.norm(direction)
                    new_coords[i] = parent_coord + direction * (local_scale * 0.3)
                else:
                    new_coords[i] = parent_coord

            new_coords[i] += rng.normal(0, 1, size=2).astype(np.float32) * jitter_scale

        return new_coords

    def _extend_tree(
        self,
        new_indices: NDArray[np.int32],
        new_distances: NDArray[np.float32],
        m: int,
    ) -> None:
        """Append each new point to the tree via its nearest existing neighbor."""
        old_tree = self.tree_  # force lazy extraction if needed
        n_existing = old_tree.n_nodes

        new_edges = np.empty((m, 2), dtype=np.int32)
        new_weights = np.empty(m, dtype=np.float32)

        for i in range(m):
            new_node = n_existing + i
            # Connect to nearest valid existing neighbor
            nn_idx = int(new_indices[i, 0]) if new_indices[i, 0] >= 0 else 0
            nn_dist = float(new_distances[i, 0]) if new_indices[i, 0] >= 0 else 1.0
            new_edges[i] = [nn_idx, new_node]
            new_weights[i] = nn_dist

        all_edges = np.concatenate([old_tree.edges, new_edges])
        all_weights = np.concatenate([old_tree.weights, new_weights])

        self._tree = Tree(
            n_nodes=n_existing + m,
            edges=all_edges,
            weights=all_weights,
            root=old_tree.root,
        )

    # Tree exploration convenience methods

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

    def _encode_jaccard(self, X: Any) -> tuple[NDArray[np.uint64], int]:
        """Detect input type for metric='jaccard' and return MinHash signatures.

        Supports three input formats:
        - 2D binary array (n_samples, n_features) ->  batch_from_binary_array
        - list of string sequences ->  batch_from_string_array
        - list of integer sequences ->  batch_from_sparse_binary_array

        Returns:
            (signatures, n_samples)
        """
        if X is None:
            raise ValueError("X cannot be None for metric='jaccard'.")

        encoder = MinHash(num_perm=self.n_permutations, seed=self.minhash_seed)

        # 2D numpy array ->  binary path
        if isinstance(X, np.ndarray):
            binary_matrix = self._coerce_binary_matrix(X)
            n_samples = binary_matrix.shape[0]
            if self.n_neighbors >= n_samples:
                raise ValueError(
                    f"n_neighbors={self.n_neighbors} must be < n_samples={n_samples}"
                )
            signatures = encoder.batch_from_binary_array(binary_matrix)
            del binary_matrix
            return signatures, n_samples

        if not isinstance(X, (list, tuple)) or len(X) < 2:
            raise ValueError(
                "metric='jaccard' expects a 2D binary array or a list of sequences "
                "(at least 2 samples)."
            )

        # List of uniform-length numeric lists (e.g. data.tolist()) -> try binary path
        try:
            arr = np.asarray(X)
            if arr.ndim == 2 and np.issubdtype(arr.dtype, np.number):
                return self._encode_jaccard(arr)  # recurse into the ndarray branch
        except (ValueError, TypeError):
            pass  # ragged lists, mixed types, etc. --> falls through

        n_samples = len(X)
        if self.n_neighbors >= n_samples:
            raise ValueError(
                f"n_neighbors={self.n_neighbors} must be < n_samples={n_samples}"
            )

        # first non-empty element to decide string vs integer
        first_elem = None
        for seq in X:
            if seq:
                first_elem = next(iter(seq))
                break

        if first_elem is not None and isinstance(first_elem, str):
            signatures = encoder.batch_from_string_array(X)
        else:
            signatures = encoder.batch_from_sparse_binary_array(X)

        return signatures, n_samples

    def _coerce_binary_matrix(self, X: Any | None, min_samples: int = 2) -> NDArray[np.uint8]:
        if X is None:
            raise ValueError("X cannot be None for metric='jaccard'.")
        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError(
                "metric='jaccard' expects a 2D binary matrix of shape (n_samples, n_features)."
            )
        if arr.shape[0] < min_samples:
            raise ValueError(f"Need at least {min_samples} samples.")
        if arr.dtype != np.bool_ and not np.issubdtype(arr.dtype, np.number):
            raise ValueError("Binary matrix must contain numeric/boolean values.")
        if arr.shape[0] > 0 and not np.all((arr == 0) | (arr == 1)):
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

    def _coerce_dense_matrix(self, X: Any | None, min_samples: int = 2) -> NDArray[np.float32]:
        if X is None:
            raise ValueError(f"X cannot be None for metric={self.metric!r}.")
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(
                f"metric={self.metric!r} expects a 2D matrix of shape (n_samples, n_features)."
            )
        if arr.shape[0] < min_samples:
            raise ValueError(f"Need at least {min_samples} samples.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Input matrix must contain only finite values.")
        return arr
