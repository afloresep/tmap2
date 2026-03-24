"""Matplotlib static plot for TMAP embeddings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray


def plot_static(
    embedding: NDArray[np.float32] | Sequence[Sequence[float]],
    *,
    color_by: str | NDArray[np.generic] | Sequence[Any] | None = None,
    color_map: str | None = None,
    data: Any | None = None,
    edges: NDArray[np.int32] | None = None,
    edge_color: str = "#cccccc",
    edge_alpha: float = 0.3,
    edge_linewidth: float = 0.3,
    point_size: float = 1.0,
    alpha: float = 0.8,
    ax: Any | None = None,
    figsize: tuple[float, float] = (8, 8),
) -> Any:
    """Render an embedding as a static matplotlib scatter plot.

    Parameters
    ----------
    embedding : array-like of shape (n_samples, 2)
        2D coordinates.
    color_by : str, array-like, or None
        Column name in *data*, or a raw array of color values.
    color_map : str or None
        Matplotlib colormap name. Defaults to ``'tab10'`` for categorical
        data and ``'viridis'`` for continuous data.
    data : DataFrame or None
        Metadata DataFrame whose columns can be referenced by *color_by*.
    edges : ndarray of shape (n_edges, 2) or None
        Edge list to draw behind points (e.g. ``model.tree_.edges``).
    edge_color : str, default '#cccccc'
        Color for edge lines.
    edge_alpha : float, default 0.3
        Opacity for edge lines.
    edge_linewidth : float, default 0.3
        Line width for edges.
    point_size : float, default 1.0
        Marker size passed to ``scatter(s=...)``.
    alpha : float, default 0.8
        Point opacity.
    ax : matplotlib Axes or None
        If provided, draw into this axes. Otherwise a new figure is created.
    figsize : tuple, default (8, 8)
        Figure size when *ax* is None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    from tmap.visualization.jupyter import _is_categorical

    coords = np.asarray(embedding, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"embedding must have shape (n_samples, 2); got {coords.shape!r}")

    n = coords.shape[0]
    own_fig = ax is None
    if own_fig:
        _, ax = plt.subplots(figsize=figsize)

    # -- Edges (behind points) -----------------------------------------------
    if edges is not None:
        from matplotlib.collections import LineCollection

        edge_arr = np.asarray(edges, dtype=np.int32)
        segments = coords[edge_arr].reshape(-1, 2, 2)
        lc = LineCollection(
            segments, colors=edge_color, linewidths=edge_linewidth, alpha=edge_alpha
        )
        ax.add_collection(lc)

    # -- Resolve color --------------------------------------------------------
    c_values = None
    categorical = False

    if color_by is not None:
        if isinstance(color_by, str):
            if data is None or color_by not in data.columns:
                raise ValueError(f"color_by={color_by!r} is not a column in data")
            c_values = data[color_by].values
            categorical = _is_categorical(None, data=data, col_name=color_by)
        else:
            c_values = np.asarray(color_by)
            if len(c_values) != n:
                raise ValueError(
                    f"color_by has {len(c_values)} elements but embedding has {n} points"
                )
            categorical = _is_categorical(c_values)

    # -- Scatter --------------------------------------------------------------
    if c_values is not None and categorical:
        cat = pd.Categorical(c_values)
        cmap_name = color_map or "tab10"
        cmap = plt.get_cmap(cmap_name)
        for code, label in enumerate(cat.categories):
            mask = cat.codes == code
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=point_size,
                alpha=alpha,
                color=cmap(code % cmap.N),
                label=label,
                rasterized=True,
            )
        ax.legend(markerscale=3, frameon=False)
    elif c_values is not None:
        cmap_name = color_map or "viridis"
        sc = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=c_values.astype(float),
            cmap=cmap_name,
            s=point_size,
            alpha=alpha,
            rasterized=True,
        )
        ax.figure.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    else:
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=point_size,
            alpha=alpha,
            color="#4a9eff",
            rasterized=True,
        )

    # -- Clean axes (DR convention) -------------------------------------------
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal", adjustable="datalim")

    return ax
