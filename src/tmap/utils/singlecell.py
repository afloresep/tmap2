"""Single-cell RNA-seq utilities for TMAP.

Bridge between the scverse/AnnData ecosystem and TMAP. Provides helpers
to extract matrices, cell metadata, and gene scores from AnnData objects.

Requires ``anndata`` (install via ``pip install tmap[singlecell]``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

if TYPE_CHECKING:
    from anndata import AnnData

logger = logging.getLogger(__name__)


def _to_dense(X: np.ndarray | sparse.spmatrix) -> NDArray[np.float32]:
    """Convert sparse or dense matrix to dense float32."""
    if sparse.issparse(X):
        return np.asarray(X.toarray(), dtype=np.float32)
    return np.asarray(X, dtype=np.float32)


def from_anndata(
    adata: AnnData,
    use_rep: str | None = "X_pca",
    layer: str | None = None,
    n_top_genes: int | None = None,
) -> NDArray[np.float32]:
    """Extract a numeric matrix from an AnnData object for TMAP.

    By default pulls ``adata.obsm['X_pca']`` which is the standard
    representation after ``sc.pp.pca()``. Falls back to ``adata.X``
    if no PCA is available.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (e.g. from scanpy).
    use_rep : str or None
        Key in ``adata.obsm`` to use (e.g. ``'X_pca'``, ``'X_scVI'``).
        If ``None`` or the key is missing, falls back to ``adata.X``
        (or the specified *layer*).
    layer : str or None
        Layer in ``adata.layers`` to use instead of ``adata.X``.
        Only used when *use_rep* is ``None`` or missing.
    n_top_genes : int or None
        If using ``adata.X`` / a layer, subset to the top *n* highly
        variable genes (requires ``adata.var['highly_variable']``).
        Ignored when using an obsm representation.

    Returns
    -------
    ndarray of shape ``(n_cells, n_features)``, dtype ``float32``
        Ready to pass to ``TMAP(metric='cosine').fit(X)``.
    """
    # Try obsm representation first
    if use_rep is not None and use_rep in adata.obsm:
        X = np.asarray(adata.obsm[use_rep], dtype=np.float32)
        logger.info("from_anndata: using obsm[%r] shape %s", use_rep, X.shape)
        return X

    if use_rep is not None and use_rep not in adata.obsm:
        logger.info(
            "from_anndata: %r not in obsm, falling back to %s",
            use_rep,
            f"layers[{layer!r}]" if layer else "X",
        )

    # Use layer or X
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer {layer!r} not found. Available: {list(adata.layers.keys())}")
        raw = adata.layers[layer]
    else:
        raw = adata.X

    # Optionally subset to HVGs
    if n_top_genes is not None and "highly_variable" in adata.var.columns:
        hvg_mask = adata.var["highly_variable"].values
        n_hvg = int(hvg_mask.sum())
        if n_hvg >= n_top_genes:
            raw = raw[:, hvg_mask]
            logger.info("from_anndata: subset to %d HVGs", n_hvg)
        else:
            logger.info(
                "from_anndata: only %d HVGs found (requested %d), using all",
                n_hvg,
                n_top_genes,
            )

    X = _to_dense(raw)
    logger.info("from_anndata: using expression matrix shape %s", X.shape)
    return X


def cell_metadata(
    adata: AnnData,
    keys: Sequence[str] | None = None,
) -> dict[str, NDArray]:
    """Extract cell-level metadata from ``adata.obs`` as arrays.

    Returns a dict compatible with ``TmapViz.add_metadata()`` and
    ``model.plot(color=...)``.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    keys : list[str] or None
        Column names from ``adata.obs`` to extract. ``None`` extracts
        all columns.

    Returns
    -------
    dict[str, ndarray]
        Numeric columns → ``float64``, categorical/string → ``object``.
    """
    obs = adata.obs
    if keys is None:
        keys = list(obs.columns)
    else:
        missing = [k for k in keys if k not in obs.columns]
        if missing:
            raise KeyError(f"Keys not in adata.obs: {missing}. Available: {list(obs.columns)}")

    result: dict[str, NDArray] = {}
    for k in keys:
        col = obs[k]
        if hasattr(col, "cat"):
            # Categorical → string array
            result[k] = col.astype(str).values
        elif np.issubdtype(col.dtype, np.number):
            result[k] = col.values.astype(np.float64)
        else:
            result[k] = col.astype(str).values

    return result


def marker_scores(
    adata: AnnData,
    gene_list: Sequence[str],
    layer: str | None = None,
) -> NDArray[np.float64]:
    """Compute mean expression of a gene set per cell.

    A lightweight alternative to ``sc.tl.score_genes`` — no dependencies
    beyond anndata. Useful for coloring a TMAP by pathway activity or
    marker gene expression.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    gene_list : list[str]
        Gene names (must be in ``adata.var_names``).
    layer : str or None
        Layer to use. ``None`` uses ``adata.X``.

    Returns
    -------
    ndarray of shape ``(n_cells,)``, dtype ``float64``
        Mean expression across the gene set for each cell.
    """
    var_names = list(adata.var_names)
    indices = []
    missing = []
    for g in gene_list:
        if g in var_names:
            indices.append(var_names.index(g))
        else:
            missing.append(g)

    if missing:
        logger.warning(
            "marker_scores: %d/%d genes not found: %s", len(missing), len(gene_list), missing[:5]
        )

    if not indices:
        raise ValueError("None of the requested genes found in adata.var_names")

    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer {layer!r} not found. Available: {list(adata.layers.keys())}")
        raw = adata.layers[layer]
    else:
        raw = adata.X

    subset = raw[:, indices]
    if sparse.issparse(subset):
        scores = np.asarray(subset.mean(axis=1)).ravel()
    else:
        scores = np.mean(subset, axis=1)

    return np.asarray(scores, dtype=np.float64)
