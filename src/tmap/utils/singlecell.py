"""Single-cell RNA-seq utilities for TMAP.

Bridge between the scverse/AnnData ecosystem and TMAP. Provides helpers
to extract matrices, subset AnnData objects, sample observations, parse
observation columns, and compute lightweight marker scores.

Requires ``anndata`` (install via ``pip install anndata``).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

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


def _group_quotas(counts: NDArray[np.int64], max_items: int, mode: str) -> NDArray[np.int64]:
    """Allocate sample sizes across groups without replacement."""
    if mode == "proportional":
        quotas = np.floor(max_items * counts / counts.sum()).astype(np.int64)
        quotas = np.minimum(np.maximum(quotas, 1), counts)

        while quotas.sum() > max_items:
            idx = int(np.argmax(quotas))
            if quotas[idx] > 1:
                quotas[idx] -= 1
        while quotas.sum() < max_items:
            remaining = counts - quotas
            idx = int(np.argmax(remaining))
            if remaining[idx] == 0:
                break
            quotas[idx] += 1
        return quotas

    if mode == "balanced":
        quotas = np.zeros_like(counts, dtype=np.int64)
        remaining = int(max_items)
        active = np.arange(len(counts), dtype=np.int64)

        while remaining > 0 and len(active) > 0:
            share = max(1, remaining // len(active))
            new_active: list[int] = []
            changed = False
            for idx in active:
                capacity = int(counts[idx] - quotas[idx])
                if capacity <= 0:
                    continue
                take = min(capacity, share)
                if take > 0:
                    quotas[idx] += take
                    remaining -= take
                    changed = True
                if quotas[idx] < counts[idx]:
                    new_active.append(int(idx))
                if remaining == 0:
                    break
            if not changed:
                break
            active = np.asarray(new_active, dtype=np.int64)

        if remaining > 0:
            remaining_cap = counts - quotas
            for idx in np.argsort(-remaining_cap):
                take = min(int(remaining_cap[idx]), remaining)
                if take <= 0:
                    continue
                quotas[idx] += take
                remaining -= take
                if remaining == 0:
                    break
        return quotas

    raise ValueError(f"Unknown sampling mode: {mode!r}")


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
        If using ``adata.X`` / a layer and ``adata.var['highly_variable']``
        exists, use the marked highly variable genes when at least
        *n_top_genes* are available. Ignored when using an obsm
        representation.

    Returns
    -------
    ndarray of shape ``(n_cells, n_features)``, dtype ``float32``
        Ready to pass to ``TMAP(metric='cosine').fit(X)``. Expression
        matrices and layers are densified before return, so for large
        sparse inputs it is better to subset first or use a precomputed
        obsm representation.
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
            logger.info(
                "from_anndata: subset to %d highly variable genes (requested at least %d)",
                n_hvg,
                n_top_genes,
            )
        else:
            logger.info(
                "from_anndata: only %d HVGs found (requested %d), using all",
                n_hvg,
                n_top_genes,
            )

    X = _to_dense(raw)
    logger.info("from_anndata: using expression matrix shape %s", X.shape)
    return X


def subset_anndata(
    adata: AnnData,
    *,
    obs_mask: Sequence[bool] | NDArray[np.bool_] | None = None,
    obs_indices: Sequence[int] | NDArray[np.int64] | None = None,
    copy: bool = True,
) -> AnnData:
    """Subset observations in one step.

    This is mainly useful for backed ``AnnData`` objects, where repeated
    slicing into views is not allowed. Backed inputs are materialized with
    ``to_memory()`` after subsetting.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    obs_mask : sequence[bool] or None
        Boolean mask over observations.
    obs_indices : sequence[int] or None
        Explicit observation indices.
    copy : bool
        Whether to return a copy for in-memory AnnData inputs.

    Returns
    -------
    AnnData
        Subsetted object. Backed inputs return an in-memory ``AnnData``.
    """
    if obs_mask is not None and obs_indices is not None:
        raise ValueError("Provide either obs_mask or obs_indices, not both.")

    if obs_mask is None and obs_indices is None:
        if getattr(adata, "isbacked", False):
            return adata.to_memory()
        return adata.copy() if copy else adata

    if obs_mask is not None:
        mask = np.asarray(obs_mask, dtype=bool)
        if mask.ndim != 1 or len(mask) != adata.n_obs:
            raise ValueError("obs_mask must be a 1D boolean array of length adata.n_obs.")
        idx = np.flatnonzero(mask)
    else:
        idx = np.asarray(obs_indices, dtype=np.int64)
        if idx.ndim != 1:
            raise ValueError("obs_indices must be a 1D array.")

    subset = adata[np.sort(idx)]
    if getattr(adata, "isbacked", False):
        return subset.to_memory()
    return subset.copy() if copy else subset


def sample_obs_indices(
    groups: Sequence[object] | NDArray,
    *,
    max_items: int,
    seed: int,
    mode: str = "proportional",
) -> NDArray[np.int64]:
    """Sample observation indices from groups without replacement.

    Parameters
    ----------
    groups : sequence
        Group label per observation.
    max_items : int
        Maximum number of observations to keep.
    seed : int
        Random seed for group-wise sampling.
    mode : {"proportional", "balanced"}
        ``"proportional"`` preserves group frequencies as much as
        possible. ``"balanced"`` spreads observations more evenly across
        groups, capped by the number available in each group.

    Returns
    -------
    ndarray of shape ``(n_kept,)``, dtype ``int64``
        Sorted indices into the original array.
    """
    groups_arr = np.asarray(groups).astype(str)
    if groups_arr.ndim != 1:
        raise ValueError("groups must be a 1D array.")
    if max_items <= 0:
        raise ValueError("max_items must be positive.")
    if len(groups_arr) <= max_items:
        return np.arange(len(groups_arr), dtype=np.int64)

    unique, counts = np.unique(groups_arr, return_counts=True)
    quotas = _group_quotas(counts.astype(np.int64), max_items, mode)
    rng = np.random.default_rng(seed)

    keep_parts: list[np.ndarray] = []
    for group, quota in zip(unique, quotas, strict=True):
        idx = np.where(groups_arr == group)[0]
        chosen = np.sort(rng.choice(idx, size=int(quota), replace=False))
        keep_parts.append(chosen)

    return np.sort(np.concatenate(keep_parts))


def obs_to_numeric(values: Sequence[object] | NDArray) -> NDArray[np.float32] | None:
    """Convert an observation-like array to numeric values when possible."""
    arr = np.asarray(values)

    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(np.float32, copy=False)

    out = np.full(arr.shape[0], np.nan, dtype=np.float32)
    for i, value in enumerate(arr.astype(str)):
        digits = "".join(ch for ch in value if ch.isdigit() or ch in ".-")
        if digits and digits not in {"-", ".", "-."}:
            try:
                out[i] = float(digits)
            except ValueError:
                continue

    if np.isnan(out).all():
        return None
    return out


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
    var_index = {str(name): i for i, name in enumerate(adata.var_names)}
    indices = []
    missing = []
    for g in gene_list:
        idx = var_index.get(g)
        if idx is not None:
            indices.append(idx)
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
