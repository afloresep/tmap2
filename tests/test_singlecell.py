"""Tests for tmap.utils.singlecell."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

anndata = pytest.importorskip("anndata")

from tmap.utils.singlecell import (  # noqa: E402
    cell_metadata,
    from_anndata,
    marker_scores,
    obs_to_numeric,
    sample_obs_indices,
    subset_anndata,
)


def _make_adata():
    X = sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0, 3.0, 0.0],
                [0.0, 2.0, 0.0, 4.0],
                [5.0, 0.0, 6.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    obs = pd.DataFrame(
        {
            "celltype": pd.Categorical(["prog", "prog", "diff"]),
            "day": [2, 2, 7],
            "quality": [0.1, 0.2, 0.9],
        },
        index=["c0", "c1", "c2"],
    )
    var = pd.DataFrame(
        {"highly_variable": [True, False, True, False]},
        index=["GeneA", "GeneB", "GeneC", "GeneD"],
    )

    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.obsm["X_pca"] = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=np.float32,
    )
    adata.layers["counts"] = sparse.csr_matrix(
        np.array(
            [
                [10.0, 1.0, 30.0, 0.0],
                [0.0, 20.0, 0.0, 40.0],
                [50.0, 0.0, 60.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    return adata


def test_from_anndata_uses_obsm_representation():
    adata = _make_adata()
    X = from_anndata(adata, use_rep="X_pca")

    assert X.shape == (3, 2)
    assert X.dtype == np.float32
    np.testing.assert_allclose(X, adata.obsm["X_pca"])


def test_from_anndata_falls_back_to_layer_and_hvgs():
    adata = _make_adata()
    X = from_anndata(adata, use_rep="missing_rep", layer="counts", n_top_genes=2)

    assert X.shape == (3, 2)
    assert X.dtype == np.float32
    np.testing.assert_allclose(
        X,
        np.array(
            [
                [10.0, 30.0],
                [0.0, 0.0],
                [50.0, 60.0],
            ],
            dtype=np.float32,
        ),
    )


def test_cell_metadata_preserves_numeric_and_categorical_columns():
    adata = _make_adata()
    meta = cell_metadata(adata, keys=["celltype", "day", "quality"])

    assert set(meta) == {"celltype", "day", "quality"}
    assert meta["celltype"].dtype == object
    assert meta["day"].dtype == np.float64
    assert meta["quality"].dtype == np.float64
    assert meta["celltype"].tolist() == ["prog", "prog", "diff"]


def test_obs_to_numeric_parses_embedded_numbers():
    values = pd.Series(["day_2", "t=7.5", "unknown"], dtype="object")
    out = obs_to_numeric(values)

    assert out is not None
    np.testing.assert_allclose(out[:2], [2.0, 7.5])
    assert np.isnan(out[2])


def test_sample_obs_indices_balanced_and_proportional():
    groups = np.array(["0"] * 6 + ["1"] * 3 + ["2"] * 1)

    proportional = sample_obs_indices(groups, max_items=6, seed=42, mode="proportional")
    balanced = sample_obs_indices(groups, max_items=6, seed=42, mode="balanced")

    prop_groups = groups[proportional]
    bal_groups = groups[balanced]
    assert {g: int((prop_groups == g).sum()) for g in np.unique(prop_groups)} == {
        "0": 4,
        "1": 1,
        "2": 1,
    }
    assert {g: int((bal_groups == g).sum()) for g in np.unique(bal_groups)} == {
        "0": 3,
        "1": 2,
        "2": 1,
    }


def test_subset_anndata_materializes_backed_file(tmp_path):
    adata = _make_adata()
    path = tmp_path / "toy.h5ad"
    adata.write_h5ad(path)

    backed = anndata.read_h5ad(path, backed="r")
    subset = subset_anndata(backed, obs_mask=np.array([True, False, True]))

    assert subset.n_obs == 2
    assert getattr(subset, "isbacked", False) is False
    assert subset.obs_names.tolist() == ["c0", "c2"]
    np.testing.assert_allclose(subset.obsm["X_pca"], adata.obsm["X_pca"][[0, 2]])


def test_marker_scores_dense_mean_expression():
    adata = _make_adata()
    scores = marker_scores(adata, ["GeneA", "GeneC"])

    np.testing.assert_allclose(scores, [2.0, 0.0, 5.5])
    assert scores.dtype == np.float64


def test_marker_scores_raises_when_no_genes_found():
    adata = _make_adata()
    with pytest.raises(ValueError, match="None of the requested genes found"):
        marker_scores(adata, ["Missing1", "Missing2"])
