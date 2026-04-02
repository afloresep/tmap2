# Examples

## Chemistry

| Example | Description | Key features |
|---------|-------------|--------------|
| [`molecules_tmap.py`](molecules_tmap.py) | High-level molecular TMAP from a SMILES CSV | `TMAP(metric="jaccard")`, `fingerprints_from_smiles`, `molecular_properties`, `murcko_scaffolds` |
| [`smiles_tmap.py`](smiles_tmap.py) | Low-level pipeline: MinHash → LSHForest → OGDF layout → TmapViz | `MinHash`, `LSHForest`, `layout_from_lsh_forest`, `LayoutConfig` |
| [`sar_egfr.py`](sar_egfr.py) | SAR navigation for EGFR kinase inhibitors (ChEMBL data) | Activity cliffs, scaffold analysis, SAR paths, `boundary_edges`, `subtree_purity` |

## Image datasets

| Example | Description | Key features |
|---------|-------------|--------------|
| [`mnist_cosine_tmap.py`](mnist_cosine_tmap.py) | MNIST 70k digits with cosine metric | `TMAP(metric="cosine")`, `LayoutConfig`, `model.path()` |
| [`pet_breed_audit.py`](pet_breed_audit.py) | Oxford-IIIT Pets classifier audit with ResNet-50 embeddings | `TMAP(metric="cosine")`, graph analysis, image tooltips, failure path tracing |

## Proteins

| Example | Description | Key features |
|---------|-------------|--------------|
| [`afdb_clusters_tmap.py`](afdb_clusters_tmap.py) | AlphaFold DB: 2.3M structural clusters from Foldseek | Precomputed `KNNGraph`, taxonomy resolution, `node_diversity`, large-scale pipeline |

## Single-cell

| Example | Description | Key features |
|---------|-------------|--------------|
| [`singlecell_trajectory_tmap.py`](singlecell_trajectory_tmap.py) | Murine lung regeneration trajectory from an official AnnData `.h5ad` | `from_anndata`, `cell_metadata`, `marker_scores`, pseudotime via `distances_from()` |
| [`singlecell_reprogramming_tmap.py`](singlecell_reprogramming_tmap.py) | Morris fibroblast-to-iEP direct reprogramming trajectory from an official AnnData `.h5ad` | Backed AnnData filtering, explicit root/target anchors, reference pseudotime comparison |

## Quick start

The fastest way to try TMAP:

```python
from tmap import TMAP, fingerprints_from_smiles

fps = fingerprints_from_smiles(["CCO", "c1ccccc1", "CC(=O)O", ...])
model = TMAP(metric="jaccard").fit(fps)
model.to_html("output.html")
```

## Data files

- `cluster_65053.csv` — ~6k SMILES from an Enamine chemical cluster (used by `molecules_tmap.py` and `smiles_tmap.py`)
- `afdb_cluster_data/` — downloaded automatically by `afdb_clusters_tmap.py`
- `data/` — cached embeddings and datasets (created by examples on first run)
