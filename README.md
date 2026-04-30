[![Tests](https://github.com/afloresep/tmap2/actions/workflows/tests.yml/badge.svg)](https://github.com/afloresep/tmap2/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/tmap2)](https://pypi.org/project/tmap2/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

# TMAP2

Tree-based visualization for high-dimensional data. Organizes similar items into interactive tree structures. Ideal for chemical space, protein embeddings, single-cell data, or any high-dimensional dataset.

<table>
  <tr>
    <td><img src="docs/images/enamine.png" alt="Interactive HTML export" width="100%"/></td>
    <td><img src="docs/images/protein-shot.png" alt="AlphaFold protein clusters" width="100%"/></td>
  </tr>
</table>

## Why Trees?

Most dimensionality reduction tools (UMAP, t-SNE) produce point clouds. TMAP produces a **tree**, a connected structure where every point is linked to its neighbors through branches. This makes the layout itself explorable: you can follow branches, trace paths between any two points, and discover how regions connect.

For example, in a TMAP of pet breed images, following the branch from terriers toward cats reveals that the bridge between the two groups runs through chihuahuas and sphynx cats (the bald ones) which is both hilarious and logical; both are small, short-haired, big-eyed. The tree doesn't just cluster similar things it also shows you *how* dissimilar things are connected.

<p align="center">
  <img src="docs/images/breed-tree.gif" alt="Exploring pet breed tree" width="80%"/>
</p>

Because the layout is a tree, you get operations that point clouds can't support:

```python
path = model.path(idx_a, idx_b)        # nodes along the tree path
d = model.distance(idx_a, idx_b)        # sum of edge weights along the path
pseudotime = model.distances_from(idx)  # tree distance from one point to all others
```

## Installation

```bash
pip install tmap2
```

Optional extras:

```bash
pip install rdkit # chemistry helpers (fingerprints_from_smiles, molecular_properties)
pip install jupyter-scatter # notebook interactive widgets
```

> **Note:** The import name is `tmap`, not `tmap2`.

## Quick Start

```python
import numpy as np
from tmap import TMAP

# Binary fingerprints (Jaccard)
X = np.random.randint(0, 2, (1000, 2048), dtype=np.uint8)
model = TMAP(metric="jaccard", n_neighbors=20, seed=42).fit(X)
model.to_html("map.html")
```

```python
# Dense embeddings (cosine / euclidean)
X = np.random.random((1000, 128)).astype(np.float32)
model = TMAP(metric="cosine", n_neighbors=20).fit(X)
new_coords = model.transform(X[:10])
```

```python
# Interactive notebook widget
model.plot(color_by="label", data=df, tooltip_properties=["name", "score"])
```

## Key Features

- **Tree structure**: follow branches, trace paths, compute pseudotime
- **Deterministic**: same input + seed = same output
- **Multiple metrics**: `jaccard`, `cosine`, `euclidean`, `precomputed`
- **Incremental**: `add_points()` and `transform()` for new data
- **Model persistence**: `save()` / `load()`
- **Three viz backends**: interactive HTML, jupyter-scatter, matplotlib

## Visualization

**Interactive HTML**: lasso selection, light/dark theme, filter and search panels, pinned metadata cards, binary mode for large datasets.

**Notebook widgets**:  color switching, categorical filtering, and lasso selection with pandas-backed metadata:

```python
viz = model.to_tmapviz()
viz.add_color_layout("Molecular Weight", mw.tolist(), categorical=False)
viz.add_color_layout("Scaffold", scaffolds, categorical=True, color="tab10")
viz.add_label("SMILES", smiles_list)
viz.show(width=1000, height=620, controls=True)
```

**Static plots** — matplotlib for publication figures: `model.plot_static(color_by=labels)`

## Domain Utilities

Built-in helpers for common scientific workflows:

```python
from tmap.utils.chemistry import fingerprints_from_smiles, molecular_properties
from tmap.utils.proteins import fetch_uniprot, sequence_properties
from tmap.utils.singlecell import from_anndata
```

| Domain | Metric | Utilities |
|--------|--------|-----------|
| Chemoinformatics | `jaccard` | `fingerprints_from_smiles`, `molecular_properties`, `murcko_scaffolds` |
| Proteins | `cosine` / `euclidean` | `fetch_uniprot`, `fetch_alphafold`, `read_fasta`, `sequence_properties` |
| Single-cell | `cosine` / `euclidean` | `from_anndata`, `cell_metadata`, `marker_scores` |
| Generic embeddings | `cosine` / `euclidean` / `precomputed` | No domain utils needed |

## Notebooks

| Notebook | Topic |
|----------|-------|
| [01 Quickstart](notebooks/01_quickstart.ipynb) | Shortest end-to-end walkthrough on a small molecule table |
| [02 Cheminformatics](notebooks/02_cheminformatics.ipynb) | SMILES → fingerprints → interactive molecular map |
| [03 Continuous Embeddings](notebooks/03_continuous_embeddings.ipynb) | Cosine and euclidean on MNIST: when to use each |
| [04 What's New](notebooks/04_new_functionalities.ipynb) | `add_points`, `transform`, tree paths, save/load, external kNN |
| [05 Single-Cell](notebooks/05_single_cell.ipynb) | RNA-seq with PBMC 3k, pseudotime, UMAP comparison |
| [06 FAQ](notebooks/06_faq.ipynb) | Troubleshooting and common questions |
| [07 MinHash Deep Dive](notebooks/07_minhash_deep_dive.ipynb) | Encoding methods and when to use each |
| [08 Notebook Widgets](notebooks/08_jscatter_demo.ipynb) | Coloring, tooltips, lasso selection with jupyter-scatter |
| [09 Card Configuration](notebooks/09_card_configuration.ipynb) | Pinned card layout, fields, and links |
| [10 Protein Analysis](notebooks/10_protein_analysis.ipynb) | FASTA, ESM embeddings, AlphaFold |
| [11 USearch Jaccard](notebooks/11_usearch_jaccard.ipynb) | Native binary Jaccard backend (high recall, low memory) |
| [12 Legacy LSH Pipeline](notebooks/12_legacy_lsh_pipeline.ipynb) | Lower-level MinHash + LSHForest + layout workflow |

## Lower-Level Pipeline

For direct control over indexing, hashing, and layout, see the [legacy pipeline notebook](notebooks/12_legacy_lsh_pipeline.ipynb). The main building blocks:

```python
from tmap.index import USearchIndex           # dense / binary kNN
from tmap import MinHash, LSHForest           # Jaccard on sets / strings
from tmap.layout import LayoutConfig, layout_from_lsh_forest
```

```text
Your Data
   ├─→ Binary matrix ─────────→ USearch        (Jaccard / cosine / euclidean)
   └─→ Sets / strings ───────→ MinHash → LSHForest
                ↓
             k-NN Graph → MST → OGDF Tree Layout → Interactive Visualization
```

## Development

```bash
git clone https://github.com/afloresep/tmap2.git
cd tmap2
pip install ".[dev]"
pytest -v
```

## License

MIT License - see [LICENSE](LICENSE) for details.

Based on the original [TMAP](https://github.com/reymond-group/tmap) by Daniel Probst and Jean-Louis Reymond.
