# TMAP Documentation

TMAP builds a tree-shaped 2D map from high-dimensional data.

Most users only need one pattern:

1. Prepare data.
2. Call `TMAP(...).fit(X)`.
3. Explore the result in a notebook, save it as HTML, or serve it locally.

## Start Here

- [Molecule Tutorial](molecule_tutorial.md)
  Load `cluster_65053.csv`, compute fingerprints and molecular properties, fit a TMAP, and explore it with `TmapViz`.
- [Visualization Guide](visualization_guide.md)
  Learn when to use notebook widgets, HTML export, and `serve()`.
- [API Reference](api_reference.md)
  Every public class and function: `TMAP`, `TmapViz`, the index and graph classes, layout functions, and the chemistry, protein and single-cell helpers.

## Choose Your API

### Estimator API

Use `TMAP(...).fit(X)` when you want the simplest path. After fitting, the same object answers tree queries (`path`, `distance`, `hops`, `distances_from`), places new data (`transform`, `add_points`), and saves itself (`save`, `TMAP.load`).

### Lower-level pipeline

Use `MinHash`, `LSHForest`, `USearchIndex`, `layout_from_knn_graph` and related functions when you want direct control over hashing, indexing, or graph construction. [Notebook 12](../notebooks/12_lower_level_pipeline.ipynb) walks through it end to end.

## Quick Example

Run this from the repository root:

```python
import pandas as pd
from tmap import TMAP
from tmap.utils import fingerprints_from_smiles

df = pd.read_csv("examples/cluster_65053.csv", nrows=3000)
smiles = df["smiles"].tolist()

fps, valid = fingerprints_from_smiles(smiles, fp_type="morgan", radius=2, n_bits=2048, return_valid=True)
smiles = [smi for smi, ok in zip(smiles, valid) if ok]

model = TMAP(metric="jaccard", n_neighbors=20, seed=42).fit(fps)

viz = model.to_tmapviz()
viz.title = "Cluster 65053"
viz.add_smiles(smiles)
viz.write_html("cluster_65053.html")
```

`return_valid=True` drops SMILES that RDKit can't parse, so the SMILES list stays aligned with the fingerprints. In a script, put the code under `if __name__ == "__main__":`, because `fingerprints_from_smiles` starts worker processes.

## Supported Input Paths

| Input | Metric | Good for |
|-------|--------|----------|
| Binary matrix | `jaccard` | Molecular fingerprints and other 0/1 features |
| Lists of feature indices, lists of strings, SciPy sparse matrix | `jaccard` | Sets of tokens, k-mers, very sparse features |
| Dense float matrix | `cosine` | Embeddings where direction matters |
| Dense float matrix | `euclidean` | Embeddings where magnitude matters |
| Distance matrix | `precomputed` | Distances you already computed elsewhere |
| `KNNGraph` | `fit(knn_graph=...)` | Neighbors from FAISS, Annoy, alignment tools and so on |

The first two rows use different search backends (USearch and MinHash + LSHForest). The [API reference](api_reference.md#input-routing) has the details.

## Choose The Right Output

| Goal | Method |
|------|--------|
| Quick notebook exploration | `viz.show()` or `model.plot(...)` |
| Share one file | `viz.write_html(...)` or `model.to_html(...)` |
| Host a large map on a web server | `viz.write_static(...)` |
| Browse a large map locally | `viz.serve(...)` or `model.serve(...)` |
| Figure for a paper | `model.plot_static(...)` |

## Advanced Guides

- [MinHash Guide](minhash_guide.md)
- [LSHForest Guide](lshforest_guide.md)
- [USearch Backend](usearch_backend.md)
- [Graph Guide](graph_guide.md)
- [Layout Guide](layout_guide.md)

## Notebooks

The [`notebooks/`](../notebooks) folder has worked examples for cheminformatics, embeddings, single-cell data, proteins and the notebook widgets. The [README](../README.md#notebooks) lists them all.
