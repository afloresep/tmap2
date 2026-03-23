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
  Quick reference for `TMAP`, `TmapViz`, chemistry helpers, and layout functions.

## Choose Your API

### Estimator API

Use `TMAP(...).fit(X)` when you want the simplest path.

### Lower-level pipeline

Use `MinHash`, `LSHForest`, `layout_from_lsh_forest`, and related functions when you want direct control over hashing, indexing, or graph construction.

## Quick Example

```python
import pandas as pd
from tmap import TMAP
from tmap.utils import fingerprints_from_smiles

df = pd.read_csv("../examples/cluster_65053.csv", nrows=3000)
smiles = df["smiles"].tolist()

fps = fingerprints_from_smiles(smiles, fp_type="morgan", radius=2, n_bits=2048)
model = TMAP(metric="jaccard", n_neighbors=20, seed=42).fit(fps)

viz = model.to_tmapviz()
viz.title = "Cluster 65053"
viz.add_smiles(smiles)
viz.write_html("cluster_65053.html")
```

## Supported Input Paths

| Input | Metric | Good for |
|-------|--------|----------|
| Binary matrix | `jaccard` | Molecular fingerprints and other 0/1 features |
| Dense float matrix | `cosine` | Embeddings where direction matters |
| Dense float matrix | `euclidean` | Embeddings where magnitude matters |
| Distance matrix | `precomputed` | Distances you already computed elsewhere |

## Choose The Right Output

| Goal | Method |
|------|--------|
| Quick notebook exploration | `viz.to_widget(...)` or `model.plot(...)` |
| Share one file | `viz.write_html(...)` or `model.to_html(...)` |
| Browse a large map locally | `viz.serve(...)` or `model.serve(...)` |

## Advanced Guides

- [MinHash Guide](minhash_guide.md)
- [LSHForest Guide](lshforest_guide.md)
- [Graph Guide](graph_guide.md)
- [Layout Guide](layout_guide.md)

The files in `docs/` that are not linked here are planning notes or historical material and may not track the current public API line by line.
