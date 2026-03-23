# TMAP API Reference

This is a practical reference for the public API. For a guided workflow, start with [molecule_tutorial.md](molecule_tutorial.md).

## `TMAP`

High-level estimator for building a tree-shaped 2D map.

### Constructor

```python
TMAP(
    n_neighbors=20,
    metric="jaccard",
    n_permutations=512,
    kc=50,
    seed=42,
    layout_iterations=1000,
    layout_config=None,
    store_index=False,
)
```

### Key Methods

| Method | What it does |
|--------|---------------|
| `fit(X)` | Build the graph, tree, and 2D embedding |
| `fit_transform(X)` | Fit and return `(x, y, s, t)` |
| `transform(X_new)` | Place new points on the existing map without changing the model |
| `add_points(X_new)` | Add new points into the fitted model |
| `to_tmapviz()` | Create a `TmapViz` object for notebook or HTML output |
| `to_html(path)` | Write an HTML visualization |
| `serve(port=8050)` | Start a local HTTP server for the visualization |
| `save(path)` / `load(path)` | Save or load the fitted model |

### Key Attributes

| Attribute | Meaning |
|-----------|---------|
| `embedding_` | 2D coordinates, shape `(n, 2)` |
| `tree_` | Tree extracted from the kNN graph |
| `graph_` | k-nearest-neighbor graph |
| `lsh_forest_` | Jaccard search index |
| `index_` | Dense ANN index when `store_index=True` |

### Metrics

| Metric | Input | Backend |
|--------|-------|---------|
| `jaccard` | Binary matrix | MinHash + LSHForest |
| `cosine` | Dense float matrix | USearch |
| `euclidean` | Dense float matrix | USearch |
| `precomputed` | Distance matrix | Direct graph construction |

## `TmapViz`

Visualization object returned by `model.to_tmapviz()`.

### Common Methods

| Method | What it does |
|--------|---------------|
| `add_color_layout(name, values, ...)` | Add a colorable column |
| `add_label(name, values)` | Add a tooltip column |
| `add_smiles(values)` | Add molecule structures to tooltips |
| `to_widget(...)` | Build a Jupyter widget |
| `write_html(path)` | Write one self-contained HTML file |
| `write_static(path)` | Write static assets for hosting |
| `serve(port=8050)` | Start a local HTTP server |

### Minimal Example

```python
viz = model.to_tmapviz()
viz.title = "My Map"
viz.add_color_layout("Score", scores.tolist(), color="viridis")
viz.add_label("Name", names)
viz.write_html("my_map.html")
```

## Chemistry Helpers

These live in `tmap.utils`.

| Function | What it does |
|----------|---------------|
| `fingerprints_from_smiles(smiles, fp_type="morgan", ...)` | Build fingerprints from SMILES |
| `molecular_properties(smiles, properties=None)` | Compute RDKit properties |
| `murcko_scaffolds(smiles)` | Compute Murcko scaffold strings |

### Chemistry Example

```python
from tmap.utils import fingerprints_from_smiles, molecular_properties

fps = fingerprints_from_smiles(smiles, fp_type="morgan", radius=2, n_bits=2048)
props = molecular_properties(smiles, properties=["mw", "logp", "n_rings"])
```

## Power-User Layout API

If you need lower-level control, these functions are still available:

| Function | Use it when |
|----------|-------------|
| `layout_from_lsh_forest(...)` | You want the classic MinHash + LSH workflow directly |
| `layout_from_knn_graph(...)` | You already have a `KNNGraph` |
| `tree_from_knn_graph(...)` | You want a `Tree` from a custom kNN graph |
| `layout_from_edge_list(...)` | You already have weighted edges |

For the parameter details, see [layout_guide.md](layout_guide.md).
