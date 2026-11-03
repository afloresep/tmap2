# TMAP API Reference

This is a practical reference for the public API. For a guided workflow, start with [molecule_tutorial.md](molecule_tutorial.md).

## `TMAP`

High-level estimator for building a tree-shaped 2D map. Import it with `from tmap import TMAP`.

### Constructor

```python
TMAP(
    n_neighbors=20,
    metric="jaccard",          # "jaccard", "cosine", "euclidean" or "precomputed"
    n_permutations=512,        # MinHash only (sets, strings, sparse input)
    kc=50,                     # LSHForest only (sets, strings, sparse input)
    seed=42,                   # layout seed
    minhash_seed=42,           # MinHash seed
    layout_iterations=1000,
    layout_config=None,
    store_index=False,
    reproducible=False,
    connect_components=True,
)
```

| Parameter | What it does |
|-----------|--------------|
| `n_neighbors` | Neighbors per point in the kNN graph |
| `metric` | Distance to use. See [Input routing](#input-routing) for which inputs each metric takes |
| `n_permutations` | MinHash signature length. Only used when the input goes through MinHash + LSHForest |
| `kc` | LSHForest search multiplier: each query collects `n_neighbors * kc` candidates and ranks them exactly. Larger values give better recall and cost query time. Only used on the LSHForest path |
| `seed` | Seed for the OGDF layout |
| `minhash_seed` | Seed for the MinHash permutations |
| `layout_iterations` | Force-directed iterations. Ignored when you pass `layout_config` |
| `layout_config` | A `LayoutConfig` for fine control of the layout. The estimator sets its `deterministic` and `seed` fields in place; everything else is yours. See [layout_guide.md](layout_guide.md) |
| `store_index` | Keep the cosine/euclidean index after `fit()`, which `kneighbors()`, `transform()` and `add_points()` need. Binary Jaccard keeps its index regardless |
| `reproducible` | Build the HNSW index on one thread so repeated runs give identical coordinates. Slower (roughly 5 to 7 times for the HNSW build at 10k+ points) |
| `connect_components` | If the kNN graph falls apart into several pieces, add the cheapest edges that join them so the result is one tree. `False` keeps the forest |

`metric` is case-insensitive.

### Fitting

```python
model = TMAP(metric="cosine").fit(X)                   # from data
model = TMAP().fit(knn_graph=KNNGraph.from_arrays(indices, distances))  # from your own neighbors
x, y, s, t = TMAP(metric="cosine").fit_transform(X)    # coordinates and tree edges
```

`fit()` takes either `X` or `knn_graph`, never both. With `knn_graph`, TMAP skips the neighbor search.

### Methods

| Method | What it does |
|--------|---------------|
| `fit(X=None, *, knn_graph=None)` | Build the graph, tree and 2D embedding. Returns the model |
| `fit_transform(X=None, *, knn_graph=None)` | Fit and return `(x, y, s, t)`: coordinates plus the tree edges as source and target arrays |
| `kneighbors(X, *, return_distance=True)` | Nearest fitted points for new data. The model doesn't change |
| `transform(X)` | Coordinates for new points, placed next to their nearest fitted neighbors. The model doesn't change |
| `add_points(X)` | Place new points and append them to the embedding, tree and graph. Returns their coordinates |
| `path(a, b)` | Node indices along the tree path from `a` to `b` |
| `distance(a, b)` | Sum of edge weights along that path |
| `distances_from(i)` | Tree distance from point `i` to every point |
| `hops(a, b)` | Number of tree edges between `a` and `b` |
| `hops_from(i)` | Number of tree edges from point `i` to every point |
| `to_tmapviz(include_edges=True)` | A `TmapViz` with the coordinates and tree edges filled in |
| `to_html(path, *, title=None, include_edges=True)` | Write an HTML map and return its path |
| `serve(port=8050, include_edges=True, **kwargs)` | Serve the map on a local HTTP server. Extra arguments go to `TmapViz.serve()` |
| `plot(...)` | Notebook widget through jupyter-scatter. Takes `color_by`, `color_map`, `data`, `tooltip_properties` and size options |
| `plot_static(...)` | Matplotlib figure. Takes `color_by`, `color_map`, `data`, `edges=True` and styling options. Returns the `Axes` |
| `save(path)` | Save the fitted model, including any stored index |
| `TMAP.load(path)` | Class method that loads a model written by `save()` |

`kneighbors`, `transform` and `add_points` need the fitted index. For cosine and euclidean that means `store_index=True`; without it they raise `RuntimeError`. `add_points` only works for binary matrix input under Jaccard. If `fit()` got sets or strings, it raises `TypeError`, although `transform` and `kneighbors` still work.

### Attributes

| Attribute | Meaning |
|-----------|---------|
| `embedding_` | 2D coordinates, shape `(n, 2)`, float32 |
| `tree_` | The fitted `Tree` (see [Graph](#graph-tmapgraph)) |
| `graph_` | The raw kNN graph as the search returned it, before any components were joined |
| `n_components_` | Number of connected pieces in the raw kNN graph. `1` means it was already connected, `None` means `connect_components=False` |
| `n_bridges_` | Number of edges added to join those pieces |
| `index_` | The USearch index (binary Jaccard, or cosine/euclidean with `store_index=True`). Raises otherwise |
| `lsh_forest_` | The LSHForest (sets, strings and sparse input). Raises otherwise |

### Input routing

`fit()` chooses the neighbor search from the metric and the input type:

| Metric | Input | Backend |
|--------|-------|---------|
| `jaccard` | Dense binary matrix (0/1 NumPy array, bool array, DataFrame) | USearch, HNSW |
| `jaccard` | Lists of feature indices, lists of string tokens, SciPy sparse matrix | MinHash + LSHForest |
| `cosine` | Dense float matrix | USearch, exact below 50,000 points, HNSW above |
| `euclidean` | Dense float matrix | USearch, exact below 50,000 points, HNSW above |
| `precomputed` | Square distance matrix | Neighbors read from the matrix |
| any | `knn_graph=KNNGraph(...)` | Your neighbors, used as given |

### Reproducibility

The OGDF layout always runs seeded and deterministic, so the same kNN graph and the same `seed` give identical coordinates.

The kNN step depends on the backend:

- **MinHash + LSHForest**: deterministic. The permutations come from `minhash_seed`.
- **USearch exact search** (cosine/euclidean below 50,000 points): deterministic.
- **USearch HNSW** (binary Jaccard, and cosine/euclidean from 50,000 points): built on several threads by default, so neighbors at equal distance can come back in a different order between runs. That can change the tree and the layout. Pass `reproducible=True` to build on one thread and get identical results every time.

## `TmapViz`

Interactive map object. You usually get one from `model.to_tmapviz()`. To build one from your own coordinates, use `from tmap.visualization import TmapViz` and call `set_points()` and optionally `set_edges()`.

### Points and edges

| Method | What it does |
|--------|---------------|
| `set_points(x, y)` | Set the coordinates |
| `set_edges(s, t, weights=None)` | Set the tree edges. Weights feed the neighbor list in the HTML inspector |
| `set_edge_style(color=None, width=None, opacity=None)` | Edge color, width in CSS pixels, and opacity |

### Columns

| Method | What it does |
|--------|---------------|
| `add_color_layout(name, values, categorical=False, add_as_label=True, color=None)` | A column you can color by. Also filterable, and shown in tooltips unless `add_as_label=False` |
| `add_filter(name, values, categorical=False, add_as_label=True)` | A filter-panel column without a color map. Raises if `name` is already a color layout |
| `add_label(name, values)` | A column shown in tooltips and cards |
| `add_smiles(values, name="SMILES")` | SMILES strings, drawn as 2D structures in tooltips and cards |
| `add_images(values, name="image", tooltip_size=128)` | One image per point as a `data:image/...;base64,...` URI |
| `add_protein_ids(values, name="UniProt ID")` | UniProt accessions. Cards load the AlphaFold structure and UniProt metadata on demand |
| `add_3d_structures(values, *, source="url", fmt="pdb", name="structure")` | PDB or mmCIF structures, either as URLs (`source="url"`) or as file text (`source="text"`) |
| `add_3d_structure_files(paths, *, copy=True, directory="structures", fmt="pdb", name="structure")` | Local structure files. With `copy=True`, `write_html` and `write_static` copy them into `directory` next to the page |

For `add_color_layout`, `color` is a matplotlib colormap name. Continuous layouts take a gradient colormap (default `viridis`). Categorical layouts take a qualitative one such as `tab10` (the default), `tab20`, `Set1` or `Dark2`, or a list or dict of hex colors. Mixing them up raises an error.

### Panels and cards

| Member | What it does |
|--------|---------------|
| `filterable` | Columns in the filter panel. Empty by default, which means every `add_filter` column plus every color layout. Assigning a list sets the exact columns |
| `searchable` | Columns for text search. Empty by default, which means all label columns |
| `configure_column(name, *, display_name=None, link_template=None, copyable=None, format=None)` | Per-column display hints. `link_template` takes `{column name}` placeholders. `format` accepts `"fixed:N"`, `"percent:N"`, `"scientific:N"`, `"integer"` and `"stars:N"` |
| `configure_card(*, title_column=None, subtitle_column=None, fields=None, links=None)` | What the pinned card shows. `links` is a list of `{"label": ..., "url": ...}` dicts with `{column name}` placeholders |

### Output

| Method | What it does |
|--------|---------------|
| `show(*, width=800, height=420, controls=True)` | Display a jupyter-scatter widget in a notebook |
| `to_widget(*, width=800, height=420, controls=True)` | Build the same widget without displaying it |
| `to_dataframe(*, include_coords=True, ...)` | All columns as a pandas DataFrame, one row per point |
| `to_html()` | The HTML page as a string |
| `write_html(path)` | Write a single HTML file with the data embedded. If `path` is a directory, the file is named after `title` |
| `write_static(output_dir)` | Write the page and its data as separate files for a static web host. Better for very large maps |
| `serve(port=8050, open_browser=True)` | Write static files to a temporary folder, serve them, and open a browser. Blocks until Ctrl+C |

The notebook widget needs the `notebook` extra (`pip install "tmap2[notebook]"`). It draws points only and warns if the map has edges. The HTML page loads a few JavaScript libraries from a CDN, so viewers need internet access.

Page settings are plain attributes: `title`, `background_color`, `point_color`, `point_size`, `opacity`, `edge_color`, `edge_width` and `edge_opacity`. Read-only helpers: `n_points`, `layouts` and `labels`.

### Minimal example

```python
viz = model.to_tmapviz()
viz.title = "My Map"
viz.add_color_layout("Score", scores.tolist(), color="viridis")
viz.add_color_layout("Group", groups, categorical=True, color="tab10")
viz.add_label("Name", names)
viz.write_html("my_map.html")
```

### Static figures

`tmap.visualization.plot_static(embedding, *, color_by=None, color_map=None, data=None, edges=None, ...)` draws any 2D embedding with matplotlib and returns the `Axes`. Pass `edges=model.tree_.edges` to draw the tree. `model.plot_static()` does the same for a fitted model.

## Graph (`tmap.graph`)

### `Tree`

`model.tree_` returns a `Tree`. It stores `n_nodes`, `edges` (shape `(n - 1, 2)` for a connected tree), `weights`, and `root` (the node with the most edges).

| Method | What it does |
|--------|---------------|
| `neighbors(node)` | `[(neighbor, weight), ...]` |
| `children(node, parent=None)` | Neighbors other than `parent` |
| `bfs(start=None)` / `dfs(start=None)` | Iterate `(node, parent, depth)` from `start` (default `root`) |
| `subtree(node, depth=None)` | Nodes within `depth` edges of `node`, closest first. `depth=None` returns the whole connected component. Despite the name, it walks outward in every direction, including toward the root |
| `subtree_sizes()` | Size of each node's subtree when rooted at `root` |
| `path(a, b)` / `distance(a, b)` / `hops(a, b)` | Same as on the estimator |
| `distances_from(i)` / `hops_from(i)` | Same as on the estimator. Unreachable nodes get `inf` or `-1` |

### Analysis

| Function | Returns |
|----------|---------|
| `boundary_edges(tree, labels)` | Indices of edges whose two ends have different labels |
| `confusion_matrix_from_tree(tree, labels)` | `(matrix, classes)`: how often each pair of labels is joined by an edge |
| `edge_delta(tree, values)` | Absolute change in `values` across each edge |
| `path_properties(tree, a, b, values)` | `values` along the tree path from `a` to `b` |
| `node_diversity(tree, method="mean")` | Per-node summary (`"mean"`, `"max"` or `"median"`) of the weights of its edges |
| `subtree_purity(tree, labels, min_size=10)` | For each node, the share of its subtree that has its most common label. `NaN` for subtrees smaller than `min_size` |

`tree_from_knn_graph(knn, config=None)` builds a `Tree` straight from a `KNNGraph`.

## Index (`tmap.index`)

### `KNNGraph`

Holds `indices` and `distances`, both of shape `(n, k)`. Missing neighbors have index `-1`.

| Constructor | Use it when |
|-------------|-------------|
| `KNNGraph.from_arrays(indices, distances)` | You have neighbors from FAISS, Annoy or another library |
| `KNNGraph.from_distance_matrix(D, k)` | You have a full distance matrix |

Properties: `n_nodes`, `k`.

### `USearchIndex`

```python
USearchIndex(seed=None, mode="auto", connectivity=32, expansion_add=256, expansion_search=200, threads=0)
```

`mode="auto"` uses exact search below 50,000 dense vectors and HNSW above. Binary Jaccard always uses HNSW. `threads=0` uses every core; `threads=1` makes the build reproducible.

| Method | What it does |
|--------|---------------|
| `build_from_vectors(X, metric="euclidean")` | Build a dense cosine or euclidean index |
| `build_from_binary(X)` | Build a binary Jaccard index |
| `query_point(x, k)` | Query one vector. Returns `(indices, distances)` |
| `query_batch(X, k)` | Query many vectors |
| `query_knn(k)` | All-vs-all kNN graph as a `KNNGraph` |
| `add(X)` | Append vectors |
| `save(path)` / `USearchIndex.load(path)` | Persist the index. `save` writes `path`, `path.meta` and `path.vectors.npy`, and `load` needs all three |

Properties: `n_nodes`, `metric`, `effective_mode` (`"exact"` or `"hnsw"`), `is_built`.

### `LSHForest`

```python
LSHForest(d=128, l=None, store=True, weighted=False)
```

`d` is the signature length and must match the MinHash `num_perm`. `l` is the number of prefix trees and defaults to 8. With `store=False`, only `query()` works; the methods that look up signatures by ID raise. Import it with `from tmap import LSHForest`.

| Method | What it does |
|--------|---------------|
| `add(signature)` / `batch_add(signatures)` | Add signatures. Call `index()` afterwards |
| `index()` | Build the prefix trees. Queries before this use the old index |
| `query(signature, k)` | Candidates from the prefix trees only |
| `query_by_id(id, k)` | Same, for an indexed point. The point itself is among the results |
| `query_linear_scan(signature, k, kc=10)` | Collect `k * kc` candidates and rank them by exact MinHash distance. Returns `[(distance, id), ...]` |
| `query_linear_scan_by_id(id, k, kc=10)` | Same, for an indexed point |
| `query_external_batch(signatures, k, kc=10)` | Batch version for new signatures. Returns `(indices, distances)` padded with `-1` and `inf` |
| `get_knn_graph(k, kc=10)` | All-vs-all kNN graph as a `KNNGraph` |
| `get_all_nearest_neighbors(k, kc=10)` | The same neighbors as a flat index array of length `n * k` |
| `get_all_distances(signature)` | Exact MinHash distances to every indexed signature |
| `get_distance(a, b)` / `get_distance_by_id(a, b)` / `get_weighted_distance(a, b)` | Distance between two signatures |
| `get_hash(id)` | The stored signature for a point |
| `clear()` | Remove everything |
| `save(path)` / `LSHForest.load(path)` | Persist the forest |

Properties: `d`, `l`, `size`, `is_indexed`, `is_clean`.

### `MinHash` and `WeightedMinHash`

Import both from `tmap`.

| Class | Main methods |
|-------|--------------|
| `MinHash(num_perm=128, seed=1)` | `batch_from_binary_array(X)`, `batch_from_sparse_binary_array(index_lists)`, `batch_from_string_array(string_lists)`, the single-item `from_*` versions, `encode(data)`, `MinHash.get_distance(a, b)` |
| `WeightedMinHash(dim, num_perm=128, seed=1)` | `batch_from_weight_array(X, n_jobs=None)`, `from_weight_array(x)`, `encode(X)`, `WeightedMinHash.get_weighted_distance(a, b)` |

To reproduce the signatures that `TMAP` builds, use `MinHash(num_perm=512, seed=42)`. See [minhash_guide.md](minhash_guide.md) for when to use each method.

## Layout (`tmap.layout`)

| Function | Use it when |
|----------|-------------|
| `layout_from_lsh_forest(lsh_forest, config=None, create_mst=True)` | You built an `LSHForest` yourself. Reads `config.k` and `config.kc` for the kNN query |
| `layout_from_knn_graph(knn, config=None, create_mst=True)` | You already have a `KNNGraph` |
| `layout_from_edge_list(vertex_count, edges, config=None, create_mst=True)` | You already have weighted edges as `[(source, target, weight), ...]` |

All three return `(x, y, s, t)`: float32 coordinates scaled to [-0.5, 0.5] and uint32 tree edges.

`LayoutConfig` holds the layout settings. It takes no constructor arguments, so set fields one at a time:

```python
from tmap.layout import LayoutConfig, layout_from_knn_graph

cfg = LayoutConfig()
cfg.deterministic = True
cfg.seed = 42
cfg.untangle = False
x, y, s, t = layout_from_knn_graph(knn, config=cfg)
```

The enums `Placer`, `Merger`, `ScalingType` and `UntangleMode` live in the same module. `OGDF_AVAILABLE` tells you whether the compiled extension loaded. For every field and what it does, see [layout_guide.md](layout_guide.md).

## Utilities (`tmap.utils`)

The functions below are also importable from `tmap` directly.

### Chemistry

Needs RDKit (`pip install "tmap2[chemistry]"`). All four functions run on several worker processes, so scripts that call them need an `if __name__ == "__main__":` guard.

| Function | What it does |
|----------|---------------|
| `fingerprints_from_smiles(smiles, fp_type="morgan", n_workers=None, return_valid=False, **kwargs)` | Fingerprints from SMILES. `fp_type` is `"morgan"` (binary, takes `radius` and `n_bits`), `"map4"`, `"mhfp"` and `"drfp"` (binary), or `"mqn"` and `"mxfp"` (integer counts, use cosine or euclidean). `map4`, `mhfp`, `mxfp` and `drfp` need their own packages |
| `molecular_properties(smiles, properties=None, n_workers=None)` | Dict of RDKit descriptors, `NaN` for SMILES that don't parse. Names are in `AVAILABLE_PROPERTIES` |
| `murcko_scaffolds(smiles, n_workers=None)` | Murcko scaffold strings, `""` for SMILES that don't parse |
| `reaction_properties(rxn_smiles, properties=None, n_workers=None)` | Dict of reaction descriptors. Names are in `AVAILABLE_REACTION_PROPERTIES` |

```python
from tmap.utils import fingerprints_from_smiles, molecular_properties

fps, valid = fingerprints_from_smiles(smiles, fp_type="morgan", radius=2, n_bits=2048, return_valid=True)
smiles = [smi for smi, ok in zip(smiles, valid) if ok]

props = molecular_properties(smiles, properties=["mw", "logp", "n_rings"])
```

`fingerprints_from_smiles` skips any SMILES it cannot read, so it can give back fewer rows than you passed in. `molecular_properties` and `murcko_scaffolds` keep every input and fill the bad ones with `NaN` or `""`. Filter with `valid` as above before computing anything else, or the lists stop matching and points get the wrong colors and labels.

### Proteins

`sequence_properties` and `read_pdb` need Biopython (`pip install "tmap2[proteins]"`). The fetch functions need internet access.

| Function | What it does |
|----------|---------------|
| `read_fasta(path, max_seqs=None)` | `(ids, sequences)` from a FASTA file |
| `read_protein_csv(path, id_col="id", seq_col="sequence")` | `(ids, sequences)` from a CSV or TSV file |
| `read_id_list(path)` | UniProt accessions from a text file, one per line |
| `read_pdb(path, chain=None)` | Sequence and structural properties from one PDB file |
| `read_pdb_dir(directory, pattern="*.pdb", chain=None)` | `(ids, sequences, properties)` for a folder of PDB files |
| `sequence_properties(sequences, properties=None)` | Dict of physicochemical properties. Names are in `AVAILABLE_SEQUENCE_PROPERTIES` |
| `fetch_uniprot(ids, fields=..., chunk_size=50, max_workers=4)` | Annotations from the UniProt REST API |
| `fetch_alphafold(ids, max_workers=20)` | Structure metadata (such as pLDDT) from the AlphaFold DB API |
| `parse_alignment(path, k=20, score_col="bitscore", as_distance=True)` | A kNN graph from BLAST, MMseqs2 or Foldseek tabular output, for `TMAP().fit(knn_graph=...)` |

### Single-cell

Needs anndata (`pip install "tmap2[singlecell]"`).

| Function | What it does |
|----------|---------------|
| `from_anndata(adata, use_rep="X_pca", layer=None, n_top_genes=None)` | A float32 matrix for `TMAP.fit`, from `adata.obsm[use_rep]` or from expression values |
| `cell_metadata(adata, keys=None)` | Columns of `adata.obs` as arrays, ready for `add_color_layout` |
| `marker_scores(adata, gene_list, layer=None)` | Mean expression of a gene set in each cell |
| `obs_to_numeric(values)` | Numeric version of an `obs` column, or `None` if it isn't numeric |
| `subset_anndata(adata, *, obs_mask=None, obs_indices=None, copy=True)` | Subset cells in one step |
| `sample_obs_indices(groups, *, max_items, seed, mode="proportional")` | Sample cells per group without replacement |
