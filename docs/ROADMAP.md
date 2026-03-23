# TMAP2 Roadmap

**Last updated:** 2026-03-12

This document defines the future direction for TMAP2. Each task is self-contained:
an agent or contributor with no prior context should be able to pick up any section
and execute it.

---

## Strategic Goal

Make TMAP a general-purpose dimensionality reduction + exploration tool that competes
with t-SNE and UMAP on ease of use while offering strictly more: an explorable tree
structure with paths between points. The tree is the differentiator but should not be
the first thing users see — the default experience is a familiar scatter plot.

**Positioning:** "Like UMAP, but you can trace paths between any two points."

**Operating priorities:**
- Keep the sklearn-style high-level API (`TMAP.fit`, `TMAP.fit_transform`) as the default user path.
- Keep implementation simple; prefer short, clear solutions over layered abstractions.
- Preserve power-user paths in parallel (custom KNN, lower-level modules).
- Make notebook usability a first-class product goal (regl-scatterplot + jupyter-scatter).
- Domain utilities (chemistry, proteins, singlecell) are first-class public API.

---

## Architecture Overview (current state)

The pipeline has 3 stages, each independent and swappable:

```txt
1. Index    ─ Nearest-neighbor search     ─ src/tmap/index/
2. Layout   ─ k-NN -> OGDF MST + layout   ─ src/tmap/layout/
3. Viz      ─ Coordinates -> interactive   ─ src/tmap/visualization/
```

Supporting modules:
```txt
src/tmap/graph/     ─ Tree type, MSTBuilder (internal), analysis utilities
src/tmap/utils/     ─ Domain utilities: chemistry, proteins, singlecell, structural
```

**Computation backend:** Numba JIT for encoding/indexing (MinHash, LSHForest).
USearch for cosine/euclidean kNN. OGDF (via pybind11) for layout.

**Visualization backend:** regl-scatterplot (WebGL) for HTML, jupyter-scatter for
notebooks, matplotlib for static plots. Outputs self-contained HTML.
Edge rendering via canvas overlay synchronized with the scatter view.

**Public API today** (`src/tmap/__init__.py`):
- Core: `TMAP`, `MinHash`, `WeightedMinHash`, `LSHForest`
- Chemistry: `fingerprints_from_smiles`, `molecular_properties`, `murcko_scaffolds`
- Proteins: `fetch_uniprot`, `fetch_alphafold`, `read_pdb`, `sequence_properties`, `parse_alignment`, `read_fasta`, `read_id_list`, `read_pdb_dir`, `read_protein_csv`
- Single-cell: `from_anndata`, `cell_metadata`, `marker_scores`

**Supported metrics:**
| Metric | Backend | Input |
|--------|---------|-------|
| `jaccard` (default) | MinHash + LSHForest | Binary matrix |
| `cosine` | USearch | Dense float matrix |
| `euclidean` | USearch | Dense float matrix |
| `precomputed` | Direct KNNGraph | Dense distance matrix |

**Build system:** scikit-build-core + pybind11 for the OGDF C++ extension only.
OGDF is optional — the `OGDF_AVAILABLE` flag gates layout exports.

---

## Completed (shipped)

### P0-2: Sklearn-style Estimator API — Done

`TMAP.fit()`, `TMAP.fit_transform()`, `embedding_`, `tree_`, `graph_`, `lsh_forest_`.
All four metrics supported. `knn_graph=` power-user path available.

### P0-3: Accept precomputed KNN graphs and distance matrices — Done

`KNNGraph.from_arrays()`, `KNNGraph.from_distance_matrix()`, `TMAP(metric='precomputed').fit(D)`.

### P0-4: jupyter-scatter integration — Done

`TMAP.plot()` delegates to `to_jscatter()`. Interactive scatter in notebooks.
Requires `pip install tmap[notebook]`.

### P0-5: Matplotlib static plot method — Done

`TMAP.plot_static()` and `visualization/static.py`. Edges, categorical/continuous coloring,
colorbar/legend. Returns matplotlib Axes.

### P0-6: Clean up advertised-but-unimplemented extras — Done

Unused optional ANN extras were removed. Dense metrics now use the built-in USearch backend.

### P0-8: LSH parameter sensitivity (ISS-018) — Resolved by design

Default `l=d//2`. Estimator auto-selects `l`. Disconnected graphs for well-separated
data are correct behavior.

### P1-1: Dense metric support (cosine, euclidean) — Done

`USearchIndex` wraps USearch. Small datasets use exact search. Larger datasets use HNSW.
Tests live in `tests/test_ann_backends.py`.

### P1-2: Graph exploration API — Done

`Tree.path()`, `Tree.distance()`, `Tree.subtree()`, `Tree.distances_from()`.
Estimator wrappers: `model.path()`, `model.distance()`, `model.distances_from()`.
31 tests. Tutorial: `notebooks/05_single_cell.ipynb`.

### Domain utilities — Done

- `tmap.utils.chemistry`: `fingerprints_from_smiles`, `molecular_properties`, `murcko_scaffolds`
- `tmap.utils.proteins`: UniProt/AlphaFold fetch, PDB parsing, sequence properties, alignment parsing
- `tmap.utils.singlecell`: `from_anndata`, `cell_metadata`, `marker_scores`

### Graph analysis module — Done

`tmap.graph.analysis`: `boundary_edges`, `confusion_matrix_from_tree`, `edge_delta`,
`path_properties`, `node_diversity`, `subtree_purity`.

### Incremental insertion — Done

`TMAP.add_points(X)` extends the embedding, tree, and graph. Dense insertion also updates the stored USearch index after each batch so later dense additions can see earlier added points.

### Model persistence — Done

`TMAP.save(path)` / `TMAP.load(path)` via pickle.

### Visualization enhancements — Done

- Binary gzip encoding (default for all output)
- NaN handling (warning + black fallback)
- Edge rendering (canvas overlay)
- `TmapViz.serve()` HTTP server for large datasets
- `add_images()`, `add_protein_ids()` tooltip support
- Sorted categorical legends, theme toggle, resize button

---

## P0 — Remaining work for initial release

### P0-7: `layout='graph'` — Removed (deferred)

Graph mode was removed from the estimator. The SGD-based graph layout produced good
quality given exact kNN, but MinHash+LSH kNN recall bottleneck made end-to-end quality
insufficient. Tree layout is TMAP's differentiator.

**Status:** Closed. May revisit if kNN recall improves.

---

## P1 — Important for adoption

### P1-3: Pure-Python layout fallback

**Problem:** When OGDF is not installed, ALL layout functionality is unavailable.
There is no fallback. `ForceDirectedLayout` exists in `layout/force_directed.py`
but the estimator does not use it.

**Goal:** A pure-Python force-directed layout that works without OGDF. Slower,
simpler, but functional. `pip install tmap` without building OGDF gives users
a working pipeline.

**Implementation:**
1. Wire `ForceDirectedLayout` (or a new spring layout) into the estimator as fallback.
2. Auto-select: OGDF if available, Python fallback otherwise.
3. Performance target: usable for trees up to ~50k nodes.

---

## P2 — Nice to have, build after core is solid

### P2-1: `transform(new_X)` — sklearn API completeness

**Status:** Done.

**Semantics:**
- `add_points()` is implemented (mutates model state, returns new coordinates).
- `transform()` is non-mutating and returns coordinates only.
- `transform()` reuses the same neighbor-query / placement logic as
  `add_points()` but does not modify ``embedding_``, ``tree_``,
  ``graph_``, or the ANN/LSH index state.

### P2-2: Edge weight-based visual styling

**Tracked as:** TODO-006 in `ISSUES.md`.

Edge thickness/opacity proportional to similarity weight.

### P2-3: Domain-specific tutorials (NOT adapters)

Show TMAP working on non-chemistry data. Tutorials, not library code.
Domain adapters should NOT be in the core library.

**Current notebooks:**
- `03_legacy_lsh_pipeline.ipynb` — Lower-level MinHash + LSHForest workflow
- `08_cheminformatics.ipynb` — Estimator-first molecule workflow on `cluster_65053.csv`
- `11_card_configuration.ipynb` — Pinned-card configuration on molecule metadata
- `09_protein_analysis.ipynb` — Protein sequences, AlphaFold
- `05_single_cell.ipynb` — Single-cell RNA-seq, pseudotime

**Remaining:**
- NLP / text embeddings tutorial
- Image datasets tutorial (CLIP/ResNet embeddings)

### P2-4: Speed benchmarks

**Goal:** Published benchmarks comparing TMAP vs UMAP vs t-SNE.

Benchmark tooling exists in `benchmarks/`. Subprocess isolation implemented.
Metrics: runtime, trustworthiness, k-NN preservation.

**Remaining:**
- Repeated runs + aggregate statistics
- Memory measurements
- Publish baseline results table

---

## Deferred / Not doing

### Graph mode
Removed. May return if kNN recall improves. See P0-7.

### Domain adapters in core
RDKit, AnnData, sentence-transformers should never be tmap dependencies.
Provide tutorials (P2-3) instead.

### Custom visualization framework
The current stack (regl-scatterplot for HTML, jupyter-scatter for notebooks,
matplotlib for static) covers all use cases. Do not build a custom framework.

---

## Dependency Policy

| Dependency | Status | Notes |
|---|---|---|
| numpy, scipy | Required | Core computation |
| datasketch | Required | Core install includes `WeightedMinHash` support |
| numba | Required | Numba-accelerated encoding/indexing |
| xxhash | Required | Core install includes string-token MinHash support |
| jinja2 | Required | HTML visualization templates |
| matplotlib | Required | Colormaps + static plots |
| pandas | Required | Data handling |
| usearch | Required | Dense ANN backend for cosine/euclidean |
| jscatter | Optional (`[notebook]`) | Jupyter widget |
| pybind11 + OGDF | Optional (build-time) | Force-directed layout |

**Rule:** `pip install tmap` installs core dependencies.
Everything else is opt-in via extras.

---

## Issue cross-reference

| Roadmap task | Related issues in ISSUES.md |
|---|---|
| P1-3 | Not yet tracked |
| P2-1 | TODO-005, ISS-022 |
| P2-2 | TODO-006 |
| P2-3 | — |
| P2-4 | — |
