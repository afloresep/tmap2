# Understanding LSH Forest: A Visual Guide

This guide explains how LSH Forest works in TMAP and helps you choose the right query methods and parameters.

---

## What LSH Forest Does

LSH Forest is an **index structure** for fast approximate nearest neighbor search. It takes MinHash signatures and enables rapid similarity queries.

Comparing every item with every other item takes O(n²) time. LSH Forest sorts slices of the MinHash signatures so that likely neighbors end up next to each other, and then only compares those.

**When TMAP uses it:** `TMAP(metric="jaccard").fit(X)` builds an LSH Forest for lists of feature indices, lists of string tokens and SciPy sparse matrices. Dense 0/1 matrices go to USearch instead (see the [USearch guide](usearch_backend.md)). The estimator picks `l` itself and uses its own `n_permutations` (512) and `kc` (50); after fitting, the forest is available as `model.lsh_forest_`. This guide covers using the class directly.

```txt
MinHash Signatures
        ↓
    [LSH Forest Index]
        ↓
Fast Neighbor Queries → k-NN Graph → TMAP Layout
```

---

## Quick Start

```python
from tmap import MinHash, LSHForest

# 1. Encode your data
mh = MinHash(num_perm=128, seed=42)
signatures = mh.batch_from_binary_array(fingerprints)

# 2. Build the index
lsh = LSHForest(d=128, l=32)  # d must match num_perm
lsh.batch_add(signatures)
lsh.index()  # Don't forget this!

# 3. Build k-NN graph (main output for TMAP)
knn = lsh.get_knn_graph(k=20, kc=50)
```

---

## How LSH Forest Works (Conceptually)

Think of LSH Forest as maintaining several sorted indexes over different parts of each signature.

### Adaptive Prefix Search

1. **Split**: Each signature is split into `l` consecutive bands.
2. **Sort**: Each band becomes a prefix tree represented by lexicographically sorted row IDs.
3. **Query**: Search for complete-band matches first. If there are too few candidates, shorten every tree prefix by one MinHash value and repeat.
4. **Refine**: Rank the collected candidates by their full MinHash-signature distance.

```txt
Signature (128 values)
    ↓
Split into l=32 bands (4 values each)
    ↓
Tree 0: sort values 0..3
Tree 1: sort values 4..7
Tree 2: sort values 8..11
...
    ↓
Query: try prefix lengths 4, 3, 2, 1 until enough candidates exist
```

**Why adaptive prefixes?** A strict full-band collision is often too rare. Backoff preserves selective matches when they exist but still returns useful candidates for moderately similar data.

---

## Parameters

### `d` - Signature Dimensionality

**Must match your MinHash `num_perm`.**

```python
mh = MinHash(num_perm=128)  # Creates 128-dimensional signatures
lsh = LSHForest(d=128)      # d=128 to match
```

### `l` - Number of Prefix Trees (Bands)

Controls the **recall vs speed** tradeoff.

| Configuration | Band width | Tradeoff |
|---------------|------------|----------|
| `d=128, l=8` | 16 | Constructor default (`l=None` means `min(8, d)`); lowest index memory |
| `d=128, l=16` | 8 | Fewer trees, less memory |
| `d=128, l=32` | 4 | What `TMAP` picks for up to a million points |
| `d=128, l=64` | 2 | Aggressive, usually unnecessary with prefix backoff |

**Rule of thumb:** Start with `l=d//4`, the same band width of 4 that `TMAP` uses (it was the best setting on 100k molecules), and tune `kc` next. Memory grows linearly with `l`. Above a million points `TMAP` widens the bands gradually (width 5 at about 2M points, 6 at about 4M).

If `d` is not a multiple of `l`, the last `d % l` values of each signature are not indexed. For example, `d=128, l=48` gives bands of 2 values and leaves 32 values out.

```python
# For a 50k molecule dataset
lsh = LSHForest(d=128, l=32)
```

### `store` - Signature Storage

Controls whether the methods that look up signatures by ID are available. The prefix search keeps the band values either way, so `store=False` saves no memory.

| Value | Capabilities |
|-------|--------------|
| `True` (default) | Full functionality: queries, distances, k-NN graph |
| `False` | `query(signature, k)` only |

```python
# Prefix candidate queries only
lsh = LSHForest(d=128, store=False)

# Full functionality (recommended)
lsh = LSHForest(d=128, store=True)
```

**Note:** With `store=False`, `get_knn_graph()`, `query_by_id()`, `query_linear_scan()`, `query_linear_scan_by_id()`, `query_external_batch()`, `linear_scan()`, `get_distance_by_id()`, `get_all_distances()` and `get_hash()` all raise `ValueError`.

### `weighted` - Weighted MinHash Support

Set to `True` when using `WeightedMinHash` signatures.

```python
# For regular MinHash (binary fingerprints)
lsh = LSHForest(d=128, weighted=False)

# For WeightedMinHash (float vectors)
lsh = LSHForest(d=128, weighted=True)
```

**Important:** Weighted signatures have shape `(d, 2)` vs regular `(d,)`.

---

## Query Methods: Which One to Use?

Which query method to use depends on whether the point is already in the index and whether you need distances.

### Decision Tree

```txt
What do you need?
│
├─ k-NN graph of ALL points → get_knn_graph()  (main method)
│
├─ Query a NEW point (not in index)?
│   │
│   ├─ Need accurate distances? → query_linear_scan()
│   │
│   ├─ Many new points at once? → query_external_batch()
│   │
│   └─ Just need candidates? → query()
│
└─ Query an EXISTING point (by ID)?
    │
    ├─ Need accurate distances? → query_linear_scan_by_id()
    │
    └─ Just need candidates? → query_by_id()
```

### Method Comparison

| Method | Returns | Accuracy | Speed | Use Case |
|--------|---------|----------|-------|----------|
| `get_knn_graph(k, kc)` | KNNGraph | High | Fast | **Build TMAP layout** |
| `query_linear_scan(sig, k, kc)` | [(dist, idx), ...] | High | Medium | Query new point |
| `query_linear_scan_by_id(id, k, kc)` | [(dist, idx), ...] | High | Medium | Query existing point |
| `query_external_batch(sigs, k, kc)` | (indices, distances) | High | Fast | Query many new points |
| `query(sig, k)` | int32 array | Low | Fast | Get candidates only |
| `query_by_id(id, k)` | int32 array | Low | Fast | Get candidates only (includes the point itself) |
| `linear_scan(sig, indices, k)` | [(dist, idx), ...] | Exact | Varies | Refine candidates |

---

## Method Details

### `get_knn_graph(k, kc)`: The Primary Method

**Use this for building TMAP visualizations.**

```python
# Build k-NN graph for layout
knn = lsh.get_knn_graph(k=20, kc=50)

# knn.indices: shape (n_points, k) - neighbor indices
# knn.distances: shape (n_points, k) - distances to neighbors

print(f"Found neighbors for {knn.n_nodes} points")
print(f"Each point has up to {knn.k} neighbors")
```

**Parameters:**

- `k`: Number of neighbors per point
- `kc`: Search quality multiplier (searches `k * kc` candidates, keeps best `k`)

**Returns:** `KNNGraph` object with `indices` and `distances` arrays.

**Note:** Self is excluded from neighbors. Point 0's neighbors won't include 0.

---

### `query_linear_scan(signature, k, kc)`: Query New Point

**For querying a point not in the index.**

```python
# Encode a new molecule
new_fp = load_new_fingerprint()
new_sig = mh.from_binary_array(new_fp)

# Find its neighbors in the indexed set
neighbors = lsh.query_linear_scan(new_sig, k=10, kc=20)

for distance, idx in neighbors:
    print(f"Neighbor {idx}: distance = {distance:.3f}")
```

**How it works:**

1. LSH query finds `k * kc` candidates
2. Linear scan computes exact distances to candidates
3. Returns best `k` sorted by distance

For many new points, `query_external_batch(signatures, k, kc)` does the same in one call and returns `(indices, distances)` arrays of shape `(m, k)`, padded with `-1` and `inf` where fewer than `k` neighbors were found. `TMAP.transform()` and `add_points()` use it on this path.

---

### `query_linear_scan_by_id(id, k, kc)`: Query Existing Point

**For querying a point that's already indexed.**

```python
# Find neighbors of point 42
neighbors = lsh.query_linear_scan_by_id(42, k=10, kc=20)

# Self (42) is automatically excluded
for distance, idx in neighbors:
    print(f"Neighbor {idx}: distance = {distance:.3f}")
```

---

### `query(signature, k)`: Fast Approximate Query

**For getting candidate indices without distance computation.**

```python
# Get candidates (no distances, may include false positives)
candidates = lsh.query(signature, k=100)
print(f"Found {len(candidates)} candidates")
```

**Note:** Results come back as an int32 array, not sorted by distance. Use `linear_scan()` to refine. `query_by_id()` works the same way for an indexed point, and the point itself is among its results.

---

### `linear_scan(signature, indices, k)`: Refine Candidates

**For computing exact distances on a subset.**

```python
# Get rough candidates
candidates = lsh.query(signature, k=100)

# Refine with exact distances
results = lsh.linear_scan(signature, candidates, k=10)

for distance, idx in results:
    print(f"Neighbor {idx}: distance = {distance:.3f}")
```

---

## Understanding k and kc

These parameters control the **connectivity vs speed** tradeoff.

### `k` - Number of Neighbors

How many neighbors to keep per point.

| k | Effect | Use Case |
|---|--------|----------|
| 5-10 | Sparse graph, may disconnect | Quick exploration |
| 20-30 | Good connectivity | Most cases |
| 50-100 | Very dense graph | When connectivity is critical |

**Tip:** If your map falls apart into separate islands, increase `k` or `kc`. `TMAP` itself joins the pieces with bridging edges by default (`connect_components=True`) and reports how many there were in `model.n_components_`; the low-level `layout_from_lsh_forest` leaves them separate.

### `kc` - Search Quality Multiplier

How many candidates to consider before keeping best `k`. Every `LSHForest` method defaults to `kc=10`; `TMAP` uses `kc=50`. The candidate count `k * kc` is capped at the number of indexed points.

| kc | Candidates Searched | Quality | Speed |
|----|---------------------|---------|-------|
| 10 | k × 10 | Approximate | Fast |
| 50 | k × 50 | Good | Good |
| 100 | k × 100 | High | Slower |

**Why kc matters:** LSH is approximate. Higher `kc` means checking more candidates, increasing the chance of finding true nearest neighbors.

```python
# Fast but may miss some neighbors
knn = lsh.get_knn_graph(k=20, kc=10)  # Searches 200 candidates

# Thorough search
knn = lsh.get_knn_graph(k=20, kc=100)  # Searches 2000 candidates
```

---

## Distance Methods

### Static Distance Functions

```python
# Jaccard distance between two signatures
dist = LSHForest.get_distance(sig_a, sig_b)

# Weighted Jaccard distance (for WeightedMinHash)
dist = LSHForest.get_weighted_distance(sig_a, sig_b)
```

### Instance Methods (Require Indexing)

```python
# Distance between indexed points
dist = lsh.get_distance_by_id(0, 1)

# Distances from a signature to ALL indexed points
distances = lsh.get_all_distances(query_signature)
print(distances.shape)  # (n_indexed,)
```

---

## Persistence

Save and load your index to avoid recomputation.

```python
# Save
lsh.save("my_index.pkl")

# Load
lsh = LSHForest.load("my_index.pkl")

# Ready to query immediately
knn = lsh.get_knn_graph(k=20, kc=50)
```

**What's saved:**

- All signatures
- Sorted prefix-tree row IDs
- Configuration (d, l, weighted, store)

---

## Workflow Patterns

### Pattern 1: Build Once, Query Many

```python
# Build index (expensive)
lsh = LSHForest(d=128, l=32)
lsh.batch_add(all_signatures)
lsh.index()
lsh.save("index.pkl")

# Later: load and query (fast)
lsh = LSHForest.load("index.pkl")
knn = lsh.get_knn_graph(k=20, kc=50)
```

### Pattern 2: Incremental Updates

```python
# Initial index
lsh = LSHForest(d=128, l=32)
lsh.batch_add(initial_signatures)
lsh.index()

# Add more data
lsh.batch_add(new_signatures)
lsh.index()  # Must re-index after adding

# Query the full index
knn = lsh.get_knn_graph(k=20, kc=50)
```

### Pattern 3: Query New Points

```python
# Build index of known compounds
lsh = LSHForest(d=128, l=32)
lsh.batch_add(known_signatures)
lsh.index()

# Later: find similar compounds for a new molecule
new_sig = mh.from_binary_array(new_fingerprint)
neighbors = lsh.query_linear_scan(new_sig, k=10, kc=50)
```

---

## Common Pitfalls

### 1. Forgetting to Call `index()`

```python
# Wrong: index() not called
lsh = LSHForest(d=128)
lsh.batch_add(signatures)
knn = lsh.get_knn_graph(k=20, kc=50)  # RuntimeError

# Right
lsh = LSHForest(d=128)
lsh.batch_add(signatures)
lsh.index()
knn = lsh.get_knn_graph(k=20, kc=50)
```

### 2. Mismatched Dimensions

```python
# Wrong: d doesn't match the signature dimension
mh = MinHash(num_perm=128)
signatures = mh.batch_from_binary_array(fps)  # Shape: (n, 128)

lsh = LSHForest(d=64)       # should be 128
lsh.batch_add(signatures)   # ValueError

# Right
lsh = LSHForest(d=128)  # Match num_perm
```

### 3. Not Re-indexing After Adding

```python
# After adding more data, index is stale
lsh.batch_add(new_signatures)
assert not lsh.is_clean  # True - needs re-indexing

# Re-index before querying
lsh.index()
assert lsh.is_clean  # Now it's clean
```

Querying a stale index does not raise. `get_knn_graph()` and the other queries quietly answer from the old index, so points added since the last `index()` are missing from the results.

### 4. Using store=False Then Expecting Full Functionality

```python
# Wrong: store=False limits functionality
lsh = LSHForest(d=128, store=False)
lsh.batch_add(signatures)
lsh.index()
knn = lsh.get_knn_graph(k=20, kc=50)  # ValueError

# Right: if you need a k-NN graph, use store=True (the default)
lsh = LSHForest(d=128, store=True)
```

### 5. Mixing Regular and Weighted Signatures

```python
# Wrong: weighted mode mismatch
wmh = WeightedMinHash(dim=100, num_perm=128)
sigs = wmh.batch_from_weight_array(weights, n_jobs=1)  # Shape: (n, 128, 2)

lsh = LSHForest(d=128, weighted=False)  # should be weighted=True
lsh.batch_add(sigs)  # ValueError

# Right
lsh = LSHForest(d=128, weighted=True)
```

---

## KNNGraph Object

The `get_knn_graph()` method returns a `KNNGraph` object:

```python
knn = lsh.get_knn_graph(k=20, kc=50)

# Properties
print(knn.n_nodes)    # Number of points
print(knn.k)          # Neighbors per point

# Arrays
print(knn.indices.shape)    # (n_nodes, k)
print(knn.distances.shape)  # (n_nodes, k)

# Access neighbors of point 0
point_0_neighbors = knn.indices[0]
point_0_distances = knn.distances[0]
```

To get a tree from it, use `tmap.graph.tree_from_knn_graph(knn)`; to lay it out, use `tmap.layout.layout_from_knn_graph(knn)`; or pass it to `TMAP().fit(knn_graph=knn)` for a full map.

**Special values:**
- `indices[i, j] == -1`: No valid neighbor (sparse)
- `distances[i, j] >= 2.0`: Invalid distance marker

---

## Performance Tips

### 1. Batch Add Is Faster

```python
# Fast
lsh.batch_add(all_signatures)
lsh.index()

# Slow
for sig in all_signatures:
    lsh.add(sig)
lsh.index()
```

### 2. Index Once, Query Many

```python
# Index once
lsh.index()

# Can query many times without re-indexing
knn1 = lsh.get_knn_graph(k=10, kc=50)
knn2 = lsh.get_knn_graph(k=20, kc=50)
knn3 = lsh.get_knn_graph(k=30, kc=100)
```

### 3. Tune Candidate Budget Before Adding Trees

```python
# Same band width as TMAP
lsh = LSHForest(d=128, l=32)

# Search more candidates before increasing index memory
knn = lsh.get_knn_graph(k=20, kc=100)
```

---

## Complete Example

```python
import numpy as np
from tmap import MinHash, LSHForest
from tmap.layout import layout_from_lsh_forest, LayoutConfig

# 1. Generate/load fingerprints
n_molecules = 10000
n_bits = 2048
fps = (np.random.rand(n_molecules, n_bits) < 0.1).astype(np.uint8)

# 2. MinHash encoding
mh = MinHash(num_perm=128, seed=42)
signatures = mh.batch_from_binary_array(fps)

# 3. Build LSH Forest
lsh = LSHForest(d=128, l=32)
lsh.batch_add(signatures)
lsh.index()

# 4. Verify index state
print(f"Indexed {lsh.size} signatures")
print(f"Index is clean: {lsh.is_clean}")

# 5. Build k-NN graph
knn = lsh.get_knn_graph(k=20, kc=50)
print(f"k-NN graph: {knn.n_nodes} nodes, {knn.k} neighbors each")

# 6. Create layout
cfg = LayoutConfig()
cfg.k = 20
cfg.kc = 50
cfg.deterministic = True  # same coordinates on every run
cfg.seed = 42
x, y, s, t = layout_from_lsh_forest(lsh, cfg)

print(f"Layout complete: {len(x)} points, {len(s)} edges")

# 7. Save for later
lsh.save("molecular_index.pkl")
```

---

## Next Steps

- See [MinHash Guide](minhash_guide.md) for encoding options
- See [Layout Guide](layout_guide.md) for visualization parameters
- Run the lower-level pipeline end to end in [notebook 12](../notebooks/12_lower_level_pipeline.ipynb)
