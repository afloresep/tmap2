# Understanding TMAP Layout

This guide explains how TMAP turns neighbors into a 2D tree, what each layout setting does, and which ones are worth changing. It stays at the level of what you see on the map; the OGDF internals are only mentioned where they explain a setting.

---

## The Big Picture: What TMAP Does

TMAP takes high-dimensional data (like molecular fingerprints) and creates a 2D tree visualization where:

- **Similar items are close together**
- **Clusters form branches**
- **The overall structure reveals relationships**

Think of it like creating a family tree, but for your data points based on their similarity.

```txt
Your Data (1000 molecules with 2048-bit fingerprints)
        ↓
    [Neighbor search] → Find the k most similar items for each point
        ↓
    [Graph Building] → Connect neighbors
        ↓
    [Tree Extraction] → Keep the minimum spanning tree
        ↓
    [Layout] → Arrange the tree in 2D space
        ↓
    [Untangle] → Rotate small branches to remove edge crossings
        ↓
    Tree map (x, y coordinates plus tree edges)
```

The neighbor search depends on the input. `TMAP.fit` uses USearch for binary matrices and dense vectors, and MinHash + LSHForest for sets, strings and sparse matrices (see the [API reference](api_reference.md#input-routing)). Steps 1 to 3 below show the MinHash + LSHForest route through the lower-level API, because that is the route where you set the neighbor-search parameters by hand. From Step 4 on, everything applies to every input.

---

## Step 1: Encoding with MinHash

**What it does:** Compresses your data into compact "signatures" that preserve similarity.

```python
from tmap import MinHash

mh = MinHash(num_perm=128, seed=42)
signatures = mh.batch_from_binary_array(fingerprints)
```

### Parameters

| Parameter | What it controls | Trade-off |
|-----------|-----------------|-----------|
| `num_perm` | Signature size (class default: 128; `TMAP` uses 512) | Higher = more accurate similarity, but slower |
| `seed` | Permutation seed (class default: 1; `TMAP` uses `minhash_seed=42`) | The same seed always gives the same signatures |

### How to think about it

Imagine each molecule as a long barcode. MinHash creates a much shorter "summary barcode" that still captures what makes each molecule unique. When two molecules have similar original barcodes, their summary barcodes will also be similar.

**Rule of thumb:** Start with `num_perm=128`. Increase to 256 or 512 if you need more precision for very similar items.

---

## Step 2: Building the LSH Forest Index

**What it does:** Creates a searchable index so we can quickly find similar items.

```python
from tmap import LSHForest

lsh = LSHForest(d=128, l=32)
lsh.batch_add(signatures)
lsh.index()
```

### Parameters

| Parameter | What it controls | Trade-off |
|-----------|-----------------|-----------|
| `d` | Signature dimension | Must equal `num_perm` |
| `l` | Number of prefix trees in the forest (default: 8) | Higher = better recall, but more memory |

### How to think about it

The LSH Forest is like a library catalog that keeps `l` separate card indexes, each sorted by a different slice of the signature. A search looks in every index and pools what it finds. More indexes means fewer similar items slip through the cracks.

**Rule of thumb:** `TMAP` uses `l = d // 4` (at least 8) for up to a million points, so `l=32` for `d=128` and `l=128` for `d=512`. That is a good starting point for the low-level API too.

---

## Step 3: Finding Neighbors (k-NN Graph)

**What it does:** For each item, finds its k most similar neighbors.

With the low-level API this happens inside `layout_from_lsh_forest`, controlled by two `LayoutConfig` fields:

```python
from tmap.layout import LayoutConfig

cfg = LayoutConfig()
cfg.k = 20   # Number of neighbors to keep
cfg.kc = 50  # Search multiplier
```

`cfg.k` and `cfg.kc` are only read by `layout_from_lsh_forest`. The estimator uses its own `n_neighbors` and `kc`, and `layout_from_knn_graph` takes the neighbors you give it.

### Parameters

| Parameter | What it controls | Trade-off |
|-----------|-----------------|-----------|
| `k` | Neighbors per item (default: 10) | Higher = denser graph, better connectivity |
| `kc` | Search depth multiplier (default: 10) | Higher = better quality neighbors, slower |

### How to think about it

Imagine you're at a party trying to find your 20 closest friends (`k=20`). Asking everyone in the room would take all night. So you ask around your general social circle, check maybe 1000 people (`k * kc = 20 * 50 = 1000`), and pick the 20 you're closest to.

**Why kc matters:** LSH is approximate. Checking more candidates (`k * kc`) makes it more likely that the true nearest neighbors are among them.

**Rule of thumb:**

- For exploration: `k=10, kc=10` (fast, might miss some connections)
- For publication quality: `k=20-50, kc=50-100` (slower, better connectivity)

---

## Step 4: The Minimum Spanning Tree (MST)

**What it does:** From the neighbor graph, extracts the essential tree structure.

The k-NN graph has many connections (each node connects to k neighbors). The MST keeps only the connections needed to:

1. Connect all points
2. Minimize total "distance" (maximize similarity)

### Where the MST comes from

The MST is computed by OGDF (the layout engine) on the **full k-NN graph**:

```
k-NN Graph (many edges)  →  MST (n-1 edges)  →  Tree Structure
     ↓                           ↓
   Dense                    Sparse but
  connections              connected
```

If the k-NN graph falls apart into disconnected pieces (some molecules have no path to others), its MST is a forest, and each piece is laid out as a separate island.

The estimator bridges the pieces for you. With the default `connect_components=True`, `TMAP.fit` adds the cheapest edges that join them into one tree and warns that it did so; `model.n_components_` and `model.n_bridges_` tell you how many pieces and bridges there were. The low-level `layout_from_*` functions leave the forest as it is.

**To get fewer pieces in the first place:**

- Increase `k` (more neighbors)
- Increase `kc` (better neighbor quality, LSHForest only)
- Increase `l` in LSHForest (better recall)

---

## Step 5: Force-Directed Layout

**What it does:** Arranges the tree in 2D space so that branches are readable.

OGDF runs a physics simulation:

- **Edges act like springs** (pull connected nodes together)
- **Nodes repel each other** (prevent overlap)
- **The system finds equilibrium** (balanced forces)

It does this on several levels. The tree is first coarsened into smaller and smaller versions, the smallest one is laid out, and then each finer level starts from the coarser layout and relaxes again.

### The adaptive layout (default)

With `adaptive=True` (the default), TMAP sets the repulsion separately for each level. A level with `n` nodes gets a node size of

```
min(ns_cap, ns_coef / n ** ns_exp)
```

so small, coarse levels use `ns_cap`, and very large levels use a smaller size so that millions of points still fit. With the defaults (`ns_cap=0.03`, `ns_coef=2.5`, `ns_exp=0.4`), the cap applies to every level below roughly 60,000 nodes.

```python
cfg = LayoutConfig()
cfg.fme_iterations = 1000  # Simulation steps per level
cfg.ns_cap = 0.03          # Spread
```

#### `ns_cap` (default: 0.03)

**The main spread knob.**

| Value | What the map looks like |
|-------|--------|
| 0.01 | A few long branches with open space between them. Easy to follow individual branches |
| 0.03 | Default. Branches fill a disc evenly |
| 0.1 | Fills the disc even more evenly, with short, bushy branches |

These descriptions come from a 10,000-molecule map; the exact look depends on your data, so try two or three values.

#### `ns_coef` and `ns_exp` (defaults: 2.5 and 0.4)

Control how fast the node size shrinks on large levels. They matter mostly above about 60,000 points. Leave them alone unless a very large map looks crowded or too sparse at the finest level.

#### `quad_rotate` (default: True)

Rotates the simulation frame at each level. Without it, a faint cross can show up through the center of the map.

#### `fme_iterations` (default: 1000)

**Number of simulation steps**

| Value | Effect |
|-------|--------|
| 100-500 | Fast but rough. Good for exploration. |
| 1000 | Default. Good balance. |
| 2000-5000 | Smoother, more refined. Takes longer. |

Think of it like shaking a box of magnets. More iterations = more time to settle into a stable arrangement. The estimator sets this from `layout_iterations` unless you pass your own `layout_config`.

### The legacy layout (`adaptive=False`)

`adaptive=False` switches to the older single-pass multilevel layout from the original TMAP. In this mode two other fields take over, and they are ignored in adaptive mode:

- `node_size` (default 1/65): one repulsion size for all levels. Larger values spread the tree out more.
- `mmm_repeats` (default 1): how many times each level is refined.

Use this mode if you want maps closer to the look of the original TMAP.

---

## Step 6: Untangle

**What it does:** Removes edge crossings after the layout is done.

A force-directed layout usually leaves some branches lying across each other. The untangle pass (`untangle=True` by default) goes through the subtrees of the finished tree, smallest first, and tries turning each one rigidly around the point where it attaches to its parent. It keeps a move only if the total number of crossing tree edges goes down. Edge lengths don't change, and the pass uses no randomness, so the same layout always untangles the same way.

| Field | Default | What it does |
|-------|---------|--------------|
| `untangle` | `True` | Turn the pass on or off |
| `untangle_mode` | `UntangleMode.Rotate` | `Rotate` tries rotations; `Reflect` only mirrors each subtree across the line from its parent |
| `untangle_max_sub` | 2000 | Largest subtree (in points) that may be moved. `0` means no limit |
| `untangle_max_angle` | 90 | Largest turn in degrees. `0` means no limit |
| `untangle_rot_steps` | 64 | Rotation angles tried per subtree, spread around the full circle |
| `untangle_passes` | 4 | Sweeps over all subtrees. Stops early when a sweep changes nothing |
| `untangle_slide_eps` | 0.0 | Lets a subtree also slide along its stem by up to this fraction of the stem length. `0` keeps edge lengths exact |
| `untangle_slide_steps` | 5 | Slide positions tried when `untangle_slide_eps > 0` |

Because it counts crossings only, the pass will sometimes swing a large branch to a spot with fewer crossings that looks worse, for example out to the edge of the map. If that happens, lower `untangle_max_sub` or set `untangle=False`. [`examples/layout/untangle_demo.py`](../examples/layout/untangle_demo.py) shows the pass on and off side by side.

---

## Step 7: Scaling and Final Touches

### `sl_scaling_type`

**How the layout scales during refinement**

| Type | When to use |
|------|-------------|
| `RelativeToDrawing` (default) | Most cases. Scales based on current drawing. |
| `RelativeToAvgLength` | When edge weights vary a lot. Respects average edge length. |
| `RelativeToDesiredLength` | Scales relative to the desired edge length. |
| `Absolute` | Fixed scaling factors (`sl_scaling_min`, `sl_scaling_max`). |

#### `sl_extra_scaling_steps` (default: 2)

Additional refinement steps for scaling. Higher = smoother scaling transitions.

### Output coordinates

The final `x` and `y` are each rescaled to [-0.5, 0.5] on their own, so the returned layout always fills a square. The original aspect ratio is not kept.

---

## Advanced: Placer and Merger

These control the multilevel algorithm's internal behavior.

### Placer (where to put nodes when expanding)

| Type | Description | Best for |
|------|-------------|----------|
| `Barycenter` (default) | Center of neighbors | Most cases |
| `Zero` | Same as parent | Stable, predictable |
| `Median` | Median of neighbors | Outlier-resistant |
| `Circle` | Circle around neighbors | Spreading out |
| `Solar` | Solar system arrangement | Hierarchical data |
| `Random` | Random placement | Exploration. Still reproducible with `deterministic=True` and a seed |

### Merger (how to simplify the graph)

| Type | Description | Best for |
|------|-------------|----------|
| `LocalBiconnected` (default) | Preserves local structure | Most cases |
| `EdgeCover` | Based on edge coverage | Dense graphs |
| `Solar` | Sun/planet hierarchy | Hierarchical data |
| `IndependentSet` | GRIP-style | Very large graphs |

`merger_factor` (default 2.0) sets how much each coarsening step shrinks the graph for `EdgeCover` and `LocalBiconnected`, and the search depth for `IndependentSet`. `Solar` ignores it.

**Rule of thumb:** Stick with defaults (`Barycenter` + `LocalBiconnected`) unless you have a specific reason to change.

---

## Using a LayoutConfig with the estimator

Pass a config to `TMAP` to change the layout while keeping the high-level API:

```python
from tmap import TMAP
from tmap.layout import LayoutConfig

cfg = LayoutConfig()
cfg.ns_cap = 0.015
cfg.untangle_max_sub = 500

model = TMAP(metric="cosine", layout_config=cfg).fit(X)
```

A few things to know:

- The estimator sets `cfg.deterministic = True` and `cfg.seed` to its own `seed`, on the object you passed. Everything else you set is kept.
- `layout_iterations` is ignored when you pass a config; set `cfg.fme_iterations` instead.
- `cfg.k` and `cfg.kc` are ignored; use `n_neighbors` and `kc` on `TMAP`.

---

## Reproducible Layouts

The layout is reproducible when `deterministic=True` and `seed` is set. `deterministic=True` runs the simulation on one thread with a seeded random generator. Without it, the simulation uses all cores and each run comes out a little different, even with a seed. With `deterministic=True` and no seed, the seed is 0.

The estimator always uses a deterministic, seeded layout. If two estimator runs on the same data still give different maps, the difference comes from the neighbor search: the HNSW index is built on several threads. Pass `reproducible=True` to `TMAP` to build it on one thread.

---

## Putting It All Together

Starting points for common situations:

### Quick Exploration

```python
cfg = LayoutConfig()
cfg.fme_iterations = 500
cfg.untangle = False
```

### Clear Branches For Figures

```python
cfg = LayoutConfig()
cfg.ns_cap = 0.015           # longer branches with more space between them
cfg.fme_iterations = 2000
cfg.deterministic = True
cfg.seed = 42
```

### Very Large Datasets (>100k points)

```python
cfg = LayoutConfig()
cfg.fme_iterations = 500             # Lower for speed
cfg.merger = Merger.IndependentSet   # Worth trying on large graphs
```

`Merger` comes from `tmap.layout`, like `LayoutConfig`.

---

## Common Issues and Fixes

### Problem: Separate islands instead of one tree

**Cause:** The k-NN graph has disconnected components.

**Fixes:**

1. With the estimator, keep `connect_components=True` (the default) and check `model.n_components_`
2. Increase `k` (more neighbors per point)
3. Increase `kc` (better quality neighbors, LSHForest only)
4. Increase `l` in LSHForest (better recall)
5. Check if your data naturally has isolated clusters

### Problem: Tree looks too compact or cluttered

**Fixes:**

1. Lower `ns_cap` (e.g., from 0.03 to 0.015) to get fewer, longer branches with more space around them
2. Increase `fme_iterations`

### Problem: Tree looks too sparse, with lots of empty space

**Fixes:**

1. Raise `ns_cap` (e.g., from 0.03 to 0.05)

### Problem: A big branch jumped to a strange place

**Cause:** The untangle pass moved it to reduce crossings.

**Fixes:**

1. Lower `untangle_max_sub` so only smaller subtrees can move
2. Set `untangle = False`

### Problem: Layout changes each run

**Fixes:**

```python
cfg.deterministic = True
cfg.seed = 42  # Any fixed number
```

With the estimator, the layout is already deterministic; pass `reproducible=True` to `TMAP` to make the neighbor search deterministic too.

### Problem: Layout takes too long

**Fixes:**

1. Decrease `fme_iterations` (e.g., 500 instead of 1000)
2. Set `untangle = False`
3. Decrease `k` and `kc`

---

## Complete Example

```python
import numpy as np
from tmap import MinHash, LSHForest
from tmap.layout import layout_from_lsh_forest, LayoutConfig

# Your data: binary fingerprints (n_samples x n_bits)
fingerprints = load_your_data()  # shape: (1000, 2048)

# Step 1: Encode
mh = MinHash(num_perm=128, seed=42)
signatures = mh.batch_from_binary_array(fingerprints)

# Step 2: Index
lsh = LSHForest(d=128, l=32)
lsh.batch_add(signatures)
lsh.index()

# Steps 3 to 7: neighbors, tree, layout and untangle in one call
cfg = LayoutConfig()
cfg.k = 20
cfg.kc = 50
cfg.deterministic = True
cfg.seed = 42

x, y, s, t = layout_from_lsh_forest(lsh, cfg)

# x, y: coordinates for each point
# s, t: edges (s[i] connects to t[i])
print(f"Laid out {len(x)} points with {len(s)} edges")
```

---

## Parameter Reference Card

| Parameter | Default | Effect |
|-----------|---------|--------|
| **k** | 10 | Neighbors per point (`layout_from_lsh_forest` only) |
| **kc** | 10 | Search depth (`layout_from_lsh_forest` only) |
| **fme_iterations** | 1000 | Simulation steps. Higher = smoother |
| **fme_precision** | 4 | Accuracy of the multipole approximation of repulsion |
| **adaptive** | True | Per-level node size. `False` = legacy layout |
| **ns_cap** | 0.03 | Spread (adaptive mode). Lower = longer, more separated branches |
| **ns_coef** | 2.5 | Node-size coefficient for large levels (adaptive mode) |
| **ns_exp** | 0.4 | Node-size exponent for large levels (adaptive mode) |
| **quad_rotate** | True | Rotate the frame per level to avoid a central cross (adaptive mode) |
| **node_size** | 1/65 | Repulsion (legacy mode only) |
| **mmm_repeats** | 1 | Refinement passes (legacy mode only) |
| **sl_repeats** | 1 | Scaling layout repeats |
| **sl_extra_scaling_steps** | 2 | Scaling refinement. Higher = smoother |
| **sl_scaling_type** | RelativeToDrawing | How scaling is measured |
| **sl_scaling_min / sl_scaling_max** | 1.0 / 1.0 | Scaling range |
| **placer** | Barycenter | Where nodes go when a level is expanded |
| **merger** | LocalBiconnected | How the graph is coarsened |
| **merger_factor** | 2.0 | Coarsening step size (see Merger above) |
| **merger_adjustment** | 0 | Edge length adjustment when merging |
| **untangle** | True | Crossing-reduction pass (see Step 6 for its other fields) |
| **deterministic** | False | One thread, seeded random generator. The estimator sets it to True |
| **seed** | None | Random seed. The estimator sets it to its own `seed` |

---

## Next Steps

- Run the lower-level pipeline end to end in [notebook 12](../notebooks/12_lower_level_pipeline.ipynb)
- Compare untangle on and off with `examples/layout/untangle_demo.py`
- Experiment with `ns_cap` on your data
- Check `TmapViz` for interactive visualization options

---

## Related Documentation

- [MinHash Guide](minhash_guide.md): data encoding with MinHash signatures
- [LSHForest Guide](lshforest_guide.md): building the LSH index and k-NN graphs
- [Graph Guide](graph_guide.md): MST extraction and Tree traversal
- [API Reference](api_reference.md): quick reference for all functions
