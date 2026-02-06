# TMAP Layout API Reference

Quick reference for the layout module. For conceptual explanations, see [layout_guide.md](layout_guide.md).

---

## Functions

### `layout_from_lsh_forest`

The recommended high-level API. Computes layout directly from an indexed LSHForest.

```python
from tmap.layout import layout_from_lsh_forest, LayoutConfig

x, y, s, t = layout_from_lsh_forest(
    lsh_forest,           # Indexed LSHForest
    config=None,          # LayoutConfig (optional)
    create_mst=True       # Whether to compute MST (default: True)
)
```

**Returns:** `(x, y, s, t)` - coordinates and edge indices as numpy arrays

**Why use this:** Passes the full k-NN graph to OGDF for MST computation, resulting in better connectivity than computing MST separately.

---

### `layout_from_knn_graph`

Layout from a pre-computed k-NN graph.

```python
from tmap.layout import layout_from_knn_graph

knn = lsh.get_knn_graph(k=20, kc=50)
x, y, s, t = layout_from_knn_graph(
    knn,                  # KNNGraph object
    config=None,          # LayoutConfig (optional)
    create_mst=True       # Whether to compute MST (default: True)
)
```

**When to use:** When you want to inspect or modify the k-NN graph before layout.

---

### `layout_from_edge_list`

Low-level function for custom edge lists.

```python
from tmap.layout import layout_from_edge_list

edges = [(0, 1, 0.5), (1, 2, 0.3), ...]  # (source, target, weight)
x, y, s, t = layout_from_edge_list(
    vertex_count,         # Number of vertices
    edges,                # List of (src, tgt, weight) tuples
    config=None,          # LayoutConfig (optional)
    create_mst=True       # Whether to compute MST (default: True)
)
```

**When to use:** When you have a custom graph not from LSHForest.

---

## LayoutConfig

Configuration object for all layout parameters.

```python
from tmap.layout import LayoutConfig, Placer, Merger, ScalingType

cfg = LayoutConfig()
```

### k-NN Parameters

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `k` | int | 10 | Number of nearest neighbors |
| `kc` | int | 10 | Query multiplier (searches k*kc candidates) |

### Force Simulation

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `fme_iterations` | int | 1000 | Number of force simulation iterations |
| `fme_precision` | int | 4 | Multipole expansion precision |
| `node_size` | float | 1/65 | Node size for repulsion calculation |

### Multilevel Algorithm

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `mmm_repeats` | int | 1 | Layout repeats per level |
| `placer` | Placer | Barycenter | Node placement strategy |
| `merger` | Merger | LocalBiconnected | Graph coarsening strategy |
| `merger_factor` | float | 2.0 | Coarsening factor |
| `merger_adjustment` | int | 0 | Edge length adjustment |

### Scaling

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `sl_repeats` | int | 1 | Scaling layout repeats |
| `sl_extra_scaling_steps` | int | 2 | Extra refinement steps |
| `sl_scaling_min` | float | 1.0 | Minimum scale factor |
| `sl_scaling_max` | float | 1.0 | Maximum scale factor |
| `sl_scaling_type` | ScalingType | RelativeToDrawing | Scaling method |

### Reproducibility

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `deterministic` | bool | False | Enable deterministic mode |
| `seed` | int or None | None | Random seed (requires deterministic=True) |

---

## Enums

### `Placer`

Node placement strategy during multilevel uncoarsening.

| Value | Description |
|-------|-------------|
| `Barycenter` | Center of neighbors (default, recommended) |
| `Solar` | Solar system arrangement |
| `Circle` | Circle around barycenter |
| `Median` | Median position of neighbors |
| `Random` | Random placement (non-deterministic) |
| `Zero` | Same position as representative |

### `Merger`

Graph coarsening strategy.

| Value | Description |
|-------|-------------|
| `LocalBiconnected` | Preserves local structure (default, recommended) |
| `EdgeCover` | Edge cover based merging |
| `Solar` | Solar system partitioning |
| `IndependentSet` | GRIP-style (good for large graphs) |

### `ScalingType`

How layout scales during refinement.

| Value | Description |
|-------|-------------|
| `RelativeToDrawing` | Scale relative to current drawing (default) |
| `RelativeToAvgLength` | Scale relative to average edge weight |
| `RelativeToDesiredLength` | Scale relative to desired edge length |
| `Absolute` | Absolute scaling factor |

---

## ForceDirectedLayout Class

Object-oriented interface for the Layout ABC.

```python
from tmap.layout import ForceDirectedLayout

layout = ForceDirectedLayout(
    seed=42,                    # Random seed
    max_iterations=1000,        # fme_iterations
    placer=Placer.Barycenter,   # Placement strategy
    merger=Merger.LocalBiconnected,  # Coarsening strategy
    node_size=1/65,             # Repulsion strength
    mmm_repeats=1,              # Refinement passes
    sl_extra_scaling_steps=2,   # Scaling refinement
    sl_scaling_type=ScalingType.RelativeToDrawing,
    merger_factor=2.0,          # Coarsening factor
    config=None,                # Or pass full LayoutConfig
)

# Compute layout from Tree
coords = layout.compute(tree)
print(coords.x, coords.y)
```

**When to use:** When working with the modular pipeline (MSTBuilder → Layout).

---

## Output Format

All layout functions return numpy arrays:

| Array | Type | Shape | Description |
|-------|------|-------|-------------|
| `x` | float32 | (n_nodes,) | X coordinates, normalized to [-0.5, 0.5] |
| `y` | float32 | (n_nodes,) | Y coordinates, normalized to [-0.5, 0.5] |
| `s` | uint32 | (n_edges,) | Source vertex indices |
| `t` | uint32 | (n_edges,) | Target vertex indices |

Edges connect `s[i]` to `t[i]` for each `i`.

---

## Quick Examples

### Minimal Example

```python
from tmap import MinHash, LSHForest
from tmap.layout import layout_from_lsh_forest

# Encode and index
mh = MinHash(num_perm=128)
lsh = LSHForest(d=128)
lsh.batch_add(mh.batch_from_binary_array(data))
lsh.index()

# Layout with defaults
x, y, s, t = layout_from_lsh_forest(lsh)
```

### With Custom Configuration

```python
from tmap.layout import LayoutConfig, ScalingType

cfg = LayoutConfig()
cfg.k = 30
cfg.kc = 100
cfg.node_size = 1/30
cfg.deterministic = True
cfg.seed = 42

x, y, s, t = layout_from_lsh_forest(lsh, cfg)
```

### Check Connectivity

```python
x, y, s, t = layout_from_lsh_forest(lsh, cfg)

n_nodes = len(x)
n_edges = len(s)
n_components = n_nodes - n_edges  # For MST: components = nodes - edges

if n_components > 1:
    print(f"Warning: {n_components} disconnected components")
    print("Consider increasing k, kc, or l")
```
