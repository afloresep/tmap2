# TMAP Documentation

Welcome to TMAP - Tree-based visualization for high-dimensional data.

## Getting Started

TMAP creates tree visualizations from high-dimensional data (like molecular fingerprints). Similar items cluster together, revealing the structure in your data.

```python
from tmap import MinHash, LSHForest
from tmap.layout import layout_from_lsh_forest, LayoutConfig

# 1. Encode your data
mh = MinHash(num_perm=128, seed=42)
signatures = mh.batch_from_binary_array(your_fingerprints)

# 2. Build index
lsh = LSHForest(d=128, l=64)
lsh.batch_add(signatures)
lsh.index()

# 3. Create layout
cfg = LayoutConfig()
cfg.k = 20
cfg.kc = 50
x, y, s, t = layout_from_lsh_forest(lsh, cfg)

# x, y are coordinates; s, t are tree edges
```

## Documentation

| Document | Description |
|----------|-------------|
| [MinHash Guide](minhash_guide.md) | **Start here!** Understanding MinHash encoding and choosing the right method for your data. |
| [LSHForest Guide](lshforest_guide.md) | Building the LSH index, query methods, and k-NN graph construction. |
| [Graph Guide](graph_guide.md) | MST construction from k-NN graphs, Tree traversal, and bias_factor tuning. |
| [Layout Guide](layout_guide.md) | Visual, conceptual explanation of how TMAP layout works and parameter tuning. |
| [Visualization Guide](visualization_guide.md) | Creating interactive HTML visualizations with TmapViz. |
| [API Reference](api_reference.md) | Quick reference for functions, classes, and parameters. |

## Key Concepts

### The Pipeline

```
Your Data → MinHash → LSHForest → k-NN Graph → MST → Layout → Visualization
```

1. **MinHash**: Compresses data into compact signatures preserving similarity
2. **LSHForest**: Builds a searchable index for fast neighbor queries
3. **k-NN Graph**: Connects each point to its k nearest neighbors
4. **MST**: Extracts essential tree structure (minimum spanning tree)
5. **Layout**: Arranges tree in 2D using force-directed algorithm

### Which Function to Use?

| Scenario | Function | Why |
|----------|----------|-----|
| Most cases | `layout_from_lsh_forest()` | Best connectivity, matches original TMAP |
| Custom k-NN | `layout_from_knn_graph()` | When you want to inspect/modify k-NN first |
| Custom edges | `layout_from_edge_list()` | For non-LSH graphs |
| Modular pipeline | `ForceDirectedLayout` class | When using MSTBuilder separately |

### Common Parameters

| Parameter | What it does | Typical values |
|-----------|--------------|----------------|
| `k` | Neighbors per point | 10-50 |
| `kc` | Search quality | 20-100 |
| `node_size` | Spread of tree | 1/100 to 1/10 |
| `fme_iterations` | Smoothness | 500-2000 |

## Examples

See `examples/smiles_tmap.py` for a complete molecular visualization example.

## Troubleshooting

### Disconnected points (grid pattern)

Increase connectivity:
```python
cfg.k = 30    # More neighbors
cfg.kc = 100  # Better quality neighbors
```

### Layout not reproducible

Enable deterministic mode:
```python
cfg.deterministic = True
cfg.seed = 42
```

### Layout too slow

Reduce iterations:
```python
cfg.fme_iterations = 500
cfg.mmm_repeats = 1
```
