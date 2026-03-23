# Understanding the Graph Module

This guide explains how TMAP turns a k-NN graph into a `Tree` and what you can do with that tree afterward.

---

## Overview

The graph module sits between neighbor search and visualization:

```text
Data -> k-NN graph -> MST extraction -> Tree analysis / layout
```

TMAP keeps the public graph API small:

- `tree_from_knn_graph(knn)` computes a minimum spanning tree from a `KNNGraph`
- `Tree` exposes traversal, path, distance, and subtree helpers

The supported MST path is OGDF-based. There is no separate SciPy MST builder anymore.

---

## Quick Start

```python
from tmap.graph import tree_from_knn_graph

tree = tree_from_knn_graph(knn)

print(tree.n_nodes)
print(len(tree.edges))
print(tree.root)
```

Use this when:

- you already have a `KNNGraph`
- you want tree distances or traversal
- you want to inspect connectivity before visualization

If you want coordinates directly, skip the graph module and call `layout_from_knn_graph(knn)` instead.

---

## Why a Tree

A k-NN graph is much denser than the final TMAP structure.

- `n` points with `k` neighbors produce about `n * k` directed edges
- the tree keeps only the minimum set of edges needed to connect each component
- for a connected graph, that means exactly `n - 1` edges

This makes downstream layout and tree analysis practical.

---

## Building a Tree

### From LSHForest

```python
from tmap import LSHForest
from tmap.graph import tree_from_knn_graph

knn = lsh.get_knn_graph(k=20, kc=50)
tree = tree_from_knn_graph(knn)
```

### From another k-NN backend

```python
from tmap.graph import tree_from_knn_graph
from tmap.index.types import KNNGraph

knn = KNNGraph.from_arrays(indices, distances)
tree = tree_from_knn_graph(knn)
```

That second path is the answer for USearch, Annoy, or any external neighbor search:
convert the arrays into `KNNGraph`, then extract the tree with `tree_from_knn_graph`.

---

## What `tree_from_knn_graph()` does

At a high level:

1. Convert the directed k-NN table into an undirected weighted edge list
2. Pass that graph to OGDF with MST creation enabled
3. Recover the returned MST edges as a `Tree`
4. Reattach edge weights from the original k-NN graph

For duplicated directed edges such as `i -> j` and `j -> i`, the smaller observed weight is kept.

---

## Tree Structure

The result is a `Tree` object with:

```python
tree.n_nodes
tree.edges
tree.weights
tree.root
```

`root` is chosen automatically from the highest-degree node in the extracted tree.

---

## Traversal Helpers

### Neighbors

```python
neighbors = tree.neighbors(5)
for neighbor, weight in neighbors:
    print(neighbor, weight)
```

### Breadth-first search

```python
for node, parent, depth in tree.bfs():
    print(node, parent, depth)
```

### Depth-first search

```python
for node, parent, depth in tree.dfs():
    print(node, parent, depth)
```

### Children

```python
children = tree.children(node=5, parent=2)
```

### Subtree sizes

```python
sizes = tree.subtree_sizes()
print(sizes[tree.root])
```

---

## Paths and Distances

The tree API supports path-based analysis directly.

### Shortest path in the tree

```python
path = tree.path(10, 42)
```

### Tree distance

```python
distance = tree.distance(10, 42)
```

### Distances from one source

```python
distances = tree.distances_from(10)
```

### Local subtree

```python
nearby = tree.subtree(10, depth=2)
```

These are useful for pseudotime, branch inspection, and neighborhood summaries.

---

## Disconnected Graphs

If the k-NN graph has multiple components, the extracted tree will also have multiple components.

You can detect that with:

```python
n_components = tree.n_nodes - len(tree.edges)
```

Common reasons:

- `k` is too small
- `kc` is too small
- the data genuinely has separated groups

Possible responses:

- increase `k`
- increase `kc`
- accept the disconnected structure if it reflects the data honestly

---

## Integration with Layout

### Shortest path: layout directly from k-NN

```python
from tmap.layout import layout_from_knn_graph

x, y, s, t = layout_from_knn_graph(knn, config)
```

### If you already have a Tree

```python
from tmap.layout import layout_from_edge_list

edges = [
    (int(src), int(tgt), float(weight))
    for (src, tgt), weight in zip(tree.edges, tree.weights)
]

x, y, s, t = layout_from_edge_list(tree.n_nodes, edges, config, create_mst=False)
```

`create_mst=False` matters there because the tree is already the MST.

---

## Example

```python
import numpy as np
from tmap import MinHash, LSHForest
from tmap.graph import tree_from_knn_graph

fingerprints = (np.random.rand(1000, 2048) < 0.1).astype(np.uint8)

mh = MinHash(num_perm=128, seed=42)
signatures = mh.batch_from_binary_array(fingerprints)

lsh = LSHForest(d=128, l=64)
lsh.batch_add(signatures)
lsh.index()

knn = lsh.get_knn_graph(k=20, kc=50)
tree = tree_from_knn_graph(knn)

print(f"Nodes: {tree.n_nodes}")
print(f"Edges: {len(tree.edges)}")
print(f"Components: {tree.n_nodes - len(tree.edges)}")

for node, parent, depth in tree.bfs():
    print(node, parent, depth)
    if depth == 2:
        break
```

---

## Related Docs

- [LSHForest Guide](lshforest_guide.md) for building k-NN graphs
- [Layout Guide](layout_guide.md) for coordinates and OGDF parameters
- [API Reference](api_reference.md) for signatures and return types
