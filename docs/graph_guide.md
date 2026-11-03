# Understanding the Graph Module

This guide explains how TMAP turns a k-NN graph into a `Tree` and what you can do with that tree afterward.

---

## Overview

The graph module sits between neighbor search and visualization:

```text
Data -> k-NN graph -> MST extraction -> Tree analysis / layout
```

`tmap.graph` has three parts:

- `tree_from_knn_graph(knn)` computes a minimum spanning tree from a `KNNGraph`
- `Tree` exposes traversal, path, distance, hop-count, and neighborhood helpers
- analysis functions that combine the tree with per-point labels or values: `boundary_edges`, `confusion_matrix_from_tree`, `edge_delta`, `node_diversity`, `path_properties`, `subtree_purity`

The MST is computed by OGDF, the same library that draws the layout.

---

## Quick Start

If you fitted a `TMAP`, the tree is already there:

```python
from tmap import TMAP

model = TMAP(metric="cosine").fit(X)
tree = model.tree_

print(tree.n_nodes)
print(len(tree.edges))
print(tree.root)
```

The estimator also has shortcuts for the most common queries: `model.path`, `model.distance`, `model.distances_from`, `model.hops` and `model.hops_from` do the same as the `Tree` methods below.

If you have a `KNNGraph` and no estimator, build the tree yourself:

```python
from tmap.graph import tree_from_knn_graph

tree = tree_from_knn_graph(knn)
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
from tmap import MinHash, LSHForest
from tmap.graph import tree_from_knn_graph

signatures = MinHash(num_perm=128, seed=42).batch_from_binary_array(fingerprints)

lsh = LSHForest(d=128, l=32)
lsh.batch_add(signatures)
lsh.index()

knn = lsh.get_knn_graph(k=20, kc=50)
tree = tree_from_knn_graph(knn)
```

### From USearch

`USearchIndex.query_knn` already returns a `KNNGraph`:

```python
from tmap.index import USearchIndex
from tmap.graph import tree_from_knn_graph

index = USearchIndex().build_from_vectors(X, metric="cosine")
tree = tree_from_knn_graph(index.query_knn(k=20))
```

### From another k-NN backend

For FAISS, Annoy, an alignment tool or anything else that gives you neighbor indices and distances, wrap the arrays in a `KNNGraph`:

```python
from tmap.graph import tree_from_knn_graph
from tmap.index import KNNGraph

knn = KNNGraph.from_arrays(indices, distances)   # both shape (n, k); use -1 for missing neighbors
tree = tree_from_knn_graph(knn)
```

To get a full map from those neighbors, pass the graph to the estimator instead: `TMAP().fit(knn_graph=knn)`.

---

## What `tree_from_knn_graph()` does

At a high level:

1. Convert the directed k-NN table into an undirected weighted edge list
2. Pass that graph to OGDF with MST creation enabled
3. Recover the returned MST edges as a `Tree`
4. Reattach edge weights from the original k-NN graph

For duplicated directed edges such as `i -> j` and `j -> i`, the smaller observed weight is kept. Self-matches, `-1` entries and non-finite distances are skipped.

The optional `config` argument takes a `LayoutConfig`. You rarely need it: by default the function runs OGDF with the cheapest settings, since only the tree topology is used.

---

## Tree Structure

The result is a `Tree` object with:

```python
tree.n_nodes
tree.edges     # shape (n_edges, 2), int32
tree.weights   # shape (n_edges,), float32
tree.root
```

`root` is chosen automatically as the node with the most edges in the extracted tree.

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

Both start at `root` unless you pass `start=`.

### Children

```python
children = tree.children(node=5, parent=2)
```

### Subtree sizes

```python
sizes = tree.subtree_sizes()
print(sizes[tree.root])
```

`sizes[i]` is the number of nodes below `i` (itself included) when the tree hangs from `root`.

---

## Paths and Distances

The tree API supports path-based analysis directly.

### Shortest path in the tree

```python
path = tree.path(10, 42)
```

### Tree distance

```python
distance = tree.distance(10, 42)   # sum of edge weights along the path
hops = tree.hops(10, 42)           # number of edges along the path
```

### Distances from one source

```python
distances = tree.distances_from(10)   # float32, one value per node
hops = tree.hops_from(10)             # int32, one value per node
```

### Nearby nodes

```python
nearby = tree.subtree(10, depth=2)
```

This returns every node within two edges of node 10, in any direction, closest first. With `depth=None` it returns the whole connected component.

These are useful for pseudotime, branch inspection, and neighborhood summaries.

---

## Analysis

The functions in `tmap.graph` take a tree plus one value or label per point:

```python
import numpy as np
from tmap.graph import (
    boundary_edges,
    confusion_matrix_from_tree,
    edge_delta,
    node_diversity,
    path_properties,
    subtree_purity,
)

labels = np.asarray(cell_types)            # one label per point
values = np.asarray(molecular_weight)      # one number per point

cross = boundary_edges(tree, labels)                 # edges joining two different labels
cmat, classes = confusion_matrix_from_tree(tree, labels)
jumps = edge_delta(tree, values)                     # |value change| across each edge
along = path_properties(tree, 10, 42, values)        # values along the path from 10 to 42
spread = node_diversity(tree, method="mean")         # mean weight of each node's edges
purity = subtree_purity(tree, labels, min_size=10)   # NaN below min_size
```

A few ways to read them:

- `len(cross) / len(tree.edges)` is the share of edges that cross a label boundary. A low number means the labels form clean branches.
- `cmat[i, j]` counts the edges between points labeled `classes[i]` and `classes[j]`. Large off-diagonal counts show which groups touch each other on the tree.
- Nodes with high `node_diversity` have long edges, so they tend to sit at the bridges between clusters.
- `subtree_purity` is the share of each node's subtree that has its most common label, so it shows which branches belong to one group.

---

## Disconnected Graphs

When `k` is too small, the k-NN graph can fall apart into several pieces, and a minimum spanning tree of it is a forest.

The estimator handles this by default. With `connect_components=True`, `TMAP.fit` finds the pieces, adds the cheapest edges that join them, and warns you. Afterward, `model.n_components_` tells you how many pieces there were and `model.n_bridges_` how many edges it added. Pass `connect_components=False` to keep the forest.

`tree_from_knn_graph` does not bridge anything. You can detect a forest with:

```python
n_components = tree.n_nodes - len(tree.edges)
```

On a forest, `path`, `distance` and `hops` raise `IndexError` for points in different pieces, `distances_from` gives `inf` and `hops_from` gives `-1` for unreachable points, and `bfs`, `dfs` and `subtree_sizes` only cover the piece that contains their start node.

Common reasons:

- `k` is too small
- `kc` is too small (LSHForest only)
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

lsh = LSHForest(d=128, l=32)
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
