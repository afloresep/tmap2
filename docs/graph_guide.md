# Understanding the Graph Module: MST and Tree Structures

This guide explains how TMAP converts k-NN graphs into tree structures using Minimum Spanning Trees (MST).

---

## What the Graph Module Does

The graph module sits between **indexing** (LSHForest) and **layout** (OGDF):

```
LSHForest
    ↓
k-NN Graph (many edges per node)
    ↓
[MSTBuilder] → MST (exactly n-1 edges)
    ↓
Tree structure
    ↓
Layout algorithm
```

**The key insight:** A k-NN graph has `n × k` edges (dense). An MST has exactly `n - 1` edges (sparse). This makes layout computation tractable for large datasets.

---

## Quick Start

```python
from tmap import LSHForest
from tmap.graph import MSTBuilder, Tree

# 1. Get k-NN graph from LSHForest
knn = lsh.get_knn_graph(k=20, kc=50)

# 2. Build MST
builder = MSTBuilder(bias_factor=0.1)
tree = builder.build(knn)

# 3. Use tree for layout
print(f"Tree has {tree.n_nodes} nodes and {len(tree.edges)} edges")
```

---

## Why MST?

### The Problem with k-NN Graphs

A k-NN graph connects each point to its k nearest neighbors:
- 10,000 points × k=20 neighbors = 200,000 edges
- Layout algorithms are O(E) or worse
- Too many edges = slow layout, cluttered visualization

### The MST Solution

MST keeps only the **essential** edges:
- 10,000 points → exactly 9,999 edges
- Minimizes total distance while keeping everything connected
- Perfect for tree-based visualization

### Visual Intuition

```
k-NN Graph (dense)              MST (sparse)
    A---B                         A---B
   /|\ /|\                        |
  / | X | \         →             |
 /  |/ \|  \                      |
C---D---E---F                   C-D-E-F
```

---

## MSTBuilder

### Basic Usage

```python
from tmap.graph import MSTBuilder

# Default: standard MST
builder = MSTBuilder()
tree = builder.build(knn)

# With bias toward close neighbors
builder = MSTBuilder(bias_factor=0.1)
tree = builder.build(knn)
```

### The `bias_factor` Parameter

Controls whether to prefer closer neighbors in the MST.

| Value | Effect | When to Use |
|-------|--------|-------------|
| 0.0 | Standard MST (globally optimal) | **Default** - most cases |
| 0.1 | Slight preference for close neighbors | Tighter clusters |
| 0.3 | Moderate preference | Reduce "stringy" layouts |
| 0.5+ | Strong preference | May hurt global structure |

**How it works:**

```
Original weight: distance
Modified weight: distance × (1 + bias_factor × rank / k)

For neighbor at rank 0 (closest): weight unchanged
For neighbor at rank k-1 (furthest): weight × (1 + bias_factor)
```

**Example:**
```python
# Without bias - MST may connect distant points if globally optimal
tree_standard = MSTBuilder(bias_factor=0.0).build(knn)

# With bias - prefers connecting close neighbors
tree_biased = MSTBuilder(bias_factor=0.2).build(knn)
```

**Rule of thumb:** Start with `bias_factor=0.0`. If your layout looks too "stringy" (long chains instead of compact clusters), try increasing to 0.1-0.2.

---

## Tree Structure

The `Tree` class represents the MST result.

### Attributes

```python
tree = builder.build(knn)

# Core data
tree.n_nodes      # Number of nodes
tree.edges        # Array of (source, target) pairs, shape (n-1, 2)
tree.weights      # Edge weights, shape (n-1,)
tree.root         # Root node index (highest-degree node)
```

### Neighbor Access

```python
# Get all neighbors of a node with their edge weights
neighbors = tree.neighbors(5)
# Returns: [(neighbor_id, weight), ...]

for neighbor, weight in neighbors:
    print(f"Node 5 connects to {neighbor} with weight {weight:.3f}")
```

### Tree Traversal

#### Breadth-First Search (BFS)

Process nodes level-by-level from root.

```python
# BFS traversal
for node, parent, depth in tree.bfs():
    print(f"Node {node} at depth {depth}, parent={parent}")

# Start from specific node
for node, parent, depth in tree.bfs(start=5):
    ...
```

**Use cases:**
- Level-by-level processing
- Finding all nodes at a specific depth
- Top-down layout algorithms

#### Depth-First Search (DFS)

Process entire subtrees before siblings.

```python
# DFS traversal
for node, parent, depth in tree.dfs():
    print(f"Node {node} at depth {depth}")
```

**Use cases:**
- Subtree operations
- Post-order processing (process children before parent)
- Computing subtree properties

### Children Access

```python
# Get children of a node (excluding parent)
children = tree.children(node=5, parent=3)
# Returns: [child_id, ...]
```

### Subtree Sizes

Compute the size of subtree rooted at each node.

```python
sizes = tree.subtree_sizes()
# sizes[i] = number of nodes in subtree rooted at node i

print(f"Subtree rooted at node 0 has {sizes[0]} nodes")  # Should be n_nodes if 0 is root
```

**Use case:** Layout algorithms that need to allocate space proportional to subtree size.

---

## Understanding the Algorithm

### Step 1: k-NN to Sparse Matrix

The k-NN graph is converted to a sparse adjacency matrix:

```python
# KNNGraph format:
# knn.indices[i] = [neighbor_1, neighbor_2, ..., neighbor_k]
# knn.distances[i] = [dist_1, dist_2, ..., dist_k]

# Becomes sparse matrix:
# adj[i, j] = distance from i to j (if j is neighbor of i)
```

**Important:** The matrix is symmetrized (undirected). If `i → j` and `j → i` have different distances, the minimum is used.

### Step 2: Apply Bias (Optional)

If `bias_factor > 0`, edge weights are modified:

```python
# For edge (i, j) where j is the r-th nearest neighbor of i:
modified_weight = original_weight × (1 + bias_factor × r / k)
```

### Step 3: Compute MST

Uses scipy's `minimum_spanning_tree` (Kruskal's algorithm with union-find):
- Time complexity: O(E log E)
- Returns exactly n-1 edges (for connected graph)
- May return fewer edges if graph is disconnected

### Step 4: Build Tree Object

The MST edges and weights are packaged into a `Tree` object with:
- Adjacency list for fast traversal
- Root node selection (highest-degree node)

---

## Handling Disconnected Graphs

If the k-NN graph has disconnected components, the MST will also be disconnected.

### Detection

```python
n_edges = len(tree.edges)
n_components = tree.n_nodes - n_edges

if n_components > 1:
    print(f"Warning: {n_components} disconnected components")
```

### Causes

1. **Low k:** Not enough neighbors to bridge clusters
2. **Low kc:** Poor neighbor quality
3. **Inherent structure:** Data truly has isolated clusters

### Solutions

```python
# Increase connectivity in LSHForest
knn = lsh.get_knn_graph(k=30, kc=100)  # More neighbors, better quality

# Or accept disconnected components (layout will handle them)
tree = builder.build(knn)
```

---

## Complete Example

```python
import numpy as np
from tmap import MinHash, LSHForest
from tmap.graph import MSTBuilder

# 1. Create sample data
n_samples = 1000
n_bits = 2048
fingerprints = (np.random.rand(n_samples, n_bits) < 0.1).astype(np.uint8)

# 2. MinHash encoding
mh = MinHash(num_perm=128, seed=42)
signatures = mh.batch_from_binary_array(fingerprints)

# 3. Build LSH Forest and get k-NN graph
lsh = LSHForest(d=128, l=64)
lsh.batch_add(signatures)
lsh.index()

knn = lsh.get_knn_graph(k=20, kc=50)

# 4. Build MST
builder = MSTBuilder(bias_factor=0.1)
tree = builder.build(knn)

# 5. Inspect tree
print(f"Nodes: {tree.n_nodes}")
print(f"Edges: {len(tree.edges)}")
print(f"Root: {tree.root}")

# Check connectivity
n_components = tree.n_nodes - len(tree.edges)
print(f"Components: {n_components}")

# Tree traversal example
print("\nFirst 5 nodes in BFS order:")
for i, (node, parent, depth) in enumerate(tree.bfs()):
    if i >= 5:
        break
    print(f"  Node {node}, depth {depth}, parent {parent}")

# Subtree sizes
sizes = tree.subtree_sizes()
print(f"\nSubtree size at root: {sizes[tree.root]}")
```

---

## When to Use the Graph Module Directly

### Use High-Level API (Recommended)

For most cases, use `layout_from_lsh_forest()` which handles MST internally:

```python
from tmap.layout import layout_from_lsh_forest, LayoutConfig

cfg = LayoutConfig()
cfg.k = 20
cfg.kc = 50

x, y, s, t = layout_from_lsh_forest(lsh, cfg)
```

### Use Graph Module Directly When:

1. **Custom MST parameters:** You want to tune `bias_factor`
2. **Inspect/modify tree:** You need to analyze or modify the tree structure
3. **Multiple layouts:** Compute MST once, try different layout parameters
4. **Non-LSH input:** You have k-NN from another source (FAISS, Annoy, etc.)

```python
# Example: Try different bias factors
from tmap.graph import MSTBuilder
from tmap.layout import ForceDirectedLayout

knn = lsh.get_knn_graph(k=20, kc=50)

for bias in [0.0, 0.1, 0.2]:
    tree = MSTBuilder(bias_factor=bias).build(knn)
    layout = ForceDirectedLayout(seed=42)
    coords = layout.compute(tree)
    print(f"bias={bias}: {len(tree.edges)} edges")
```

---

## Performance Characteristics

| Dataset Size | k-NN Edges | MST Edges | MST Time |
|--------------|------------|-----------|----------|
| 1,000 | 20,000 | 999 | ~10ms |
| 10,000 | 200,000 | 9,999 | ~100ms |
| 100,000 | 2,000,000 | 99,999 | ~1s |
| 1,000,000 | 20,000,000 | 999,999 | ~10s |

MST computation is typically not the bottleneck (k-NN and layout are slower).

---

## Common Pitfalls

### 1. Forgetting Disconnected Components

```python
# ❌ Assuming tree is fully connected
assert len(tree.edges) == tree.n_nodes - 1  # May fail!

# ✓ Check for disconnected components
n_components = tree.n_nodes - len(tree.edges)
if n_components > 1:
    print(f"Graph has {n_components} components")
```

### 2. Using Tree Before Building

```python
# ❌ Tree methods require adjacency to be built
tree = Tree(n_nodes=10, edges=edges, weights=weights)
# __post_init__ builds adjacency automatically, so this is OK

# But if you modify edges manually:
tree.edges = new_edges  # Adjacency is now stale!
tree._build_adjacency()  # Must rebuild
```

### 3. Invalid k-NN Input

```python
# ❌ Empty k-NN graph
knn = lsh.get_knn_graph(k=5, kc=1)  # Very low kc
# If no neighbors found, tree will have 0 edges

# ✓ Use sufficient kc
knn = lsh.get_knn_graph(k=5, kc=20)
```

---

## Integration with Layout

The Tree object is the input to layout algorithms:

```python
from tmap.layout import ForceDirectedLayout

# Build tree
tree = MSTBuilder().build(knn)

# Compute layout
layout = ForceDirectedLayout(seed=42, max_iterations=1000)
coords = layout.compute(tree)

# coords.x, coords.y are the 2D positions
```

Or use the convenience function which converts tree edges:

```python
from tmap.layout import layout_from_edge_list

edges = [
    (int(tree.edges[i, 0]), int(tree.edges[i, 1]), float(tree.weights[i]))
    for i in range(len(tree.edges))
]

x, y, s, t = layout_from_edge_list(tree.n_nodes, edges, config, create_mst=False)
```

---

## Next Steps

- See [MinHash Guide](minhash_guide.md) for encoding
- See [LSHForest Guide](lshforest_guide.md) for k-NN construction
- See [Layout Guide](layout_guide.md) for visualization parameters
