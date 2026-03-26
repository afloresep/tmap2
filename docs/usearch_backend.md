# USearch Backend: Fast kNN for Binary, Cosine, and Euclidean Data

This guide explains the USearch nearest-neighbor backend in TMAP, when it's used, and how it compares to MinHash+LSH.

---

## What USearch Does

USearch is an HNSW (Hierarchical Navigable Small World) index that supports multiple distance metrics. TMAP uses it for:

- **Binary Jaccard** (new default for binary fingerprints)
- **Cosine** distance on dense vectors
- **Euclidean** distance on dense vectors

For binary data, USearch computes exact Jaccard distance on bit-packed vectors — no MinHash approximation is needed.

```txt
Binary Matrix (0/1)
        |
    [pack to bytes]
        |
    [USearch HNSW Index]  ← Jaccard distance on raw bits
        |
Fast Neighbor Queries -> k-NN Graph -> TMAP Layout
```

---

## Quick Start

### Through the estimator (automatic)

Binary matrices are automatically routed to USearch:

```python
from tmap import TMAP

# Binary fingerprints → USearch Jaccard (automatic)
model = TMAP(metric="jaccard", n_neighbors=20).fit(fingerprints)

# Sets/strings → MinHash+LSH (automatic)
model = TMAP(metric="jaccard", n_neighbors=20).fit(list_of_sets)
```

### Standalone (no TMAP, no layout)

Use the index directly for kNN queries without building a tree:

```python
from tmap.index import USearchIndex

# Build
idx = USearchIndex(seed=42)
idx.build_from_binary(fingerprints)   # (n, d) of 0/1 values

# Query
neighbors, distances = idx.query_batch(new_fingerprints, k=50)

# Incremental
idx.add(more_fingerprints)

# Persistence
idx.save("my_index.usearch")
idx = USearchIndex.load("my_index.usearch")
```

---

## How It Works

### HNSW on bit vectors

USearch builds a navigable small-world graph where each node is connected to its approximate nearest neighbors. Queries traverse this graph greedily, visiting nodes closer and closer to the target.

For binary data (`dtype='b1x8'`), the distance function is exact bitwise Jaccard:

```
Jaccard distance = 1 - popcount(A AND B) / popcount(A OR B)
```

This is computed on packed bytes using hardware popcount instructions, making individual distance computations very fast.

**Key difference from MinHash+LSH:** USearch operates on the *original* bit vectors. MinHash first compresses the data into fixed-width hash signatures, losing information. This compression is why LSH has much lower recall on sparse binary data.

### When USearch is used vs MinHash+LSH

| Input type | Backend | Why |
|---|---|---|
| Dense binary matrix (ndarray, DataFrame) | USearch HNSW | Native Jaccard on bits, high recall |
| Sparse matrix (scipy CSR) | MinHash + LSH | Avoids densifying large sparse data |
| List of integer sets | MinHash + LSH | Variable-length, can't pack to fixed bits |
| List of string tokens | MinHash + LSH | Variable-length, needs MinHash encoding |

The estimator detects the input type automatically.

---

## Performance

All benchmarks run on a MacBook (Apple Silicon, 24 GB RAM).
Reproducible via `python scripts/bench_usearch_vs_lsh.py`.

### Recall

Measured against brute-force ground truth (1024 bits, ~10% density, k=20):

| Method | Recall@20 (n=10k) | Recall@20 (n=50k) |
|---|---|---|
| USearch (expansion=512) | **98.5%** | **81.6%** |
| USearch (expansion=256) | 97.0% | — |
| LSH (d=512, kc=50) | 0.6% | 1.5% |

USearch recall decreases at larger N because HNSW is approximate, but it remains vastly better than LSH for sparse binary data. LSH fills all neighbor slots but with essentially random points — MinHash has fundamental recall limitations when bit overlap between vectors is low.

### Build + kNN graph time

1024-bit fingerprints, ~10% density, k=20:

| n | USearch build | USearch kNN | USearch total | LSH total | LSH mem |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 1.6s | 1.0s | **2.6s** | **0.7s** | 41 MB |
| 50,000 | 7.2s | 8.9s | **16.1s** | **2.3s** | 205 MB |
| 100,000 | 16.2s | 21.8s | **37.9s** | **5.9s** | 410 MB |
| 250,000 | 47.5s | 82.7s | **130s** | **22.9s** | 1,024 MB |
| 500,000 | 114.9s | 225.3s | **340s** | **84.6s** | 2,048 MB |
| 1,000,000 | ~250s | ~550s | **~800s** | ~250s | 4,096 MB |
| 2,000,000 | ~550s | ~1200s | **~1750s** | -OOM- | 16 GB+ |

LSH is 4-7x faster for building the full kNN graph. But those neighbors are wrong (1.5% recall), so the resulting TMAP tree is meaningless. At 1M+ points, LSH hits memory limits on a 24 GB machine while USearch fits in 128 MB.

The USearch kNN query (all-vs-all search) takes longer than the HNSW build itself. This is the main bottleneck at scale.

### Batch queries (100 queries against existing index)

| n | USearch | LSH |
|---:|---:|---:|
| 10,000 | 6.5ms | 10.5ms |
| 50,000 | 11.8ms | 15.9ms |
| 100,000 | 17.0ms | 20.5ms |
| 500,000 | 33.7ms | 80.7ms |

For `transform()` and `add_points()`, USearch is consistently faster because HNSW queries are O(log n). At 500k, USearch is 2.4x faster per query.

### Memory (vector storage)

| n | USearch | LSH (d=512) |
|---:|---:|---:|
| 100,000 | 13 MB | 410 MB |
| 250,000 | 32 MB | 1,024 MB |
| 500,000 | 64 MB | 2,048 MB |
| 1,000,000 | 128 MB | 4,096 MB |
| 2,000,000 | 256 MB | -OOM- |

USearch stores packed bytes (128 bytes per 1024-bit vector). LSH stores MinHash signatures (4,096 bytes per vector with d=512). USearch uses **32x less memory** for 1024-bit fingerprints (16x for 2048-bit).

### Summary

|  | USearch Jaccard | MinHash + LSH |
|---|---|---|
| **Recall** | 82-98% | 0.6-1.5% |
| **Build speed** | 4-7x slower | Faster (but wrong neighbors) |
| **Query speed** | 1.2-2.4x faster | Slower at scale |
| **Memory** | 16-32x less | OOM at ~1-2M |
| **Max scale (24 GB)** | 2M+ | ~500k-1M |
| **Hyperparameters** | None (auto) | d, l, kc |

**Bottom line:** USearch is the right default for binary data. It trades some build time for vastly better neighbor quality and lower memory. The build cost is one-time; queries are faster.

---

## Tuning

### expansion_search

The main quality knob. Higher values improve recall at the cost of slower queries.

| expansion_search | Recall@20 (n=10k) | Effect |
|---|---|---|
| 128 | 96.1% | Fast, slightly lower recall |
| 256 | 97.0% | Good balance |
| 512 | 98.5% | Best recall (TMAP default) |

The estimator uses `expansion_search=512` by default. For standalone use, you can set this in the constructor:

```python
idx = USearchIndex(expansion_search=256)  # faster, slightly lower recall
```

### connectivity

Controls the HNSW graph density. Higher values use more memory but improve recall.

```python
idx = USearchIndex(connectivity=64)  # default is 32
```

---

## Limitations

- **Sparse matrices** go through MinHash+LSH to avoid OOM from densification
- **Variable-length data** (sets, strings) requires MinHash encoding, so LSH is used
- **HNSW is approximate** — recall decreases with dataset size (82% at 50k vs 98% at 10k)
- **Build time** is O(n log n) with high constant for binary distance — slower than LSH for one-time builds
