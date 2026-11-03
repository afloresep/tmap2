# USearch Backend: Fast kNN for Binary, Cosine, and Euclidean Data

This guide explains the USearch nearest-neighbor backend in TMAP, when it's used, and how it compares to MinHash+LSH.

---

## What USearch Does

[USearch](https://github.com/unum-cloud/usearch) is a nearest-neighbor library with several distance metrics. TMAP uses it for:

- **Binary Jaccard**: the estimator's backend for 0/1 matrices such as molecular fingerprints
- **Cosine** distance on dense vectors
- **Euclidean** distance on dense vectors

It runs in one of two modes:

- **Exact**: compares each query with every vector. Used for cosine and euclidean below 50,000 points.
- **HNSW** (Hierarchical Navigable Small World): an approximate graph index that stays fast at millions of points. Used for cosine and euclidean from 50,000 points up, and always for binary Jaccard.

For binary data, USearch computes the Jaccard distance on the packed bits themselves, so there is no MinHash approximation in the distance.

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

# Lists of feature indices or string tokens → MinHash + LSH (automatic)
model = TMAP(metric="jaccard", n_neighbors=20).fit(list_of_index_lists)
```

After fitting on a binary matrix, the index is kept as `model.index_`, so `transform()`, `kneighbors()` and `add_points()` work without `store_index=True`. For cosine and euclidean, pass `store_index=True` if you need those methods.

### Standalone (no TMAP, no layout)

Use the index directly for kNN queries without building a tree:

```python
from tmap.index import USearchIndex

# Build
idx = USearchIndex()
idx.build_from_binary(fingerprints)   # (n, d) of 0/1 values

# Query
neighbors, distances = idx.query_batch(new_fingerprints, k=50)

# All-vs-all kNN graph, ready for tmap.graph or tmap.layout
knn = idx.query_knn(k=20)

# Incremental
idx.add(more_fingerprints)

# Persistence
idx.save("my_index.usearch")          # writes my_index.usearch, .meta and .vectors.npy
idx = USearchIndex.load("my_index.usearch")
```

`load()` needs all three files that `save()` writes.

For dense vectors, use `idx.build_from_vectors(X, metric="cosine")` or `metric="euclidean"`. Euclidean distances are returned as plain (square-rooted) distances. `idx.effective_mode` tells you whether the index ended up exact or HNSW.

---

## How It Works

### HNSW on bit vectors

USearch builds a navigable small-world graph where each node is connected to its approximate nearest neighbors. Queries traverse this graph greedily, visiting nodes closer and closer to the target.

For binary data (`dtype='b1x8'`), the distance function is exact bitwise Jaccard:

```
Jaccard distance = 1 - popcount(A AND B) / popcount(A OR B)
```

This is computed on packed bytes using hardware popcount instructions, making individual distance computations very fast.

**Difference from MinHash+LSH:** USearch compares the original bit vectors. MinHash first compresses the data into fixed-width hash signatures, and the Jaccard distance is then estimated from those signatures. That estimate is part of why LSH finds fewer of the true neighbors on binary data.

### When USearch is used vs MinHash+LSH

| Input type | Backend | Why |
|---|---|---|
| Dense binary matrix (NumPy array, bool array, DataFrame, rectangular list of 0/1 lists) | USearch HNSW | Native Jaccard on bits, high recall |
| Sparse matrix (SciPy CSR) | MinHash + LSH | Avoids densifying large sparse data |
| List of feature-index lists | MinHash + LSH | Variable-length, can't pack to fixed bits |
| List of string tokens | MinHash + LSH | Variable-length, needs MinHash encoding |

The estimator detects the input type automatically. A dense matrix with values other than 0 and 1 raises an error under `metric="jaccard"`; use cosine or euclidean for counts.

---

## Performance

All numbers below come from one Apple M4 Pro laptop (12 cores, 24 GB RAM, usearch 2.23) and use the settings `TMAP.fit` uses: USearch with `expansion_search=512` on all cores, and MinHash with 512 permutations plus an LSHForest with `l=128` and `kc=50`. Rerun them on your own machine with:

```bash
python scripts/bench_usearch_vs_lsh.py
```

Two datasets are used: random binary vectors (1024 bits, 10% of bits set), and Morgan fingerprints (radius 2, 2048 bits) of the 64,098 molecules in `examples/cluster_65053.csv`. Timings are single runs (best of 2 for the smaller sizes) and move by a few percent between runs.

### Recall

Recall@20 against exact Jaccard neighbors, with the point itself left out. Jaccard distances have many ties, so a returned neighbor counts as correct when its true distance is no larger than the true 20th neighbor's.

| Dataset | n | USearch | MinHash + LSH |
|---|---:|---:|---:|
| Morgan fingerprints | 10,000 | 99.9% | 83.9% |
| Morgan fingerprints | 64,098 | 99.9% | 82.6% |
| Random binary | 10,000 | 99.6% | 24.6% |
| Random binary | 50,000 | 87.3% | 12.5% |

On real fingerprints USearch finds almost every true neighbor, and LSH finds about five in six. Random vectors are a worst case for both. In random data the true 20th neighbor is barely closer than an average point (Jaccard distance about 0.89 against 0.95), so neither the MinHash prefixes nor the HNSW graph have much to go on. For the Morgan fingerprints the gap is wide (0.45 against 0.82). A higher `expansion_search` brings USearch back on random data: 96.0% at 1024 and 99.3% at 2048 for 50,000 points, at 1.4 and 3.3 times the query time.

### Build + kNN graph time

Random binary vectors, k=20:

| n | USearch build | USearch kNN | USearch total | LSH total | USearch memory | LSH memory |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.3s | 0.4s | 0.7s | 0.5s | 21 MB | 44 MB |
| 50,000 | 2.2s | 3.3s | 5.6s | 3.5s | 43 MB | 220 MB |
| 100,000 | 6.0s | 10.3s | 16.3s | 8.6s | 98 MB | 439 MB |
| 250,000 | 22.3s | 45.8s | 68.1s | 25.2s | 215 MB | 1,099 MB |
| 500,000 | 61.9s | 98.8s | 160.7s | 49.7s | 441 MB | 2,197 MB |

LSH builds the kNN graph 1.3 to 3.2 times faster. The LSH total includes MinHash encoding (5.6s at 500,000) and indexing (3.9s). The all-vs-all USearch query takes longer than the HNSW build itself, and it is the main cost at scale. For the 64,098 Morgan fingerprints, USearch took 4.3s and LSH 3.3s.

### Batch queries (100 new points against an existing index)

| n | USearch | LSH |
|---:|---:|---:|
| 10,000 | 5.4 ms | 4.9 ms |
| 50,000 | 9.4 ms | 7.2 ms |
| 100,000 | 13.7 ms | 9.9 ms |
| 250,000 | 18.7 ms | 8.6 ms |
| 500,000 | 23.2 ms | 8.9 ms |

This is the cost of `transform()`, `kneighbors()` and `add_points()`. LSH time levels off at about 9 ms because each query checks a fixed budget of `k * kc = 1000` candidates. Encoding the 100 queries with MinHash adds another 0.9 ms.

### Memory per vector

| | USearch | MinHash + LSH |
|---|---:|---:|
| Index | about 800-900 B (HNSW graph plus its copy of the bits) | 4,096 B signatures (512 × 8 bytes) plus 512 B prefix trees |
| TMAP's packed copy of the bits | 128 B (1024 bits), 256 B (2048 bits) | none |
| Total, 50,000+ points | about 0.9-1.1 KB | about 4.6 KB |

USearch uses about 4 to 5 times less memory. Below about 50,000 points the USearch figure is higher per vector (about 2.2 KB at 10,000), because it reserves space up front.

### Summary

|  | USearch Jaccard | MinHash + LSH |
|---|---|---|
| **Recall on fingerprints** | 99.9% | 83-84% |
| **Recall on random bits** | 87-100% | 12-25% |
| **Build speed** | 1.3-3.2x slower | Faster |
| **Batch query speed** | Slower, grows with n | Faster, flat at about 9 ms per 100 queries |
| **Memory** | About 1 KB per vector | About 4.6 KB per vector |
| **Settings** | `expansion_search`, `connectivity` (TMAP picks them) | `n_permutations`, `kc`, `l` (TMAP picks `l`) |

USearch is the default for binary data because it finds more of the true neighbors, and the tree is built from those neighbors. LSH is faster to build and query, so it remains a reasonable choice when speed matters more than neighbor quality, and it is the only option for sets, strings and sparse input.

---

## Tuning

### expansion_search

The main quality knob. Higher values improve recall at the cost of slower queries.

At n=10,000 (recall@20, same tie-aware count as above):

| expansion_search | Morgan fingerprints | Random binary | kNN time (random) |
|---|---|---|---|
| 128 | 99.7% | 97.4% | 0.20s |
| 256 | 99.8% | 98.3% | 0.31s |
| 512 | 99.8% | 99.5% | 0.52s |

On real fingerprints the setting barely matters at this size. On harder data, and on larger datasets, the higher values pay off.

For binary Jaccard, the estimator uses `expansion_search=512`. For cosine and euclidean it uses the `USearchIndex` default of 200 (which only matters once the index switches to HNSW at 50,000 points). For standalone use, set it in the constructor:

```python
idx = USearchIndex(expansion_search=256)  # faster, slightly lower recall
```

### connectivity

Controls the HNSW graph density. Higher values use more memory but improve recall.

```python
idx = USearchIndex(connectivity=64)  # default is 32
```

### mode

`mode="auto"` (the default) picks exact search below 50,000 dense vectors and HNSW above. Pass `mode="exact"` or `mode="hnsw"` to force one. Binary indexes always use HNSW.

---

## Reproducibility

The HNSW build uses every core by default (`threads=0`). Threads insert points in a slightly different order on each run, so the graph, and with it the returned neighbors, can change a little between runs. That can change the tree and the layout.

- Standalone: `USearchIndex(threads=1)` builds on one thread and gives the same index every time.
- Estimator: `TMAP(..., reproducible=True)` does the same for you. The HNSW build gets roughly 5 to 7 times slower at 10k+ points.
- Exact search (cosine/euclidean below 50,000 points) gives the same neighbors on every run.

The `seed` argument of `USearchIndex` is stored with the index as metadata. USearch itself does not use it.

---

## Limitations

- **Sparse matrices** go through MinHash+LSH so they don't have to be densified
- **Variable-length data** (sets, strings) requires MinHash encoding, so LSH is used
- **HNSW is approximate**: recall can drop on large datasets with weak neighbor structure (87% on 50,000 random vectors at the default settings, see above)
- **Build time**: the HNSW build and the all-vs-all query dominate the run time at large sizes
