# TMAP2 (Python)

Tree-based visualization for high-dimensional data, built as a modular pipeline:

`Raw Data → [Index] → k-NN Graph → [Graph] → MST → [Layout] → Coordinates → [Visualization] → Output`

This repo is currently **alpha/WIP**. The core goal is a clean, swappable architecture (Strategy pattern + composition)
with deterministic, reproducible behavior.

## What’s Implemented

- `MinHash` / `WeightedMinHash` encoders (wrapper around `datasketch`)
- `MSTBuilder` (minimum spanning tree from a k-NN graph) with optional “bias toward close neighbors”
- Core data types (`KNNGraph`, `Tree`, `Coordinates`) and pipeline skeleton (`TreeMap`)

## Requirements

- Python **3.12+** (see `pyproject.toml`)

## Install (Fresh Environment)

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .\.venv\Scripts\Activate.ps1  # Windows (PowerShell)

python -m pip install -U pip

# Dev install (recommended): tests, linting, + datasketch for MinHash encoders
python -m pip install -e ".[dev]"
```

Notes:
- The import name is `tmap`. If you also need the original C++ `tmap` package, use a separate virtualenv to avoid
  conflicts.
- Optional extras are available via `.[faiss]`, `.[annoy]`, `.[viz]`, or `.[all]` (see `pyproject.toml`).

## Quickstart

### MinHash signatures

```python
import numpy as np
from tmap import MinHash

# Binary feature matrix: (n_samples, n_features)
X = (np.random.random((3, 10)) < 0.2).astype(np.uint8)

mh = MinHash(num_perm=128, seed=42)
signatures = mh.encode(X)
print(signatures.shape)  # (3, 128)

print(mh.get_distance(signatures[0], signatures[1]))
```

### MST from a k-NN graph

```python
import numpy as np

from tmap.graph.mst import MSTBuilder
from tmap.index.types import KNNGraph

# 4 nodes, k=2 neighbors per node
indices = np.array(
    [
        [1, 2],
        [0, 3],
        [0, 3],
        [1, 2],
    ],
    dtype=np.int32,
)
distances = np.array(
    [
        [0.1, 0.2],
        [0.1, 0.3],
        [0.2, 0.4],
        [0.3, 0.4],
    ],
    dtype=np.float32,
)

knn = KNNGraph(indices=indices, distances=distances)
tree = MSTBuilder(bias_factor=0.1).build(knn)

print(tree.edges.shape)    # (n_nodes - 1, 2)
print(tree.weights.shape)  # (n_nodes - 1,)
```

## Run

### Tests

```bash
pytest -v
```

### Benchmarks

```bash
python scripts/benchmark_new_tmap.py
```

To compare against the original `tmap` package, run the old benchmark in a separate environment:

```bash
python scripts/benchmark_old_tmap.py
```

## Development

```bash
ruff check src/ tests/
ruff format src/ tests/
mypy src/
```

## Docs

- Design notes: `docs/design_patterns.md`
- LSH Forest plan: `docs/LSH_FOREST_IMPLEMENTATION.md`
- Original paper: `docs/tmap_paper.pdf`

## License

MIT
