# TMAP 

> **🚧 UNDER ACTIVE DEVELOPMENT 🚧**
>
> This is a modernized reimplementation of the original TMAP library. Core features are working, but the API may change before v1.0.
> Production users should continue using the [original TMAP](https://github.com/reymond-group/tmap) until this project reaches stability.

**TMAP** (Tree-MAP) creates beautiful, interactive visualizations of high-dimensional data by organizing similar items into tree structures. Perfect for chemical space, embeddings, or any high-dimensional dataset.

```text
Your Data → MinHash → LSHForest → k-NN Graph → MST → OGDF Layout → Interactive Visualization
```

[![Tests](https://github.com/afloresep/TMAP/actions/workflows/tests.yml/badge.svg)](https://github.com/afloresep/TMAP/actions/workflows/tests.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

## What's Working

**Core Pipeline**

- ✅ `MinHash` / `WeightedMinHash` encoding (via datasketch)
- ✅ `LSHForest` for fast approximate k-NN search
- ✅ MST (Minimum Spanning Tree) construction with bias control
- ✅ OGDF-based graph layout (FastMultipoleEmbedder + Multilevel Mixer)
- ✅ Interactive HTML visualizations with WebGL rendering

**Features**

- ✅ Deterministic, reproducible layouts (fixed seeds)
- ✅ Handles large datasets (10M+ points with binary encoding)
- ✅ Multiple color schemes and categorical/continuous data
- ✅ Easier API, removed Faerun dependency
- ✅ Improved Visualization: Lasso selection, exporting functions, (partial) search on labels
- ✅ Removed C++ backend (except for OGDF) while being 3x faster
- ✅ Many more!

**In Progress**

- 🚧 Performance benchmarks vs. original TMAP
- 🚧 Additional layout algorithms
- 🚧 Adding new points to existing TMAPs
- 🚧 Adding edges to TMAPs

## 📦 Installation

### Requirements

- Python **3.12+**
- For layout functionality: OGDF library (automatically built during install)

### Quick Install

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .\.venv\Scripts\Activate.ps1  # Windows (PowerShell)

# Install
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

**Notes:**

- The import name is `tmap`. If you need the original C++ `tmap` package, use a separate virtualenv to avoid conflicts.
- Optional extras: `.[faiss]`, `.[annoy]`, `.[viz]`, or `.[all]` (see `pyproject.toml`)
- OGDF will be built automatically if not found. Ensure you have `cmake`, `g++`, and `make` installed.

## 🚀 Quick Start

### Complete Example: Molecular Visualization

```python
from tmap import MinHash, LSHForest
from tmap.layout import layout_from_lsh_forest, LayoutConfig
from tmap.visualization import TmapViz
import numpy as np

# 1. Encode your binary data (e.g., molecular fingerprints)
fingerprints = np.random.randint(0, 2, (1000, 2048), dtype=np.uint8)

mh = MinHash(num_perm=128, seed=42)
signatures = mh.batch_from_binary_array(fingerprints)

# 2. Build LSH Forest index
lsh = LSHForest(d=128, l=64)
lsh.batch_add(signatures)
lsh.index()

# 3. Compute layout
cfg = LayoutConfig()
cfg.k = 20              # k-nearest neighbors
cfg.kc = 50             # Search multiplier
cfg.fme_iterations = 1000
cfg.deterministic = True
cfg.seed = 42

x, y, s, t = layout_from_lsh_forest(lsh, cfg)

# 4. Create interactive visualization
viz = TmapViz()
viz.title = "My TMAP"
viz.set_points(x, y)

# Add color by some property
viz.add_color_layout("Property", your_property_values, categorical=False)

# Save (auto-selects binary mode for large datasets)
viz.save("output.html")
```

### Simpler Example: Just the Layout

```python
import numpy as np
from tmap import MinHash, LSHForest
from tmap.layout import layout_from_lsh_forest, LayoutConfig

# Binary data
X = np.random.randint(0, 2, (100, 512), dtype=np.uint8)

# Encode → Index → Layout
mh = MinHash(num_perm=128, seed=42)
sigs = mh.batch_from_binary_array(X)

lsh = LSHForest(d=128, l=64)
lsh.batch_add(sigs)
lsh.index()

cfg = LayoutConfig(k=10, seed=42, deterministic=True)
x, y, s, t = layout_from_lsh_forest(lsh, cfg)

# x, y = node coordinates
# s, t = tree edges (source, target indices)
```

## 📚 Examples & Tutorials

Check out the [`examples/`](examples/) directory for complete, runnable examples:

- **[`smiles_tmap.py`](examples/smiles_tmap.py)** - Full pipeline: SMILES → Fingerprints → TMAP visualization with molecular properties
- **[`visualization_demo.py`](examples/visualization_demo.py)** - Visualization API examples with different configurations

Run the molecular example:

```bash
pip install rdkit tqdm  # Additional dependencies
python examples/smiles_tmap.py
```

This creates an interactive HTML file with:

- Pan & zoom navigation
- Hover to see molecule structures (via SMILES rendering)
- Multiple color schemes (molecular weight, LogP, ring count)
- Responsive design

## 📖 Documentation

Comprehensive guides are available in the [`docs/`](docs/) directory:

| Guide | Description |
|-------|-------------|
| [**Documentation Index**](docs/index.md) | Start here! Overview and getting started |
| [MinHash Guide](docs/minhash_guide.md) | Understanding MinHash encoding and choosing parameters |
| [LSHForest Guide](docs/lshforest_guide.md) | Building the index, query methods, k-NN construction |
| [Graph Guide](docs/graph_guide.md) | MST construction, tree traversal, bias factor tuning |
| [Layout Guide](docs/layout_guide.md) | OGDF layout configuration, parameter tuning, determinism |
| [Visualization Guide](docs/visualization_guide.md) | Creating interactive visualizations with TmapViz |
| [API Reference](docs/api_reference.md) | Complete API documentation |

## 🧪 Development

### Running Tests

```bash
# Run all tests
pytest -v

# Run specific test file
pytest tests/test_lshforest.py -v

# Run without OGDF tests (faster)
pytest -v --ignore=tests/test_layout_ogdf.py
```

### Code Quality

```bash
# Format code
ruff format src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

### Benchmarks

```bash
# Benchmark new implementation
python scripts/benchmark_new_tmap.py

# Compare with original (requires separate venv with old tmap)
python scripts/benchmark_old_tmap.py
```

## 🏗️ Architecture

TMAP2 is built with a modular, composable architecture:

```python
# Each stage is independent and swappable
Data → Encoder (MinHash) → Index (LSHForest) → Graph (MST) → Layout (OGDF) → Viz (TmapViz)
```

**Key Design Principles:**

- **Deterministic**: Same input + seed = same output
- **Type-safe**: Full type hints, validated with mypy
- **Testable**: High test coverage with clear test cases
- **Extensible**: Strategy pattern allows swapping implementations

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.
