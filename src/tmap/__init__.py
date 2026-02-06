"""
TreeMap: Tree-based visualization for high-dimensional data.

This package is generic - it works with ANY data that can be represented
as edges/distances between points. Molecules, papers, images, embeddings, etc.

Architecture Overview
---------------------
The pipeline has 4 stages, each with a clear interface:

1. Index: Build a data structure for fast nearest-neighbor queries
   - Input: Distance matrix OR edge list OR raw vectors
   - Output: k-NN graph (sparse adjacency with distances)

2. Graph: Convert k-NN graph to a tree structure
   - Input: k-NN graph
   - Output: MST (minimum spanning tree)

3. Layout: Compute 2D coordinates for the tree
   - Input: MST
   - Output: (x, y) coordinates per node

4. Visualization: Render the layout
   - Input: Coordinates + metadata
   - Output: HTML/PNG/interactive plot

Design Principles
-----------------
- Each stage is independent and swappable (Strategy pattern)
- Base classes define the interface; you implement concrete strategies
- Composition over inheritance for the main TreeMap class
- Type hints everywhere for IDE support and self-documentation
"""

import sysconfig
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

# In editable installs, Python modules load from src/ but compiled extensions
# live in site-packages. Extend __path__ so imports can find the extensions.
_platlib_tmap = Path(sysconfig.get_paths()["platlib"]) / "tmap"
if _platlib_tmap.is_dir() and str(_platlib_tmap) not in __path__:
    __path__.append(str(_platlib_tmap))

if TYPE_CHECKING:
    from tmap.index.encoders.minhash import MinHash, WeightedMinHash
    from tmap.index.lsh_forest import LSHForest

__version__ = "0.1.0"


__all__ = [
    "__version__",
    "MinHash",
    "WeightedMinHash",
    "LSHForest",
    # "TreeMap",
    # "Index", "FaissIndex", "AnnoyIndex",
    # "GraphBuilder", "MSTBuilder",
    # "Layout", "ForceDirectedLayout",
    # "Visualizer", "HTMLVisualizer",
]


def __getattr__(name: str) -> Any:
    if name == "LSHForest":
        module = import_module("tmap.index.lsh_forest")
        value = getattr(module, name)
        globals()[name] = value
        return value

    if name in {"MinHash", "WeightedMinHash"}:
        try:
            module = import_module("tmap.index.encoders.minhash")
        except ModuleNotFoundError as exc:
            if exc.name == "datasketch":
                raise ModuleNotFoundError(
                    f"Optional dependency 'datasketch' is required for `tmap.{name}`. "
                    "Install it with `pip install datasketch`."
                ) from exc
            raise

        value = getattr(module, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + ["MinHash", "WeightedMinHash", "LSHForest"])
