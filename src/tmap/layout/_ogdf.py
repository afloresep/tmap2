"""
OGDF integration layer.

This module wraps the C++ extension (_tmap_ogdf) and provides:
1. Import guard with friendly error messages
2. Python-friendly convenience functions
3. Type conversions between tmap types and C++ types
"""

from __future__ import annotations

import importlib.util
import sysconfig
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from tmap.graph.types import Tree

# =============================================================================
# Load C++ extension
# =============================================================================

_AVAILABLE = False
_IMPORT_ERROR: ImportError | None = None

# These will be set by _load_extension()
LayoutConfig = None
LayoutResult = None
Merger = None
Placer = None
ScalingType = None
_cpp_layout_from_edge_list = None


def _load_extension() -> bool:
    """
    Load the _tmap_ogdf C++ extension.

    In editable installs, Python code comes from src/ but the compiled
    extension lives in site-packages. We handle this by:
    1. Trying normal import first
    2. Falling back to direct file loading from site-packages
    """
    global _AVAILABLE, _IMPORT_ERROR
    global LayoutConfig, LayoutResult, Merger, Placer, ScalingType, _cpp_layout_from_edge_list

    # Try normal import first
    try:
        from tmap.layout._tmap_ogdf import ( # type: ignore
            LayoutConfig as _LC,
            LayoutResult as _LR,
            Merger as _M,
            Placer as _P,
            ScalingType as _ST,
            layout_from_edge_list as _lfel,
        )
        # Assign to module-level globals (global declaration above makes this work)
        LayoutConfig = _LC
        LayoutResult = _LR
        Merger = _M
        Placer = _P
        ScalingType = _ST
        _cpp_layout_from_edge_list = _lfel
        _AVAILABLE = True
        return True
    except ImportError as e:
        _IMPORT_ERROR = e

    # Fallback: find and load the .so file directly from site-packages
    platlib = Path(sysconfig.get_paths()["platlib"])
    ext_dir = platlib / "tmap" / "layout"

    if not ext_dir.exists():
        return False

    # Find the extension file (name varies by platform)
    ext_files = (
        list(ext_dir.glob("_tmap_ogdf.cpython-*.so")) +  # Linux/macOS
        list(ext_dir.glob("_tmap_ogdf.*.pyd")) +          # Windows
        list(ext_dir.glob("_tmap_ogdf*.dylib"))           # macOS alternative
    )

    if not ext_files:
        return False

    try:
        # Load the extension directly by file path
        spec = importlib.util.spec_from_file_location("_tmap_ogdf", ext_files[0])
        if spec is None or spec.loader is None:
            return False

        module = importlib.util.module_from_spec(spec)
        sys.modules["_tmap_ogdf"] = module  # Cache it
        spec.loader.exec_module(module)

        # Extract the symbols we need
        LayoutConfig = module.LayoutConfig
        LayoutResult = module.LayoutResult
        Merger = module.Merger
        Placer = module.Placer
        ScalingType = module.ScalingType
        _cpp_layout_from_edge_list = module.layout_from_edge_list

        _AVAILABLE = True
        _IMPORT_ERROR = None
        return True

    except Exception as e:
        _IMPORT_ERROR = ImportError(f"Failed to load extension: {e}")
        return False


# Load on module import
_load_extension()


def require_ogdf() -> None:
    """Raise ImportError if OGDF extension is not available."""
    if not _AVAILABLE:
        raise ImportError(
            "OGDF layout extension not available. "
            "Install OGDF and rebuild: OGDF_DIR=/path/to/ogdf pip install --no-cache-dir -e ."
        ) from _IMPORT_ERROR


# =============================================================================
# Convenience functions
# =============================================================================

def layout_from_edge_list(
    vertex_count: int,
    edges: list[tuple[int, int, float]],
    config: "LayoutConfig | None" = None,
    create_mst: bool = True,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.uint32], NDArray[np.uint32]]:
    """
    Compute 2D layout from edge list.

    Parameters
    ----------
    vertex_count : int
        Number of vertices
    edges : list of (source, target, weight)
        Edge list. Weights should be positive.
    config : LayoutConfig, optional
        Layout configuration. If None, uses defaults.
    create_mst : bool, default True
        If True, compute MST first.

    Returns
    -------
    x, y, s, t : ndarrays
        Coordinates and edge topology
    """
    require_ogdf()

    if config is None:
        config = LayoutConfig()

    result = _cpp_layout_from_edge_list(vertex_count, edges, config, create_mst)

    return (
        np.array(result.x, dtype=np.float32),
        np.array(result.y, dtype=np.float32),
        np.array(result.s, dtype=np.uint32),
        np.array(result.t, dtype=np.uint32),
    )


def layout_from_tree(
    tree: "Tree",
    config: "LayoutConfig | None" = None,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Compute 2D layout from a Tree (MST).

    Parameters
    ----------
    tree : Tree
        Tree structure from MSTBuilder
    config : LayoutConfig, optional
        Layout configuration

    Returns
    -------
    x, y : ndarrays
        Coordinates
    """
    require_ogdf()

    if config is None:
        config = LayoutConfig()

    edges = [
        (int(tree.edges[i, 0]), int(tree.edges[i, 1]), float(tree.weights[i]))
        for i in range(len(tree.edges))
    ]

    result = _cpp_layout_from_edge_list(tree.n_nodes, edges, config, create_mst=False)

    return (
        np.array(result.x, dtype=np.float32),
        np.array(result.y, dtype=np.float32),
    )


__all__ = [
    "_AVAILABLE",
    "require_ogdf",
    "LayoutConfig",
    "Placer",
    "Merger",
    "ScalingType",
    "layout_from_edge_list",
    "layout_from_tree",
]
