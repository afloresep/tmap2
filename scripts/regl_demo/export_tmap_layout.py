#!/usr/bin/env python3
"""
Export TMAP layout coordinates for regl-scatterplot visualization.

This script shows how to save layout data from:
    x, y, s, t, _ = tm.layout_from_lsh_forest(lf)

Usage:
    # In your TMAP pipeline, after computing the layout:
    from export_tmap_layout import export_layout_for_web

    x, y, s, t, _ = tm.layout_from_lsh_forest(lf)
    export_layout_for_web(x, y, output_dir="./regl_demo")

    # Then serve:
    cd regl_demo && python -m http.server 8080
"""

import json
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np


def tmap_vector_to_numpy(vec: Any, dtype=np.float32) -> np.ndarray:
    """
    Convert TMAP vector types to numpy arrays.

    Handles:
    - tm.VectorFloat -> np.float32 array
    - tm.VectorUint / tm.VectorUChar -> np.uint32 array
    - Already numpy array -> pass through
    - List -> convert to numpy
    """
    if isinstance(vec, np.ndarray):
        return vec.astype(dtype)

    # Try to convert TMAP vector types
    try:
        # TMAP vectors support list() conversion
        return np.array(list(vec), dtype=dtype)
    except (TypeError, AttributeError):
        pass

    # Last resort: assume it's array-like
    return np.asarray(vec, dtype=dtype)


def normalize_coordinates(
    x: np.ndarray,
    y: np.ndarray,
    padding: float = 0.05,
    preserve_aspect: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize coordinates to [-1, 1] for regl-scatterplot.

    Args:
        x: X coordinates
        y: Y coordinates
        padding: Padding from edges (0.05 = 5% padding)
        preserve_aspect: If True, use same scale for x and y (no distortion)

    Returns:
        Tuple of (x_normalized, y_normalized) as Float32 arrays
    """
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_range = x_max - x_min or 1
    y_range = y_max - y_min or 1

    if preserve_aspect:
        # Use the larger range for both to preserve aspect ratio
        max_range = max(x_range, y_range)
        x_scale = y_scale = max_range
    else:
        x_scale = x_range
        y_scale = y_range

    # Target range is [-1+padding, 1-padding]
    target_range = 2 * (1 - padding)

    # Center and scale
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    x_norm = ((x - x_center) / x_scale * target_range).astype(np.float32)
    y_norm = ((y - y_center) / y_scale * target_range).astype(np.float32)

    return x_norm, y_norm


def export_layout_for_web(
    x: Any,
    y: Any,
    output_dir: str | Path = ".",
    smiles: list[str] | None = None,
    labels: list[str] | None = None,
    categories: list[int] | np.ndarray | None = None,
    preserve_aspect: bool = True,
) -> dict:
    """
    Export TMAP layout for web visualization with regl-scatterplot.

    Args:
        x: X coordinates from tm.layout_from_lsh_forest()
        y: Y coordinates from tm.layout_from_lsh_forest()
        output_dir: Directory to write output files
        smiles: Optional SMILES strings for hover tooltips
        labels: Optional labels for each point
        categories: Optional category indices for coloring (0-4 recommended)
        preserve_aspect: If True, preserve aspect ratio (no squashing)

    Returns:
        Dict with file paths and stats
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert TMAP vectors to numpy
    x_arr = tmap_vector_to_numpy(x, np.float32)
    y_arr = tmap_vector_to_numpy(y, np.float32)

    n_points = len(x_arr)
    print(f"Exporting {n_points:,} points for web visualization...")

    # Stats
    stats = {
        "n_points": n_points,
        "x_range_original": [float(x_arr.min()), float(x_arr.max())],
        "y_range_original": [float(y_arr.min()), float(y_arr.max())],
    }

    # Normalize coordinates
    start = time.time()
    x_norm, y_norm = normalize_coordinates(x_arr, y_arr, preserve_aspect=preserve_aspect)
    stats["normalize_time_ms"] = (time.time() - start) * 1000

    stats["x_range_normalized"] = [float(x_norm.min()), float(x_norm.max())]
    stats["y_range_normalized"] = [float(y_norm.min()), float(y_norm.max())]

    # Prepare categories
    if categories is not None:
        cat_arr = tmap_vector_to_numpy(categories, np.float32)
    else:
        cat_arr = None

    # Write binary file
    start = time.time()
    bin_path = output_dir / "layout.bin"
    with open(bin_path, "wb") as f:
        # Header
        f.write(struct.pack("<I", n_points))
        # Normalized coordinates
        f.write(x_norm.tobytes())
        f.write(y_norm.tobytes())
        # Categories (optional but recommended for coloring)
        if cat_arr is not None:
            f.write(cat_arr.tobytes())

    stats["binary_write_time_ms"] = (time.time() - start) * 1000
    stats["binary_size_mb"] = bin_path.stat().st_size / 1024 / 1024

    print(f"  Binary file: {bin_path} ({stats['binary_size_mb']:.2f} MB)")

    # Write metadata JSON (for SMILES hover, labels, etc.)
    if smiles or labels:
        start = time.time()
        meta_path = output_dir / "metadata.json"
        meta = {}
        if smiles:
            meta["smiles"] = smiles
        if labels:
            meta["labels"] = labels

        with open(meta_path, "w") as f:
            json.dump(meta, f)

        stats["metadata_write_time_ms"] = (time.time() - start) * 1000
        stats["metadata_size_mb"] = meta_path.stat().st_size / 1024 / 1024
        print(f"  Metadata file: {meta_path} ({stats['metadata_size_mb']:.2f} MB)")

    # Write stats
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nStats:")
    print(f"  Points: {n_points:,}")
    print(f"  Original X range: [{stats['x_range_original'][0]:.2f}, {stats['x_range_original'][1]:.2f}]")
    print(f"  Original Y range: [{stats['y_range_original'][0]:.2f}, {stats['y_range_original'][1]:.2f}]")
    print(f"  Normalized X range: [{stats['x_range_normalized'][0]:.4f}, {stats['x_range_normalized'][1]:.4f}]")
    print(f"  Normalized Y range: [{stats['y_range_normalized'][0]:.4f}, {stats['y_range_normalized'][1]:.4f}]")

    return {
        "files": {
            "binary": str(bin_path),
            "stats": str(stats_path),
        },
        "stats": stats,
    }


def save_full_layout(
    x: Any,
    y: Any,
    s: Any,
    t: Any,
    output_path: str | Path,
    smiles: list[str] | None = None,
    labels: list[str] | None = None,
    categories: list[int] | np.ndarray | None = None,
) -> dict:
    """
    Save complete TMAP layout (coordinates + edges) for Python reuse.

    This saves the raw layout data in numpy format so you can reload it
    later without recomputing the layout.

    Args:
        x: X coordinates from tm.layout_from_lsh_forest()
        y: Y coordinates from tm.layout_from_lsh_forest()
        s: Edge sources from tm.layout_from_lsh_forest()
        t: Edge targets from tm.layout_from_lsh_forest()
        output_path: Output path (will add .npz extension)
        smiles: Optional SMILES strings
        labels: Optional labels
        categories: Optional categories

    Returns:
        Dict with file path and stats
    """
    output_path = Path(output_path)

    # Convert all vectors
    x_arr = tmap_vector_to_numpy(x, np.float32)
    y_arr = tmap_vector_to_numpy(y, np.float32)
    s_arr = tmap_vector_to_numpy(s, np.uint32)
    t_arr = tmap_vector_to_numpy(t, np.uint32)

    n_points = len(x_arr)
    n_edges = len(s_arr)

    print(f"Saving layout: {n_points:,} points, {n_edges:,} edges")

    # Build save dict
    save_dict = {
        "x": x_arr,
        "y": y_arr,
        "s": s_arr,
        "t": t_arr,
    }

    if smiles is not None:
        save_dict["smiles"] = np.array(smiles, dtype=object)
    if labels is not None:
        save_dict["labels"] = np.array(labels, dtype=object)
    if categories is not None:
        save_dict["categories"] = tmap_vector_to_numpy(categories, np.float32)

    # Save compressed
    npz_path = output_path.with_suffix(".npz")
    np.savez_compressed(npz_path, **save_dict)

    size_mb = npz_path.stat().st_size / 1024 / 1024
    print(f"Saved: {npz_path} ({size_mb:.2f} MB)")

    return {
        "path": str(npz_path),
        "n_points": n_points,
        "n_edges": n_edges,
        "size_mb": size_mb,
    }


def load_full_layout(path: str | Path) -> dict:
    """
    Load saved TMAP layout.

    Args:
        path: Path to .npz file

    Returns:
        Dict with x, y, s, t and optional smiles, labels, categories
    """
    data = np.load(path, allow_pickle=True)
    result = {key: data[key] for key in data.files}

    # Convert object arrays back to lists
    for key in ["smiles", "labels"]:
        if key in result:
            result[key] = result[key].tolist()

    return result


# Example usage
if __name__ == "__main__":
    print("Example: Exporting synthetic layout data\n")

    # Simulate TMAP output
    n = 100_000
    np.random.seed(42)

    # Synthetic coordinates (imagine these come from tm.layout_from_lsh_forest)
    x = np.random.randn(n).astype(np.float32) * 1000
    y = np.random.randn(n).astype(np.float32) * 1000

    # Synthetic edges (tree has n-1 edges)
    s = np.arange(1, n, dtype=np.uint32)
    t = np.random.randint(0, np.arange(1, n), dtype=np.uint32)

    # Synthetic metadata
    smiles = [f"C{'C' * (i % 10)}O" for i in range(n)]  # Dummy SMILES
    labels = [f"mol_{i}" for i in range(n)]
    categories = [i % 5 for i in range(n)]

    # Save full layout for Python reuse
    result = save_full_layout(
        x, y, s, t,
        output_path="example_layout",
        smiles=smiles,
        labels=labels,
        categories=categories,
    )
    print()

    # Export for web visualization
    export_layout_for_web(
        x, y,
        output_dir="./",
        smiles=smiles[:1000],  # Only first 1k for demo
        labels=labels,
        categories=categories,
    )
