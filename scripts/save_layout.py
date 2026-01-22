"""
Save TMAP layout data for reuse with regl-scatterplot demo.

Usage:
    # After running your TMAP pipeline:
    x, y, s, t, _ = tm.layout_from_lsh_forest(lf)

    # Save for later use:
    from scripts.save_layout import save_layout, load_layout
    save_layout(x, y, s, t, "my_layout.npz")

    # Load later:
    data = load_layout("my_layout.npz")
"""

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np


def save_layout(
    x: Any,
    y: Any,
    s: Any,
    t: Any,
    output_path: str | Path,
    metadata: dict | None = None,
) -> dict:
    """
    Save TMAP layout data in multiple formats for different use cases.

    Args:
        x: X coordinates (float array or tm.VectorFloat)
        y: Y coordinates (float array or tm.VectorFloat)
        s: Edge sources (tm.VectorUInt or array)
        t: Edge targets (tm.VectorUInt or array)
        output_path: Base path for output files (without extension)
        metadata: Optional dict with labels, smiles, categories, etc.

    Returns:
        dict with paths to saved files and stats
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert TMAP vectors to numpy arrays
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    s_arr = np.asarray(s, dtype=np.uint32)
    t_arr = np.asarray(t, dtype=np.uint32)

    n_points = len(x_arr)
    n_edges = len(s_arr)

    stats = {
        "n_points": n_points,
        "n_edges": n_edges,
        "x_range": [float(x_arr.min()), float(x_arr.max())],
        "y_range": [float(y_arr.min()), float(y_arr.max())],
    }

    # Calculate memory usage
    coords_bytes = x_arr.nbytes + y_arr.nbytes
    edges_bytes = s_arr.nbytes + t_arr.nbytes
    stats["memory_mb"] = (coords_bytes + edges_bytes) / (1024 * 1024)

    print(f"Layout stats:")
    print(f"  Points: {n_points:,}")
    print(f"  Edges: {n_edges:,}")
    print(f"  X range: [{stats['x_range'][0]:.2f}, {stats['x_range'][1]:.2f}]")
    print(f"  Y range: [{stats['y_range'][0]:.2f}, {stats['y_range'][1]:.2f}]")
    print(f"  Memory: {stats['memory_mb']:.2f} MB")

    saved_files = {}

    # 1. NumPy compressed format (.npz) - best for Python reuse
    npz_path = output_path.with_suffix(".npz")
    save_dict = {"x": x_arr, "y": y_arr, "s": s_arr, "t": t_arr}
    if metadata:
        for key, value in metadata.items():
            if isinstance(value, (list, np.ndarray)):
                save_dict[key] = np.asarray(value)
    np.savez_compressed(npz_path, **save_dict)
    saved_files["npz"] = str(npz_path)
    print(f"  Saved: {npz_path} ({npz_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # 2. Binary format for web (.bin) - efficient for JavaScript Float32Array
    bin_path = output_path.with_suffix(".bin")
    with open(bin_path, "wb") as f:
        # Header: n_points (uint32), n_edges (uint32)
        f.write(struct.pack("<II", n_points, n_edges))
        # Coordinates: x[], y[] as float32
        f.write(x_arr.tobytes())
        f.write(y_arr.tobytes())
        # Edges: s[], t[] as uint32
        f.write(s_arr.tobytes())
        f.write(t_arr.tobytes())
    saved_files["bin"] = str(bin_path)
    print(f"  Saved: {bin_path} ({bin_path.stat().st_size / 1024 / 1024:.2f} MB)")

    # 3. JSON for small datasets or metadata (skip for >100k points)
    if n_points <= 100_000:
        json_path = output_path.with_suffix(".json")
        json_data = {
            "points": {"x": x_arr.tolist(), "y": y_arr.tolist()},
            "edges": {"s": s_arr.tolist(), "t": t_arr.tolist()},
            "stats": stats,
        }
        if metadata:
            json_data["metadata"] = {
                k: v if isinstance(v, list) else v.tolist()
                for k, v in metadata.items()
            }
        with open(json_path, "w") as f:
            json.dump(json_data, f)
        saved_files["json"] = str(json_path)
        print(f"  Saved: {json_path} ({json_path.stat().st_size / 1024 / 1024:.2f} MB)")
    else:
        print(f"  Skipped JSON (too large: {n_points:,} points > 100k threshold)")

    # 4. Save stats separately
    stats_path = output_path.with_name(output_path.stem + "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    saved_files["stats"] = str(stats_path)

    return {"files": saved_files, "stats": stats}


def load_layout(path: str | Path) -> dict:
    """
    Load saved layout data.

    Args:
        path: Path to .npz file

    Returns:
        dict with x, y, s, t arrays and any metadata
    """
    path = Path(path)

    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        return {key: data[key] for key in data.files}
    elif path.suffix == ".bin":
        return load_binary_layout(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")


def load_binary_layout(path: str | Path) -> dict:
    """Load binary format layout."""
    with open(path, "rb") as f:
        n_points, n_edges = struct.unpack("<II", f.read(8))
        x = np.frombuffer(f.read(n_points * 4), dtype=np.float32)
        y = np.frombuffer(f.read(n_points * 4), dtype=np.float32)
        s = np.frombuffer(f.read(n_edges * 4), dtype=np.uint32)
        t = np.frombuffer(f.read(n_edges * 4), dtype=np.uint32)
    return {"x": x, "y": y, "s": s, "t": t}


def export_for_web(
    layout_path: str | Path,
    output_dir: str | Path,
    smiles: list[str] | None = None,
    labels: list[str] | None = None,
    categories: list[int] | None = None,
) -> str:
    """
    Export layout data optimized for web visualization.

    Creates a binary file and a JavaScript loader.

    Args:
        layout_path: Path to saved .npz layout
        output_dir: Directory for web assets
        smiles: Optional SMILES strings for hover
        labels: Optional labels for each point
        categories: Optional category indices for coloring

    Returns:
        Path to generated JavaScript module
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load layout
    data = load_layout(layout_path)
    x, y = data["x"], data["y"]
    n_points = len(x)

    # Normalize to [-1, 1] preserving aspect ratio
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)

    # Center and scale to fit in [-1, 1] with padding
    padding = 0.05
    scale = 2 * (1 - padding) / max_range

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    x_norm = ((x - x_center) * scale).astype(np.float32)
    y_norm = ((y - y_center) * scale).astype(np.float32)

    # Save normalized binary data
    bin_path = output_dir / "layout.bin"
    with open(bin_path, "wb") as f:
        f.write(struct.pack("<I", n_points))
        f.write(x_norm.tobytes())
        f.write(y_norm.tobytes())
        if categories is not None:
            cat_arr = np.asarray(categories, dtype=np.float32)
            f.write(cat_arr.tobytes())

    # Save metadata as JSON if provided
    if smiles or labels:
        meta_path = output_dir / "metadata.json"
        meta = {}
        if smiles:
            meta["smiles"] = smiles
        if labels:
            meta["labels"] = labels
        with open(meta_path, "w") as f:
            json.dump(meta, f)

    print(f"Exported {n_points:,} points to {output_dir}")
    print(f"  Normalized X range: [{x_norm.min():.3f}, {x_norm.max():.3f}]")
    print(f"  Normalized Y range: [{y_norm.min():.3f}, {y_norm.max():.3f}]")

    return str(bin_path)


if __name__ == "__main__":
    # Demo with synthetic data
    print("Demo: Generating synthetic layout data...")

    n_points = 1_000_000
    np.random.seed(42)

    # Simulate TMAP-like coordinates (tree layout tends to have fractal structure)
    x = np.random.randn(n_points).astype(np.float32) * 1000
    y = np.random.randn(n_points).astype(np.float32) * 1000

    # Simulate tree edges (n-1 edges for n nodes)
    s = np.arange(1, n_points, dtype=np.uint32)
    t = np.random.randint(0, np.arange(1, n_points), dtype=np.uint32)

    # Save
    result = save_layout(x, y, s, t, "demo_layout")
    print(f"\nSaved files: {result['files']}")
