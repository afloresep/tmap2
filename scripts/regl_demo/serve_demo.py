#!/usr/bin/env python3
"""
Serve the regl-scatterplot demo with TMAP data.

Usage:
    # With your .npy files (shape: (2, N) where row 0 = x, row 1 = y)
    python serve_demo.py --npy x_y_100k_coords.npy

    # With synthetic data for testing
    python serve_demo.py --synthetic 100000

    # With npz layout file
    python serve_demo.py --layout my_layout.npz
"""

import argparse
import http.server
import os
import socketserver
import struct
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np


def normalize_coordinates(x: np.ndarray, y: np.ndarray, padding: float = 0.05) -> tuple:
    """Normalize to [-1, 1] preserving aspect ratio."""
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    x_range = x_max - x_min or 1.0
    y_range = y_max - y_min or 1.0
    max_range = max(x_range, y_range)

    scale = 2.0 * (1.0 - padding) / max_range

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0

    x_norm = ((x - x_center) * scale).astype(np.float32)
    y_norm = ((y - y_center) * scale).astype(np.float32)

    return x_norm, y_norm


def create_binary_layout(
    x: np.ndarray,
    y: np.ndarray,
    categories: np.ndarray | None = None,
    output_path: str = "layout.bin",
) -> str:
    """Create binary file for web visualization."""
    n_points = len(x)

    # Normalize coordinates
    x_norm, y_norm = normalize_coordinates(x, y)

    with open(output_path, "wb") as f:
        # Header: number of points (uint32)
        f.write(struct.pack("<I", n_points))
        # Coordinates as float32
        f.write(x_norm.tobytes())
        f.write(y_norm.tobytes())
        # Categories (optional)
        if categories is not None:
            f.write(categories.astype(np.float32).tobytes())

    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"Created {output_path} ({size_mb:.2f} MB)")
    return output_path


def load_npy_coords(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load coordinates from .npy file.

    Supports:
    - Shape (2, N): row 0 = x, row 1 = y
    - Shape (N, 2): column 0 = x, column 1 = y
    """
    data = np.load(path)
    print(f"Loaded {path}: shape={data.shape}, dtype={data.dtype}")

    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got {data.ndim}D")

    # Determine orientation
    if data.shape[0] == 2:
        # Shape (2, N) - rows are x, y
        x = data[0].astype(np.float32)
        y = data[1].astype(np.float32)
    elif data.shape[1] == 2:
        # Shape (N, 2) - columns are x, y
        x = data[:, 0].astype(np.float32)
        y = data[:, 1].astype(np.float32)
    else:
        raise ValueError(f"Cannot interpret shape {data.shape} as coordinates")

    return x, y


def generate_synthetic_data(n_points: int = 100_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic clustered data for demo."""
    print(f"Generating {n_points:,} synthetic points...")
    start = time.time()

    np.random.seed(42)

    n_clusters = 20
    points_per_cluster = n_points // n_clusters

    x_all = []
    y_all = []
    cat_all = []

    total_so_far = 0
    for i in range(n_clusters):
        # Last cluster gets remaining points
        if i == n_clusters - 1:
            n = n_points - total_so_far
        else:
            n = points_per_cluster
        total_so_far += n

        # Cluster center
        cx = np.random.randn() * 500
        cy = np.random.randn() * 500

        # Points with exponential spread
        angle = np.random.rand(n) * 2 * np.pi
        radius = np.random.exponential(50, n)

        x_all.append(cx + np.cos(angle) * radius)
        y_all.append(cy + np.sin(angle) * radius)
        cat_all.append(np.full(n, i % 5))

    x = np.concatenate(x_all).astype(np.float32)
    y = np.concatenate(y_all).astype(np.float32)
    categories = np.concatenate(cat_all).astype(np.float32)

    elapsed = time.time() - start
    print(f"Generated {len(x):,} points in {elapsed:.2f}s")

    return x, y, categories


class QuietHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler that only logs errors."""

    def log_message(self, format, *args):
        # Only log non-200 responses
        if len(args) > 1 and args[1] != "200":
            super().log_message(format, *args)


def serve(directory: str, port: int = 8080, open_browser: bool = True):
    """Start HTTP server."""
    os.chdir(directory)

    # Try to find an available port
    for p in range(port, port + 10):
        try:
            with socketserver.TCPServer(("", p), QuietHTTPHandler) as httpd:
                httpd.allow_reuse_address = True
                url = f"http://localhost:{p}"
                print(f"\nServing at {url}")
                print("Press Ctrl+C to stop\n")

                if open_browser:
                    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

                httpd.serve_forever()
        except OSError as e:
            if "Address already in use" in str(e):
                print(f"Port {p} in use, trying {p + 1}...")
                continue
            raise
        except KeyboardInterrupt:
            print("\nStopping server...")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Serve regl-scatterplot TMAP demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Your .npy file (shape 2xN)
    python serve_demo.py --npy x_y_100k_coords.npy

    # Synthetic data
    python serve_demo.py --synthetic 1000000

    # NPZ layout file
    python serve_demo.py --layout my_layout.npz
        """,
    )
    parser.add_argument("--npy", help="Path to .npy coordinates file (shape 2xN or Nx2)")
    parser.add_argument("--layout", help="Path to .npz layout file")
    parser.add_argument("--synthetic", type=int, help="Generate N synthetic points")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")

    args = parser.parse_args()

    demo_dir = Path(__file__).parent

    # Load or generate data
    categories = None

    if args.npy:
        x, y = load_npy_coords(args.npy)
    elif args.layout:
        print(f"Loading {args.layout}...")
        data = np.load(args.layout)
        x = data["x"].astype(np.float32)
        y = data["y"].astype(np.float32)
        if "categories" in data:
            categories = data["categories"].astype(np.float32)
    elif args.synthetic:
        x, y, categories = generate_synthetic_data(args.synthetic)
    else:
        # Default: 100k synthetic points
        print("No data specified, generating 100k synthetic points...")
        x, y, categories = generate_synthetic_data(1_000_000)

    n_points = len(x)

    # Print stats
    print(f"\nLayout stats:")
    print(f"  Points: {n_points:,}")
    print(f"  X range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"  Y range: [{y.min():.2f}, {y.max():.2f}]")

    # Create binary file
    bin_path = demo_dir / "layout.bin"
    create_binary_layout(x, y, categories, str(bin_path))

    # Start server
    serve(str(demo_dir), args.port, not args.no_browser)


if __name__ == "__main__":
    main()
