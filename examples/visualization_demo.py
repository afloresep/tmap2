#!/usr/bin/env python3
"""
Demo of the TmapViz visualization.

This example creates a spiral dataset with synthetic metadata and generates
an interactive HTML visualization.

Usage:
    python examples/visualization_demo.py

Output:
    visualization_demo.html (open in browser)

Requirements:
    pip install -e ".[viz]"  # From repo root - installs Jinja2
"""

from __future__ import annotations

import numpy as np

from tmap.layout.types import Coordinates
from tmap.visualization import TmapViz


def generate_spiral_data(
    n_points: int = 500_000,
    n_turns: float = 4.0,
    noise: float = 0.02,
    seed: int = 42,
) -> tuple[Coordinates, dict[str, np.ndarray]]:
    """
    Generate a spiral dataset with synthetic metadata.

    Returns:
        Tuple of (coordinates, metadata dict)
    """
    rng = np.random.default_rng(seed)

    # Spiral coordinates
    t = np.linspace(0, n_turns * 2 * np.pi, n_points)
    r = t / (n_turns * 2 * np.pi)  # Radius grows with angle

    x = r * np.cos(t) + rng.normal(0, noise, n_points)
    y = r * np.sin(t) + rng.normal(0, noise, n_points)

    coords = Coordinates(
        x=x.astype(np.float32),
        y=y.astype(np.float32),
    )

    # Synthetic metadata
    metadata = {
        # Continuous values
        "radius": r,
        "angle": t,
        "activity": np.sin(t) * 0.5 + 0.5 + rng.normal(0, 0.1, n_points),
        "mw": 200 + 300 * r + rng.normal(0, 20, n_points),  # Molecular weight
        # Categorical values
        "cluster": (t // (np.pi / 2)).astype(int) % 8,  # 8 clusters
        "quadrant": np.where(
            x >= 0,
            np.where(y >= 0, "Q1", "Q4"),
            np.where(y >= 0, "Q2", "Q3"),
        ),
    }

    return coords, metadata


def main() -> None:
    """Generate visualization demo."""
    print("Generating spiral dataset...")
    coords, metadata = generate_spiral_data(n_points=500_000)


    print(f"Created {coords.n_nodes} points")

    # Create visualization
    viz = TmapViz(
        title="TMAP Visualization Demo",
        point_size=2,
        opacity=0.85,
        background_color="#ffffff",
    )

    # Set layout
    viz.set_layout(coords)

    # Add columns
    # Labels (shown as primary identifier in tooltip)
    labels = [f"Point-{i:05d}" for i in range(coords.n_nodes)]
    viz.add_column("id", labels, role="label")

    # Continuous columns
    viz.add_column("activity", metadata["activity"], dtype="continuous")
    viz.add_column("mw", metadata["mw"], dtype="continuous")
    viz.add_column("radius", metadata["radius"], dtype="continuous")

    # Categorical columns
    viz.add_column("cluster", metadata["cluster"], dtype="categorical")
    viz.add_column("quadrant", metadata["quadrant"], dtype="categorical")

    # Set color encoding (try different options)
    # Option 1: Continuous color
    viz.set_color("activity", colormap="viridis")

    # Option 2: Categorical color (uncomment to try)
    viz.set_color("cluster", colormap="tab10")

    # Option 3: Diverging colormap (uncomment to try)
    # viz.set_color("mw", colormap="coolwarm")

    # Save to file
    output_path = viz.save("visualization_demo.html")
    print(f"\nVisualization saved to: {output_path}")
    print("\nOpen in a web browser to interact:")
    print("  - Scroll to zoom")
    print("  - Drag to pan")
    print("  - Hover for tooltips")
    print("  - Shift+drag for lasso selection")
    print("  - Press 'R' to reset view")
    print("  - Press 'Escape' to clear selection")


if __name__ == "__main__":
    main()
