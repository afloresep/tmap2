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

from tmap.visualization import TmapViz


def generate_spiral_data(
    n_points: int = 100_000,
    n_turns: float = 4.0,
    noise: float = 0.02,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Generate a spiral dataset with synthetic metadata.

    Returns:
        Tuple of (x, y, metadata dict)
    """
    rng = np.random.default_rng(seed)

    # Spiral coordinates
    t = np.linspace(0, n_turns * 2 * np.pi, n_points)
    r = t / (n_turns * 2 * np.pi)  # Radius grows with angle

    x = r * np.cos(t) + rng.normal(0, noise, n_points)
    y = r * np.sin(t) + rng.normal(0, noise, n_points)

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

    return x.astype(np.float32), y.astype(np.float32), metadata


def main() -> None:
    """Generate visualization demo."""
    print("Generating spiral dataset...")
    x, y, metadata = generate_spiral_data(n_points=100_000)
    n_points = len(x)

    print(f"Created {n_points} points")

    # Create visualization
    viz = TmapViz()
    viz.title = "TMAP Visualization Demo"
    viz.point_size = 3.0
    viz.opacity = 0.85
    viz.background_color = "#ffffff"

    # Set coordinates
    viz.set_points(x, y)

    # Add continuous color layouts
    viz.add_color_layout(
        name="Activity",
        values=metadata["activity"].tolist(),
        categorical=False,
        color="viridis",
    )
    viz.add_color_layout(
        name="Molecular Weight",
        values=metadata["mw"].tolist(),
        categorical=False,
        color="plasma",
    )
    viz.add_color_layout(
        name="Radius",
        values=metadata["radius"].tolist(),
        categorical=False,
        color="coolwarm",
    )

    # Add categorical color layouts
    viz.add_color_layout(
        name="Cluster",
        values=metadata["cluster"].tolist(),
        categorical=True,
        color="tab10",
    )
    viz.add_color_layout(
        name="Quadrant",
        values=metadata["quadrant"].tolist(),
        categorical=True,
        color="Set2",
    )

    # Add labels for tooltips
    labels = [f"Point-{i:05d}" for i in range(n_points)]
    viz.add_label("ID", labels)

    # Save to file
    output_path = viz.save("./")
    print(f"\nVisualization saved to: {output_path}")
    print("\nOpen in a web browser to interact:")
    print("  - Scroll to zoom")
    print("  - Drag to pan")
    print("  - Hover for tooltips")
    print("  - Shift+drag for lasso selection")
    print("  - Use dropdown to change color scheme")


if __name__ == "__main__":
    main()
