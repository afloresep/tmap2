"""
Tiny demo for generating a single-file HTML scatterplot that works via file://.

Usage:
    python examples/simple_inline_demo.py
Then open examples/simple_inline_demo.html directly in a browser (double-click).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tmap.visualization import SimpleInlineViz


def main() -> None:
    rng = np.random.default_rng(0)
    n = 800

    angles = rng.uniform(0, 2 * np.pi, n)
    radii = rng.uniform(0.1, 1.0, n)

    x = radii * np.cos(angles) + rng.normal(0, 0.05, n)
    y = radii * np.sin(angles) + rng.normal(0, 0.05, n)

    # Synthetic metadata
    mw = rng.normal(350, 40, n)  # continuous
    activity = rng.uniform(-1, 1, n)  # continuous
    cluster = rng.choice(["A", "B", "C", "D"], size=n)  # categorical
    labels = [f"Point-{i:04d}" for i in range(n)]

    viz = SimpleInlineViz(
        title="TMAP Inline Demo",
        background_color="#0b1021",
        point_color="#8bd7ff",
        point_size=3.5,
        opacity=0.9,
    )

    viz.set_points(x, y)
    viz.add_column("mw", mw, dtype="continuous")
    viz.add_column("activity", activity, dtype="continuous")
    viz.add_column("cluster", cluster, dtype="categorical")
    viz.add_column("id", labels, role="label")
    viz.set_color("cluster", colormap="tab10")

    output = viz.save(Path("examples/simple_inline_demo.html"))
    print(f"Wrote {output}")
    print("Open the HTML directly (file://) to confirm it renders offline.")


if __name__ == "__main__":
    main()
