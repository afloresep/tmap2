"""
Visualization module: Render layouts to interactive HTML.

Main API
--------
TmapViz is the primary class for creating visualizations:

    from tmap.visualization import TmapViz

    viz = TmapViz(title="My Data")
    viz.set_layout(coords)
    viz.add_column("label", labels, role="label")
    viz.add_column("value", values, dtype="continuous")
    viz.set_color("value", colormap="viridis")
    viz.save("output.html")

Features:
- WebGL rendering via regl-scatterplot (handles millions of points)
- Self-contained HTML output (no server required)
- Continuous and categorical color mapping
- Interactive tooltips with metadata
- Pan, zoom, and lasso selection

Colormaps
---------
Available colormaps:
- Sequential: viridis, plasma, inferno, magma, cividis
- Diverging: coolwarm, RdYlBu
- Categorical: tab10, tab20, Set1, Set2, Dark2, Paired
"""

from tmap.visualization.base import HTMLVisualizer, NodeMetadata, Visualizer
from tmap.visualization.colormaps import (
    get_colormap,
    map_categorical,
    map_continuous,
)
from tmap.visualization.simple_inline import SimpleInlineViz
from tmap.visualization.tmap_viz import TmapViz

__all__ = [
    # Main API
    "TmapViz",
    "SimpleInlineViz",
    # Colormap utilities
    "get_colormap",
    "map_continuous",
    "map_categorical",
    # Base classes (for extension)
    "Visualizer",
    "HTMLVisualizer",
    "NodeMetadata",
]
