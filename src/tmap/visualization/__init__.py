"""
Visualization module: Render layouts to interactive HTML.

Main API
--------
TmapViz is the primary class for creating visualizations:

    from tmap.visualization import TmapViz

    viz = TmapViz()
    viz.title = "My Data"
    viz.set_points(x, y)
    viz.add_label("label", labels)
    viz.add_color_layout("value", values, categorical=False)
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
from tmap.visualization.tmapviz import BINARY_THRESHOLD, TmapViz

__all__ = ["TmapViz", "BINARY_THRESHOLD"]
