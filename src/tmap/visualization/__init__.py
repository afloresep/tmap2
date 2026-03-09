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
    viz.write_html("output.html")

Features:
- WebGL rendering via regl-scatterplot (handles millions of points)
- Binary-encoded data (gzip-compressed typed arrays) for fast loading
- Self-contained HTML output (no server required)
- Continuous and categorical color mapping
- Interactive tooltips with metadata
- Pan, zoom, and lasso selection
- Filtering and search (via ``filterable`` / ``searchable`` properties)
- Configurable pinned cards (``configure_card()``)
- SMILES molecule rendering (``add_smiles()``)
- Image thumbnails (``add_images()``)
- Protein 3D structures via Mol* (``add_protein_ids()``)
- Fetch-based serving for very large datasets (``serve()`` / ``write_static()``)

Colormaps
---------
Available colormaps:
- Sequential: viridis, plasma, inferno, magma, cividis
- Diverging: coolwarm, RdYlBu
- Categorical: tab10, tab20, Set1, Set2, Dark2, Paired
"""

from typing import Any

from tmap.visualization.tmapviz import TmapViz

__all__ = ["TmapViz", "plot_static"]


def __getattr__(name: str) -> Any:  # noqa: N807
    if name == "plot_static":
        from tmap.visualization.static import plot_static

        return plot_static
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
