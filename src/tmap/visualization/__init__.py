"""
Visualization module: Render layouts to HTML/images.

DESIGN: Minimal abstraction here
--------------------------------
Unlike Index and Layout, visualization doesn't need complex
inheritance hierarchies. You'll likely have:
- HTMLVisualizer (main one, interactive)
- Maybe a PNG/SVG exporter for static images

The Visualizer base class is simple - just render(coords, metadata).


TEMPLATE APPROACH
-----------------
For HTML output, we use Jinja2 templates. This separates:
- Python logic (data preparation)
- HTML/JS/CSS (presentation)

The template lives in templates/tmap.html and includes:
- Canvas rendering with WebGL (for performance)
- Pan/zoom controls
- Hover tooltips
- Search/filter (your "find node" feature)
- Copy data (your "copy SMILES" feature)

You customize by:
1. Editing the template (HTML/JS)
2. Passing different metadata to render()
"""

from tmap.visualization.base import Visualizer

__all__ = ["Visualizer"]

# Uncomment as you implement:
# from tmap.visualization.html import HTMLVisualizer
