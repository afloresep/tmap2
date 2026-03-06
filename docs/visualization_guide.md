# Visualization Guide: Creating Interactive TMAP Visualizations

This guide explains how to create interactive HTML visualizations from your TMAP layouts using the `TmapViz` class.

---

## Overview

The visualization module renders TMAP layouts as self-contained, interactive HTML files:

- **WebGL rendering** via regl-scatterplot (handles millions of points)
- **Self-contained** - no server required, just open the HTML file
- **Interactive** - pan, zoom, lasso selection, tooltips
- **Color mapping** - continuous (heatmaps) and categorical (distinct groups)
- **Binary encoding** - gzip-compressed typed arrays for fast loading
- **Large dataset support** - server mode for 1M+ points
- **Filtering and search** - declarative column-level filtering and text search
- **Domain extensions** - SMILES rendering, image thumbnails, protein 3D viewer

---

## Quick Start

```python
from tmap.visualization import TmapViz

# Create visualizer
viz = TmapViz()
viz.title = "My TMAP"

# Set coordinates (from layout)
viz.set_points(x, y)

# Add color column (continuous)
viz.add_color_layout("activity", activity_values, categorical=False, color="viridis")

# Add labels for tooltips
viz.add_label("name", compound_names)

# Save to HTML
viz.write_html("./output")  # Creates "My TMAP.html"
```

---

## The TmapViz Class

### Creating a Visualizer

```python
from tmap.visualization import TmapViz

viz = TmapViz()

# Configure appearance
viz.title = "Chemical Space"        # Filename and title
viz.background_color = "#7A7A7A"    # Gray background (default)
viz.point_color = "#4a9eff"         # Default point color (blue)
viz.point_size = 4.0                # Point radius in pixels
viz.opacity = 0.85                  # Point opacity [0-1]
```

### Setting Coordinates

```python
# From layout_from_lsh_forest output
x, y, s, t = layout_from_lsh_forest(lsh, cfg)
viz.set_points(x, y)

# Or from ForceDirectedLayout
coords = layout.compute(tree)
viz.set_points(coords.x, coords.y)
```

**Note:** Coordinates are automatically normalized to [-1, 1] for WebGL rendering.

### Tree Edges and Edge Style

If your layout function returns edge arrays (`s`, `t`), pass them to `TmapViz` to
render MST edges in the HTML visualization.

```python
# Typical layout output includes edges
x, y, s, t = layout_from_lsh_forest(lsh, cfg)

viz.set_points(x, y)
viz.set_edges(s, t)            # Enable edge rendering from layout edges
viz.show_edges = True          # Default True; set False to hide edges

# Optional edge appearance tuning
viz.set_edge_style(
    color="#000000",           # Hex color (#rgb or #rrggbb)
    width=2.0,                 # Line width in CSS pixels
    opacity=0.5,               # Edge alpha in [0, 1]
)
```

For large datasets, keep edge width moderate and use binary mode when possible.

---

## Adding Data Columns

### Color Layouts (Continuous)

For numeric values that should be shown as a gradient:

```python
viz.add_color_layout(
    name="activity",           # Column name (shown in UI)
    values=activity_values,    # List or array of numeric values
    categorical=False,         # Continuous colormap
    add_as_label=True,        # Also show in tooltip (default True)
    color="viridis",          # Colormap name
)
```

**Available continuous colormaps:**

- Sequential: `viridis`, `plasma`, `inferno`, `magma`, `cividis`
- Diverging: `coolwarm`, `RdYlBu`, `RdBu`
- Single-hue: `Blues`, `Reds`, `Greens`

**NaN behavior:** If a continuous column contains `NaN`, TmapViz emits a warning and
renders those points in black (`#000000`) by default.

### Color Layouts (Categorical)

For discrete groups or categories:

```python
viz.add_color_layout(
    name="cluster",
    values=cluster_labels,     # e.g., ["A", "A", "B", "C", ...]
    categorical=True,          # Discrete colors
    color="tab10",             # Categorical colormap
)
```

**Available categorical colormaps:**

- `tab10` (10 colors, default)
- `tab20` (20 colors)
- `Set1`, `Set2`, `Set3`
- `Dark2`, `Paired`

**Warning:** The number of unique values must not exceed the colormap's color count.

### Labels (Tooltip Only)

For text that appears in tooltips but doesn't affect coloring:

```python
viz.add_label(
    name="compound_name",
    values=names,              # List of strings
)
```

### SMILES (Molecular Structures)

For molecular visualization with structure rendering in tooltips:

```python
viz.add_smiles(
    name="structure",
    values=smiles_list,        # List of SMILES strings
)
```

This automatically uses the `smiles.html.j2` template which includes a SMILES renderer.

**Note:** Only one SMILES column is supported per visualization.

---

## Rendering and Saving

### Quick Save

```python
# Save to directory (uses viz.title as filename)
output_path = viz.write_html("./output")
print(f"Saved to: {output_path}")
```

### Manual Rendering

```python
# Get HTML string (for custom handling)
html = viz.to_html()

# Write manually
with open("custom_name.html", "w") as f:
    f.write(html)
```

### Large Datasets

All output uses binary encoding by default (gzip-compressed typed arrays,
uint16 quantized coordinates). For very large datasets (1M+ points), use
the server-based approach which loads columns lazily via HTTP fetch:

```python
# Local dev server (lazy column loading)
viz.serve(port=8050)

# Write static files for hosting (nginx, S3, etc.)
viz.write_static("dist/my_tmap/")
```

**Binary encoding benefits (built-in for all outputs):**

- 4x smaller file size (quantized coordinates)
- Faster loading (gzip compression)
- Non-blocking decoding (WebWorker)

---

## Interactive Features

The generated HTML includes these interactive features:

| Feature | Control |
|---------|---------|
| **Pan** | Click and drag |
| **Zoom** | Mouse wheel |
| **Hover** | Shows tooltip with labels |
| **Lasso selection** | Shift + drag |
| **Color selector** | Dropdown in top-left |
| **Reset view** | Double-click |

---

## Complete Example

```python
import numpy as np
from tmap import MinHash, LSHForest
from tmap.layout import layout_from_lsh_forest, LayoutConfig
from tmap.visualization import TmapViz

# Sample data
np.random.seed(42)
n_samples = 1000
fingerprints = (np.random.rand(n_samples, 2048) < 0.1).astype(np.uint8)
activities = np.random.rand(n_samples) * 100
clusters = np.random.choice(["Active", "Inactive", "Unknown"], n_samples)
names = [f"Compound_{i}" for i in range(n_samples)]

# 1. Encode and index
mh = MinHash(num_perm=128, seed=42)
sigs = mh.batch_from_binary_array(fingerprints)

lsh = LSHForest(d=128, l=64)
lsh.batch_add(sigs)
lsh.index()

# 2. Compute layout
cfg = LayoutConfig()
cfg.k = 20
cfg.kc = 50
cfg.deterministic = True
cfg.seed = 42

x, y, s, t = layout_from_lsh_forest(lsh, cfg)

# 3. Create visualization
viz = TmapViz()
viz.title = "Chemical Space"
viz.point_size = 3.0
viz.opacity = 0.8

# Set coordinates
viz.set_points(x, y)
viz.set_edges(s, t)
viz.set_edge_style(color="#111111", width=1.5, opacity=0.35)

# Add continuous coloring by activity
viz.add_color_layout("Activity", activities, categorical=False, color="viridis")

# Add categorical coloring by cluster
viz.add_color_layout("Cluster", clusters, categorical=True, color="Set1")

# Add labels
viz.add_label("Name", names)

# Save
viz.write_html("./")
```

---

## Configuration Reference

### TmapViz Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `title` | str | "MyTMAP" | HTML title and filename |
| `background_color` | str | "#7A7A7A" | Canvas background (hex) |
| `point_color` | str | "#4a9eff" | Default point color (hex) |
| `point_size` | float | 4.0 | Point radius in pixels |
| `opacity` | float | 0.85 | Point opacity [0-1] |
| `show_edges` | bool | True | Render edges when edge arrays are set |
| `edge_color` | str | "#000000" | Edge color (hex) |
| `edge_width` | float | 2.0 | Edge line width in pixels |
| `edge_opacity` | float | 0.5 | Edge opacity [0-1] |

### add_color_layout Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | Column name |
| `values` | list/array | required | Data values |
| `categorical` | bool | False | True for discrete, False for continuous |
| `add_as_label` | bool | True | Show in tooltip |
| `color` | str | auto | Colormap name |

### set_edges Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `s` | list/ndarray | required | Source point indices for edges |
| `t` | list/ndarray | required | Target point indices for edges |

### set_edge_style Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `color` | str | None | Edge color override (`#rgb` or `#rrggbb`) |
| `width` | float | None | Edge width override (must be > 0) |
| `opacity` | float | None | Edge opacity override in [0, 1] |

### write_html Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str/Path | required | File path (.html) or directory |

### write_static Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str/Path | required | Directory for data files + HTML shell |
| `template_name` | str | `"base.html.j2"` | Template for the HTML shell |

### serve Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `port` | int | 8050 | TCP port for local HTTP server |
| `open_browser` | bool | True | Auto-open browser |

---

## Performance Tips

### For Large Datasets (100K-500K points)

```python
viz.point_size = 2.0       # Smaller points
viz.opacity = 0.6          # More transparent
```

### For Very Large Datasets (1M+ points)

```python
# Serve locally (lazy column loading via HTTP fetch)
viz.serve(port=8050)

# Or write static files for hosting
viz.write_static("dist/my_tmap/")
```

### For Publication Quality

```python
viz.background_color = "#FFFFFF"  # White background
viz.point_size = 5.0              # Larger points
viz.opacity = 0.9                 # Less transparent
```

---

## Troubleshooting

### "Missing JS dependencies" Error

This means the vendored JavaScript files aren't found. Reinstall the package:

```bash
pip install --force-reinstall tmap
```

### "Jinja2 is required" Error

Install with visualization extras:

```bash
pip install tmap[viz]
```

### Points Not Visible

1. Check that coordinates are valid (not NaN/Inf)
2. Try increasing `point_size`
3. Check `opacity` is > 0

### Colormap Not Found

Use `matplotlib.colormaps` to list available colormaps:

```python
import matplotlib
print(list(matplotlib.colormaps))
```

### Too Many Categories for Colormap

Categorical colormaps have limited colors. Either:

1. Use a larger colormap (`tab20` instead of `tab10`)
2. Aggregate categories
3. Switch to continuous coloring

---

## Templates

All templates use binary encoding by default (gzip-compressed typed arrays).
The base template is selected automatically; specialized templates extend it
for domain-specific features.

| Template | Use Case |
|----------|----------|
| `base.html.j2` | Default visualization (binary encoded) |
| `smiles.html.j2` | Molecular structures (auto-selected with SMILES) |
| `images.html.j2` | Image thumbnails in tooltips (auto-selected with images) |
| `protein.html.j2` | Protein 3D viewer + UniProt metadata (auto-selected with protein IDs) |
| `server.html.j2` | Backward-compatible alias to `base.html.j2` |
| `afdb.html.j2` | AlphaFold DB cluster explorer (extends base template) |

Templates are auto-selected based on registered columns (SMILES, images,
protein IDs). You can override manually:

```python
html = viz.to_html(template_name="smiles.html.j2")
```

---

## Related Documentation

- [MinHash Guide](minhash_guide.md) - Data encoding
- [LSHForest Guide](lshforest_guide.md) - Building the index
- [Graph Guide](graph_guide.md) - MST construction
- [Layout Guide](layout_guide.md) - Computing coordinates
