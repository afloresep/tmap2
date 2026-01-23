"""
Minimal offline scatterplot exporter.

This prototype builds a self-contained HTML file that renders a scatterplot
using regl-scatterplot without requiring a web server or external CDNs.

It inlines the necessary JS modules (regl-scatterplot, regl, pub-sub-es) by
embedding them as base64 strings and wiring them up via blob URLs in the
browser. The generated HTML can be opened directly via file://.

Unlike the full TmapViz, this class focuses on a simple, small-footprint API:
- add normalized x/y points
- add metadata columns (continuous, categorical, label)
- choose a color column
- get tooltips + a "Color By" dropdown in the output HTML
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from tmap.visualization.colormaps import COLORMAPS


@dataclass
class Column:
    """A data column for visualization."""

    name: str
    values: list[Any]
    dtype: Literal["continuous", "categorical", "label"]
    role: Literal["color", "size", "label", "metadata"] | None = None


def _project_root() -> Path:
    """Return repository root (assumes src/tmap/visualization/...)."""
    return Path(__file__).resolve().parents[3]


def _load_js_sources() -> dict[str, str]:
    """
    Load raw JS sources from node_modules.

    Raises:
        RuntimeError: if dependencies are missing.
    """
    root = _project_root()
    deps = {
        "regl": root / "node_modules" / "regl" / "dist" / "regl.min.js",
        "scatterplot": root
        / "node_modules"
        / "regl-scatterplot"
        / "dist"
        / "regl-scatterplot.esm.js",
        "pubsub": root / "node_modules" / "pub-sub-es" / "dist" / "index.js",
    }

    missing = [name for name, path in deps.items() if not path.exists()]
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(
            f"Missing JS dependencies: {missing_list}. "
            "Run `npm install regl-scatterplot` in the repo root to fetch node_modules."
        )

    return {name: path.read_text(encoding="utf-8") for name, path in deps.items()}


def _normalize_coords(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Normalize coordinates to [-1, 1] preserving aspect ratio."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)

    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())

    x_center = (x_max + x_min) / 2.0
    y_center = (y_max + y_min) / 2.0

    x_range = x_max - x_min
    y_range = y_max - y_min
    scale = max(x_range, y_range) / 2.0
    if scale == 0:
        scale = 1.0

    x_norm = (x - x_center) / scale
    y_norm = (y - y_center) / scale
    return np.stack([x_norm, y_norm], axis=1).astype(np.float32)


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> list[float]:
    """Convert #RRGGBB to [r, g, b, a] floats in [0, 1]."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color!r}")
    return [
        int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)
    ] + [alpha]


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _runtime_base64() -> dict[str, str]:
    """Cacheable helper to fetch + encode JS sources."""
    sources = _load_js_sources()
    return {
        "regl": _b64(sources["regl"]),
        "scatterplot": _b64(sources["scatterplot"]),
        "pubsub": _b64(sources["pubsub"]),
    }


@dataclass
class SimpleInlineViz:
    """Very small offline visualization helper."""

    title: str = "TMAP Inline Demo"
    background_color: str = "#0b1021"
    point_color: str = "#4a9eff"
    point_size: float = 4.0
    opacity: float = 0.85
    _points: list[list[float]] | None = field(init=False, default=None)
    _columns: dict[str, Column] = field(init=False, default_factory=dict)
    _label_column: str | None = field(init=False, default=None)
    _color_column: str | None = field(init=False, default=None)
    _color_colormap: str = field(init=False, default="viridis")

    def set_points(
        self,
        x: Sequence[float] | np.ndarray,
        y: Sequence[float] | np.ndarray,
    ) -> SimpleInlineViz:
        """Store and normalize point coordinates."""
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        if x_arr.shape != y_arr.shape:
            raise ValueError("x and y must have the same shape")

        normalized = _normalize_coords(x_arr, y_arr)
        # Convert to list of [x, y] for JSON embedding
        self._points = normalized.tolist()
        # Validate column lengths if they were added before points
        for col in self._columns.values():
            if len(col.values) != len(self._points):
                raise ValueError(
                    f"Column '{col.name}' has {len(col.values)} values but there are "
                    f"{len(self._points)} points"
                )
        return self

    def add_column(
        self,
        name: str,
        values: Sequence[Any] | np.ndarray,
        dtype: Literal["continuous", "categorical"] = "continuous",
        role: Literal["label"] | None = None,
    ) -> SimpleInlineViz:
        """
        Add a metadata column (continuous, categorical, or label).

        Length must match the number of points (if points are already set).
        """
        if isinstance(values, np.ndarray):
            values_list = values.tolist()
        else:
            values_list = list(values)

        if self._points is not None and len(values_list) != len(self._points):
            raise ValueError(
                f"Column '{name}' has {len(values_list)} values but there are "
                f"{len(self._points)} points"
            )

        col_dtype: Literal["continuous", "categorical", "label"]
        if role == "label":
            col_dtype = "label"
            self._label_column = name
        else:
            col_dtype = dtype

        self._columns[name] = Column(
            name=name,
            values=values_list,
            dtype=col_dtype,
            role=role,
        )
        return self

    def set_color(self, column: str, colormap: str = "viridis") -> SimpleInlineViz:
        """Choose which column drives the color mapping."""
        if column not in self._columns:
            available = ", ".join(self._columns.keys()) or "(none)"
            raise ValueError(
                f"Unknown column '{column}'. Available columns: {available}"
            )
        if self._columns[column].dtype == "label":
            raise ValueError("Label columns cannot be used for color encoding.")

        self._color_column = column
        self._color_colormap = colormap
        return self

    def render(self) -> str:
        """Return the full HTML string."""
        if self._points is None:
            raise ValueError("Call set_points() before rendering.")

        # If columns were added before points, validate lengths now
        for col in self._columns.values():
            if len(col.values) != len(self._points):
                raise ValueError(
                    f"Column '{col.name}' has {len(col.values)} values but there are "
                    f"{len(self._points)} points"
                )

        # Collect column metadata + values for the JS runtime
        used_colormaps: set[str] = set()
        columns_payload: dict[str, dict[str, Any]] = {}
        for name, col in self._columns.items():
            if col.dtype == "label":
                colormap_name = "tab10"
            elif name == self._color_column:
                colormap_name = self._color_colormap
            elif col.dtype == "categorical":
                colormap_name = "tab10"
            else:
                colormap_name = "viridis"

            used_colormaps.add(colormap_name)

            columns_payload[name] = {
                "values": col.values,
                "dtype": col.dtype,
                "colormap": colormap_name,
            }

        colormaps_payload = {
            name: colors for name, colors in COLORMAPS.items() if name in used_colormaps
        }

        payload = {
            "title": self.title,
            "points": self._points,
            "pointColor": self.point_color,
            "pointSize": self.point_size,
            "opacity": self.opacity,
            "backgroundColor": _hex_to_rgba(self.background_color),
            "columns": columns_payload,
            "labelColumn": self._label_column or "",
            "initialColor": self._color_column or "__none__",
            "colormaps": colormaps_payload,
        }

        runtime = _runtime_base64()
        payload_json = json.dumps(payload, separators=(",", ":"))

        # Inline JS bootstraps blob-based modules so file:// works.
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{self.title}</title>
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      background: {self.background_color};
      color: #e0e0e0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    #canvas {{
      width: 100vw;
      height: 100vh;
      display: block;
    }}
    #title {{
      position: fixed;
      top: 12px;
      left: 12px;
      padding: 6px 10px;
      border-radius: 6px;
      background: rgba(0, 0, 0, 0.4);
      font-size: 14px;
      letter-spacing: 0.4px;
    }}
    #controls {{
      position: fixed;
      top: 12px;
      right: 12px;
      padding: 8px 10px;
      background: rgba(0, 0, 0, 0.4);
      border-radius: 6px;
      font-size: 13px;
      color: #e0e0e0;
      min-width: 160px;
    }}
    #controls label {{
      font-size: 12px;
      color: #b0b0b0;
      display: block;
      margin-bottom: 4px;
    }}
    #controls select {{
      width: 100%;
      padding: 6px 8px;
      border-radius: 4px;
      border: 1px solid #444;
      background: #111827;
      color: #e0e0e0;
    }}
    #tooltip {{
      position: fixed;
      pointer-events: none;
      background: rgba(0, 0, 0, 0.7);
      border: 1px solid #333;
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 12px;
      color: #e0e0e0;
      max-width: 300px;
      display: none;
      box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    }}
    #tooltip.visible {{ display: block; }}
    #tooltip .label {{
      font-weight: 700;
      margin-bottom: 6px;
    }}
    #tooltip .row {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin: 2px 0;
    }}
    #tooltip .key {{ color: #9ca3af; }}
    #tooltip .value {{ color: #e5e7eb; text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  </style>
</head>
<body>
  <div id="title">{self.title}</div>
  <div id="controls">
    <label for="color-select">Color By</label>
    <select id="color-select"></select>
  </div>
  <canvas id="canvas"></canvas>
  <div id="tooltip">
    <div class="label" id="tooltip-label"></div>
    <div id="tooltip-content"></div>
  </div>
  <script id="payload" type="application/json">{payload_json}</script>
  <script type="module">
    const decode = (b64) => atob(b64);
    const mkModule = (code) =>
      URL.createObjectURL(new Blob([code], {{ type: 'application/javascript' }}));

    const reglSource = decode("{runtime['regl']}");
    const pubSubSource = decode("{runtime['pubsub']}");
    const scatterSourceRaw = decode("{runtime['scatterplot']}");

    // Wrap regl's UMD build into an ES module
    const reglModuleSource = `
const module = {{ exports: {{}} }};
const exports = module.exports;
const globalThisRef = typeof globalThis !== 'undefined' ? globalThis : window;
${{reglSource}}
export default module.exports || globalThisRef.createREGL || createREGL;
`;
    const reglUrl = mkModule(reglModuleSource);

    const pubSubUrl = mkModule(pubSubSource);
    const scatterSource = scatterSourceRaw
      .replace(/from "regl"/g, 'from "' + reglUrl + '"')
      .replace(/from 'regl'/g, 'from "' + reglUrl + '"')
      .replace(/from "pub-sub-es"/g, 'from "' + pubSubUrl + '"')
      .replace(/from 'pub-sub-es'/g, 'from "' + pubSubUrl + '"');
    const scatterUrl = mkModule(scatterSource);

    const runtime = `
import createScatterplot from '${{scatterUrl}}';

const payloadEl = document.getElementById('payload');
const payload = JSON.parse(payloadEl.textContent || '{{}}');

const coords = payload.points || [];
const columns = payload.columns || {{}};
const colormaps = payload.colormaps || {{}};
const labelColumn = payload.labelColumn || "";
const initialColor = payload.initialColor || "__none__";

const canvas = document.getElementById('canvas');
const colorSelect = document.getElementById('color-select');
const tooltip = document.getElementById('tooltip');
const tooltipLabel = document.getElementById('tooltip-label');
const tooltipContent = document.getElementById('tooltip-content');

const scatterplot = createScatterplot({{
  canvas,
  width: canvas.clientWidth || window.innerWidth,
  height: canvas.clientHeight || window.innerHeight,
  pointSize: payload.pointSize || 4,
  opacity: payload.opacity ?? 0.85,
  backgroundColor: payload.backgroundColor || [0, 0, 0, 1],
  pointColor: payload.pointColor || '#4a9eff',
  colorBy: null,
  lassoInitiator: false,
}});

const x = coords.map((p) => p[0]);
const y = coords.map((p) => p[1]);
const zeroArray = new Array(x.length).fill(0);

function applyColor(columnName) {{
  const col = columns[columnName];
  if (!col || columnName === '__none__') {{
    scatterplot.set({{ colorBy: null, pointColor: payload.pointColor || '#4a9eff' }});
    scatterplot.draw({{ x, y }});
    return;
  }}

  const cmap = colormaps[col.colormap] || colormaps.viridis || ['#4a9eff'];

  if (col.dtype === 'categorical') {{
    const mapping = new Map();
    const z = new Array(x.length);
    for (let i = 0; i < x.length; i++) {{
      const v = col.values[i];
      if (!mapping.has(v)) mapping.set(v, mapping.size);
      z[i] = mapping.get(v);
    }}
    scatterplot.set({{ colorBy: 'valueA', pointColor: cmap }});
    scatterplot.draw({{ x, y, z, w: zeroArray }});
  }} else {{
    const nums = col.values.map(Number);
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < nums.length; i++) {{
      const v = nums[i];
      if (Number.isFinite(v)) {{
        if (v < min) min = v;
        if (v > max) max = v;
      }}
    }}
    const range = max === min ? 1 : max - min;
    const w = nums.map((v) => ((v - min) / range));
    scatterplot.set({{ colorBy: 'valueB', pointColor: cmap }});
    scatterplot.draw({{ x, y, z: zeroArray, w }});
  }}
}}

function formatValue(val) {{
  if (typeof val === 'number') {{
    const abs = Math.abs(val);
    if (abs >= 1000 || (abs > 0 && abs < 0.01)) return val.toExponential(2);
    return val.toFixed(3);
  }}
  return String(val);
}}

let mouseX = 0, mouseY = 0;
canvas.addEventListener('mousemove', (e) => {{ mouseX = e.clientX; mouseY = e.clientY; }});

function showTooltip(idx) {{
  const labelVal = labelColumn && columns[labelColumn] ? columns[labelColumn].values[idx] : undefined;
  tooltipLabel.textContent = labelVal !== undefined ? labelVal : 'Point ' + idx;

  let html = '';
  for (const [name, col] of Object.entries(columns)) {{
    if (name === labelColumn) continue;
    const val = col.values[idx];
    html += '<div class="row"><span class="key">' + name + '</span><span class="value">' + formatValue(val) + '</span></div>';
  }}
  tooltipContent.innerHTML = html;

  tooltip.classList.add('visible');
  const {{ width, height }} = tooltip.getBoundingClientRect();
  let left = mouseX + 12;
  let top = mouseY + 12;
  if (left + width > window.innerWidth - 8) left = mouseX - width - 12;
  if (top + height > window.innerHeight - 8) top = mouseY - height - 12;
  tooltip.style.left = left + 'px';
  tooltip.style.top = top + 'px';
}}

function hideTooltip() {{
  tooltip.classList.remove('visible');
}}

scatterplot.subscribe('pointover', (idx) => showTooltip(idx));
scatterplot.subscribe('pointout', hideTooltip);

const resize = () => {{
  const {{ width, height }} = canvas.getBoundingClientRect();
  scatterplot.set({{ width, height }});
  scatterplot.draw(points);
}};
window.addEventListener('resize', resize);

const defaultOption = document.createElement('option');
defaultOption.value = '__none__';
defaultOption.textContent = 'None (uniform)';
colorSelect.appendChild(defaultOption);
for (const [name, col] of Object.entries(columns)) {{
  if (col.dtype === 'label') continue;
  const opt = document.createElement('option');
  opt.value = name;
  opt.textContent = name + ' (' + col.dtype + ')';
  colorSelect.appendChild(opt);
}}
colorSelect.value = initialColor && columns[initialColor] ? initialColor : '__none__';
colorSelect.addEventListener('change', (e) => applyColor(e.target.value));

applyColor(colorSelect.value);
`;

    const runtimeUrl = mkModule(runtime);
    import(runtimeUrl);
  </script>
</body>
</html>
"""
        return html

    def save(self, path: str | Path) -> Path:
        """Write HTML to disk and return the path."""
        output_path = Path(path)
        output_path.write_text(self.render(), encoding="utf-8")
        return output_path
