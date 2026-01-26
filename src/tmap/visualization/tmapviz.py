from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional

import matplotlib.colors as mcolors
import numpy as np
from matplotlib import colormaps
from numpy.typing import NDArray

COLORMAPS = list(colormaps)


def _project_root() -> Path:
    """Return repository root (assumes src/tmap/visualization/...)."""
    return Path(__file__).resolve().parents[3]


def _load_js_sources() -> dict[str, str]:
    """Load raw JS sources from node_modules for inline embedding."""
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


def _normalize_coords(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Normalize coordinates to [-1, 1] preserving aspect ratio.
    This is actually required by regl-scatterplot.
    """
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
    return np.stack([x_norm, y_norm], axis=1).astype(np.float64)


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> list[float]:
    """Convert #RRGGBB to [r, g, b, a] floats in [0, 1]."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color!r}")
    return [int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)] + [alpha]



#TODO: Implement categorical=True preserves listed colors when available
def _colormap_to_hex(name:str) -> list[str]:
    """
    Convert a matplotlib colormap to a list of hex strings.
    """
    import matplotlib as mpl
    from matplotlib.colors import to_hex

    cmap = mpl.colormaps[name]
    hex_colors = [to_hex(cmap(i)) for i in range(cmap.N)]
    return hex_colors



@dataclass
class Column:
    name: str
    values: List[int | np.floating | str]
    role: Literal["layout", "label", "layout+label"]
    dtype: Literal["continuous", "categorical", "label"]
    color: Optional[str] = None


class TmapViz:
    def __init__(self) -> None:
        self.title: str = "MyTMAP"
        self.background_color: str = "#7A7A7A"
        self.point_color: str = "#4a9eff"
        self.point_size: float = 4.0
        self.opacity: float = 0.85

        self._points: List[list[float]] = []
        self._layout_keys: List[str] = []
        self._labels_keys: List[str] = []
        self._columns: dict[str, Column] = {}

    def add_layout(
        self,
        name: str,
        values: List[Any],
        categorical: bool = False,
        add_as_label: bool = True,
        color: Optional[str] = None,
    ) -> None:

        import matplotlib

        if isinstance(values, np.ndarray):
            values = values.tolist()
        else:
            values = list(values)
            
        _column_dtype = "categorical" if categorical else "continuous"


        if color is None:
            color = "tab10" if categorical else "viridis"

        if color not in COLORMAPS:
            raise ValueError(f"Color option not found. Choose from {list(matplotlib.colormaps)}")

        if categorical and color not in list(matplotlib.color_sequences):
            raise ValueError(
                f"Categorical layout requires a color scheme from {list(matplotlib.color_sequences)}"
            )

        if name not in self._layout_keys:
            self._layout_keys.append(name)

        if add_as_label and name not in self._labels_keys:
            self._labels_keys.append(name)
            self._columns[name] = Column(name, values, "layout+label", _column_dtype, color=color)
        else:
            self._columns[name] = Column(name, values, "layout", _column_dtype, color=color)

    def add_label(
        self,
        name: str,
        values: List[Any],
    ) -> None:
        if isinstance(values, np.ndarray):
            values = values.tolist()
        else:
            values = list(values)

        if name not in self._labels_keys:
            self._labels_keys.append(name)
        self._columns[name] = Column(name, values, "label", "label")

    @property
    def layouts(self) -> List[Column]:
        """Return layouts added."""
        return [self._columns[layout] for layout in self._layout_keys]

    @property
    def labels(self) -> List[Column]:
        """Return labels added."""
        return [self._columns[labels] for labels in self._labels_keys]

    def set_points(
        self,
        x: List[np.floating] | NDArray[np.floating],
        y: List[np.floating] | NDArray[np.floating],
    ) -> None:
        """
        Store and normalize point coordinates.

        x: X coordinates
        y: Y coordinates
        """
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        """
        ^^^^^^^
        Not really sure if converting here to np.array is the best idea. 
        JSON requires list so at the end we are doing list -> np.array -> list
        becasue we need to normalize. Maybe just asking for np.array is cleaner
        but I feel like a lot of people will be passing it as list and having the 
        type error could be annoying. 
        TODO: profile this to see impact 
        """


        if x_arr.shape != y_arr.shape:
            raise ValueError("x and y must have the same shape")
        if x_arr.ndim != 1 or y_arr.ndim != 1:
            raise ValueError(
                f"Both x and y should be 1 dimensional arrays, Got x: {x_arr.ndim}D and y: {y_arr.ndim}D"
            )

        normalized_coords = _normalize_coords(x_arr, y_arr)
        self._points = normalized_coords.tolist()

        for col in self._columns.values():
            if len(col.values) != len(self._points):
                raise ValueError(
                    f"Column '{col.name}' has {len(col.values)} values but there are "
                    f"{len(self._points)} points"
                )

    def render(self) -> str:
        """Return the full HTML string for the visualization."""
        if not self._points:
            raise ValueError("Call set_points() before rendering.")

        n_points = len(self._points)
        for col in self._columns.values():
            if len(col.values) != n_points:
                raise ValueError(
                    f"Column '{col.name}' has {len(col.values)} values but there are "
                    f"{n_points} points"
                )

        columns_payload: dict[str, dict[str, Any]] = {}
        colormaps_payload: dict[str, list[str]] = {}

        for name, col in self._columns.items():
            colormap_name = col.color if col.role in ("layout", "layout+label") else None

            columns_payload[name] = {
                "values": col.values,
                "dtype": col.dtype,
                "role": col.role,
                "colormap": colormap_name,
            }

            if colormap_name and colormap_name not in colormaps_payload:
                colormaps_payload[colormap_name] = _colormap_to_hex(
                    colormap_name)
                

        layout_options = list(self._layout_keys)
        label_options = [name for name in self._labels_keys if name in self._columns]
        initial_color = layout_options[0] if layout_options else None

        payload = {
            "title": self.title,
            "points": self._points,
            "pointColor": self.point_color,
            "pointSize": self.point_size,
            "opacity": self.opacity,
            "backgroundColor": _hex_to_rgba(self.background_color),
            "columns": columns_payload,
            "layoutOptions": layout_options,
            "labelOptions": label_options,
            "initialColor": initial_color,
            "colormaps": colormaps_payload,
        }

        runtime = _runtime_base64()
        payload_json = json.dumps(payload, separators=(",", ":"))

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
      background: #000;
      border-radius: 6px;
      font-size: 13px;
      color: #fff;
      min-width: 160px;
    }}
    #controls label {{
      font-size: 12px;
      color: #fff;
      display: block;
      margin-bottom: 4px;
    }}
    #controls select {{
      width: 100%;
      padding: 6px 8px;
      border-radius: 4px;
      border: 1px solid #fff;
      background: #000;
      color: #fff;
    }}
    #controls select option {{
      background: #000;
      color: #fff;
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
    <label for="color-select">Color By Layout</label>
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
const layoutOptions = payload.layoutOptions || [];
const labelOptions = payload.labelOptions || [];
const initialColor = layoutOptions.includes(payload.initialColor)
  ? payload.initialColor
  : (layoutOptions[0] || '');

const labelOnlyOptions = labelOptions.filter((name) => {{
  const col = columns[name];
  return col && col.role === 'label';
}});
const labelFallbackOptions = labelOptions.filter((name) => {{
  const col = columns[name];
  return col && col.role !== 'label';
}});
const tooltipFields = [...layoutOptions];
const tooltipSet = new Set(layoutOptions);
for (const name of labelOnlyOptions) {{
  if (!tooltipSet.has(name)) {{
    tooltipSet.add(name);
    tooltipFields.push(name);
  }}
}}

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
  if (!col) {{
    scatterplot.set({{ colorBy: null, pointColor: payload.pointColor || '#4a9eff' }});
    scatterplot.draw({{ x, y }});
    return;
  }}

  const cmap = (col.colormap && colormaps[col.colormap]) || ['#4a9eff'];

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

function getLabel(idx) {{
  for (const name of labelOnlyOptions) {{
    const col = columns[name];
    if (col && col.values[idx] !== undefined) {{
      return col.values[idx];
    }}
  }}
  for (const name of labelFallbackOptions) {{
    const col = columns[name];
    if (col && col.values[idx] !== undefined) {{
      return col.values[idx];
    }}
  }}
  return 'Point ' + idx;
}}

let mouseX = 0, mouseY = 0;
canvas.addEventListener('mousemove', (e) => {{ mouseX = e.clientX; mouseY = e.clientY; }});

function showTooltip(idx) {{
  tooltipLabel.textContent = getLabel(idx);

  let html = '';
  for (const name of tooltipFields) {{
    const col = columns[name];
    if (!col) continue;
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
}};
window.addEventListener('resize', resize);

for (const name of layoutOptions) {{
  const col = columns[name];
  if (!col) continue;
  const opt = document.createElement('option');
  opt.value = name;
  opt.textContent = name + ' (' + col.dtype + ')';
  colorSelect.appendChild(opt);
}}
if (initialColor) {{
  colorSelect.value = initialColor;
}}
colorSelect.addEventListener('change', (e) => applyColor(e.target.value));

applyColor(initialColor || '');
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
        import os 
        if not (self.title).endswith('.html'):
            title = self.title+'.html'
        else: title=self.title
        output_path = Path(os.path.join(path, title))
        output_path.write_text(self.render(), encoding="utf-8")
        return output_path
