from __future__ import annotations

import base64
import gzip
import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from matplotlib import colormaps
from numpy.typing import NDArray

try:
    from jinja2 import Environment, PackageLoader, select_autoescape

    _JINJA_AVAILABLE = True
except ImportError:
    _JINJA_AVAILABLE = False

COLORMAPS = list(colormaps)
TEMPLATES_DIR = Path(__file__).parent / "templates"
VENDOR_DIR = Path(__file__).parent / "vendor"

# Default threshold for switching to binary mode (number of points)
BINARY_THRESHOLD = 500_000


def _project_root() -> Path:
    """Return repository root (assumes src/tmap/visualization/...)."""
    return Path(__file__).resolve().parents[3]


def _load_js_sources() -> dict[str, str]:
    """Load raw JS sources for inline embedding.

    First tries vendored files (included in package), then falls back
    to node_modules (for development).
    """
    # Vendored files (preferred - included in package)
    vendor_deps = {
        "regl": VENDOR_DIR / "regl.min.js",
        "scatterplot": VENDOR_DIR / "regl-scatterplot.esm.js",
        "pubsub": VENDOR_DIR / "pub-sub-es.js",
    }

    # Check if vendored files exist
    if all(path.exists() for path in vendor_deps.values()):
        return {name: path.read_text(encoding="utf-8") for name, path in vendor_deps.items()}

    # Fallback to node_modules (for development)
    # TODO(ISS-006): Remove node_modules fallback once vendored files are stable
    root = _project_root()
    node_deps = {
        "regl": root / "node_modules" / "regl" / "dist" / "regl.min.js",
        "scatterplot": root
        / "node_modules"
        / "regl-scatterplot"
        / "dist"
        / "regl-scatterplot.esm.js",
        "pubsub": root / "node_modules" / "pub-sub-es" / "dist" / "index.js",
    }

    missing = [name for name, path in node_deps.items() if not path.exists()]
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(
            f"Missing JS dependencies: {missing_list}. "
            "Vendored files not found and node_modules unavailable. "
            "This is likely a packaging issue - please reinstall the package."
        )

    return {name: path.read_text(encoding="utf-8") for name, path in node_deps.items()}


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


@lru_cache(maxsize=1)
def _get_jinja_env() -> Environment:
    """Get or create a cached Jinja2 environment for templates."""
    if not _JINJA_AVAILABLE:
        raise ImportError(
            "Jinja2 is required for template rendering. Install full dependencies with: pip install -e ."
        )
    return Environment(
        loader=PackageLoader("tmap.visualization", "templates"),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


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
    return cast(NDArray[np.float64], np.stack([x_norm, y_norm], axis=1).astype(np.float64))


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> list[float]:
    """Convert #RRGGBB to [r, g, b, a] floats in [0, 1]."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color!r}")
    return [int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)] + [alpha]


def _normalize_hex_color(color: str) -> str:
    """Normalize color to #rrggbb format."""
    if not isinstance(color, str):
        raise ValueError("Color must be a hex string like '#rrggbb'.")

    normalized = color.strip().lstrip("#")
    if len(normalized) == 3:
        normalized = "".join(ch * 2 for ch in normalized)

    if len(normalized) != 6:
        raise ValueError(f"Invalid hex color: {color!r}")

    try:
        int(normalized, 16)
    except ValueError as exc:
        raise ValueError(f"Invalid hex color: {color!r}") from exc

    return f"#{normalized.lower()}"


def _hex_to_css_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert #RRGGBB + alpha to a CSS rgba(...) color string."""
    rgb = _hex_to_rgba(hex_color, alpha)
    r, g, b = (int(round(channel * 255)) for channel in rgb[:3])
    alpha_str = f"{alpha:.6f}".rstrip("0").rstrip(".")
    if alpha_str == "":
        alpha_str = "0"
    return f"rgba({r}, {g}, {b}, {alpha_str})"


# TODO(ISS-014): Implement categorical=True preserves listed colors when available
def _colormap_to_hex(name: str) -> list[str]:
    """
    Convert a matplotlib colormap to a list of hex strings.
    """
    import matplotlib as mpl
    from matplotlib.colors import to_hex

    cmap = mpl.colormaps[name]
    hex_colors = [to_hex(cmap(i)) for i in range(cmap.N)]
    return hex_colors


def _contains_nan(values: Sequence[Any]) -> bool:
    """Return True when values contain at least one NaN."""
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(np.isnan(arr).any())


def _to_json_safe(value: Any) -> Any:
    """Convert values to JSON-safe types, mapping non-finite numbers to null."""
    if isinstance(value, np.ndarray):
        return [_to_json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if np.isfinite(f) else None
    return value


@dataclass
class Column:
    name: str
    values: Sequence[int | np.floating | str]
    role: Literal["layout", "label", "layout+label", "smiles"]
    dtype: Literal["continuous", "categorical", "label", "smiles"]
    color: str | None = None


def _pack_coords_binary(points: np.ndarray, bits: int = 16) -> bytes:
    """Pack normalized [-1,1] coordinates as gzip-compressed quantized integers."""
    if bits == 16:
        max_val = 65535
        quantized = ((points.astype(np.float64) + 1.0) * (max_val / 2.0)).astype(np.uint16)
    else:
        max_val = 4294967295
        quantized = ((points.astype(np.float64) + 1.0) * (max_val / 2.0)).astype(np.uint32)
    raw = quantized.flatten().tobytes()
    return gzip.compress(raw, compresslevel=6)


def _pack_numeric_binary(values: np.ndarray, dtype: str = "float32") -> bytes:
    """Pack numeric column as gzip-compressed typed array."""
    arr: NDArray[np.float32] | NDArray[np.int32]
    if dtype == "float32":
        arr = values.astype(np.float32)
    elif dtype == "int32":
        arr = values.astype(np.int32)
    else:
        arr = values.astype(np.float32)
    return gzip.compress(arr.tobytes(), compresslevel=6)


def _pack_categorical_binary(values: Sequence[Any]) -> tuple[bytes, list[str]]:
    """Pack categorical column using dictionary encoding."""
    unique_values: list[str] = []
    value_to_idx: dict[str, int] = {}
    indices: NDArray[np.uint32] = np.empty(len(values), dtype=np.uint32)

    for i, v in enumerate(values):
        s = str(v)
        if s not in value_to_idx:
            value_to_idx[s] = len(unique_values)
            unique_values.append(s)
        indices[i] = value_to_idx[s]

    compressed = gzip.compress(indices.tobytes(), compresslevel=6)
    return compressed, unique_values


class TmapViz:
    """Interactive scatter-plot visualization backed by regl-scatterplot.

    Supports continuous and categorical color layouts, label tooltips,
    and optional SMILES molecule rendering. Outputs self-contained HTML.

    Attributes:
        title: Page title and default filename stem.
        background_color: Hex background color (default ``"#7A7A7A"``).
        point_color: Default hex point color (default ``"#4a9eff"``).
        point_size: Base point radius in pixels (default ``4.0``).
        opacity: Point opacity in ``[0, 1]`` (default ``0.85``).
        edge_color: Edge hex color (default ``"#000000"``).
        edge_opacity: Edge opacity in ``[0, 1]`` (default ``0.5``).
        edge_width: Edge width in CSS pixels (default ``2.0``).
    """

    def __init__(self) -> None:
        self.title: str = "MyTMAP"
        self.background_color: str = "#7A7A7A"
        self.point_color: str = "#4a9eff"
        self.point_size: float = 4.0
        self.opacity: float = 0.85

        self.edge_color: str = "#000000"
        self.edge_opacity: float = 0.5
        self.edge_width: float = 2.0

        # Store both formats for flexibility
        self._points: list[list[float]] = []
        self._points_array: np.ndarray | None = None  # Shape: (n, 2)
        self._edges_s: np.ndarray | None = None
        self._edges_t: np.ndarray | None = None
        self._layout_keys: list[str] = []
        self._labels_keys: list[str] = []
        self._smiles_column: str | None = None
        self._columns: dict[str, Column] = {}

    def add_color_layout(
        self,
        name: str,
        values: list[Any],
        categorical: bool = False,
        add_as_label: bool = True,
        color: str | None = None,
    ) -> None:

        import matplotlib

        if isinstance(values, np.ndarray):
            values = values.tolist()
        else:
            values = list(values)

        # Default to continuous because it will give less issues and having to pass
        # always the type can be annoying...
        _column_dtype: Literal["categorical", "continuous"] = (
            "categorical" if categorical else "continuous"
        )

        # Default colors
        if color is None:
            color = "tab10" if categorical else "viridis"

        if color not in COLORMAPS:
            raise ValueError(f"Color option not found. Choose from {list(matplotlib.colormaps)}")

        if (
            color not in set(matplotlib.colormaps).difference(set(matplotlib.color_sequences))
            and not categorical
        ):
            raise ValueError(
                f"Continuous layout requires a color scheme from "
                f"{set(matplotlib.colormaps).difference(set(matplotlib.color_sequences))}"
            )

        if categorical and color not in list(matplotlib.color_sequences):
            raise ValueError(
                f"Categorical layout requires a color scheme from "
                f"{list(matplotlib.color_sequences)}"
            )

        if categorical:
            try:
                unique_count = len(set(values))
            except TypeError as exc:
                raise ValueError(
                    "Categorical layout values must be hashable to compute unique categories."
                ) from exc

            cmap_size = matplotlib.colormaps[color].N
            if unique_count > cmap_size:
                raise ValueError(
                    f"Categorical layout '{name}' has {unique_count} unique values but "
                    f"colormap '{color}' only provides {cmap_size} colors."
                )

        if not categorical and _contains_nan(values):
            warnings.warn(
                f"Continuous layout '{name}' contains NaN values. "
                "NaN points will be rendered in black (#000000).",
                UserWarning,
                stacklevel=2,
            )

        if name not in self._layout_keys:
            self._layout_keys.append(name)

        if add_as_label:
            if name not in self._labels_keys:
                self._labels_keys.append(name)
            role: Literal["layout", "layout+label"] = "layout+label"
        else:
            if name in self._labels_keys:
                self._labels_keys.remove(name)
            role = "layout"

        self._columns[name] = Column(name, values, role, _column_dtype, color=color)

    def add_label(
        self,
        name: str,
        values: list[Any],
    ) -> None:
        """Add a text-only label column (shown in tooltip, not used for coloring).

        Args:
            name: Column name displayed in the tooltip header.
            values: One value per point. Non-string values are converted via ``str()``.
        """
        if isinstance(values, np.ndarray):
            values = values.tolist()
        else:
            values = list(values)

        if name not in self._labels_keys:
            self._labels_keys.append(name)
        self._columns[name] = Column(name, values, "label", "label")

    def add_smiles(
        self,
        name: str,
        values: list[str],
    ) -> None:
        """Add a SMILES column for molecular structure visualization.

        When using the SMILES template (smiles.html.j2), molecules will be
        rendered in the tooltip when hovering over points.

        Args:
            name: Column name (displayed in tooltip)
            values: List of SMILES strings, one per point
        """
        if isinstance(values, np.ndarray):
            values = values.tolist()
        else:
            values = list(values)

        if self._smiles_column is not None:
            raise ValueError(
                f"Only one SMILES column is supported. "
                f"Already have '{self._smiles_column}', cannot add '{name}'."
            )

        self._smiles_column = name
        if name not in self._labels_keys:
            self._labels_keys.append(name)
        self._columns[name] = Column(name, values, "smiles", "smiles")

    @property
    def n_points(self) -> int:
        """Return the number of points set."""
        return len(self._points) if self._points else 0

    @property
    def layouts(self) -> list[Column]:
        """Return layouts added."""
        return [self._columns[layout] for layout in self._layout_keys]

    @property
    def labels(self) -> list[Column]:
        """Return labels added."""
        return [self._columns[labels] for labels in self._labels_keys]

    def set_edges(
        self,
        s: list[int] | NDArray[np.unsignedinteger],
        t: list[int] | NDArray[np.unsignedinteger],
    ) -> None:
        """Set MST edge source/target index arrays.

        Args:
            s: Source vertex indices for each edge.
            t: Target vertex indices for each edge.

        Raises:
            ValueError: If arrays differ in length, are not 1-D, or contain
                indices outside ``[0, n_points)``.
        """
        s_arr = np.asarray(s, dtype=np.uint32)
        t_arr = np.asarray(t, dtype=np.uint32)

        if s_arr.ndim != 1 or t_arr.ndim != 1:
            raise ValueError(
                f"Edge arrays must be 1-dimensional. Got s: {s_arr.ndim}D and t: {t_arr.ndim}D"
            )

        if s_arr.shape != t_arr.shape:
            raise ValueError(
                f"Edge arrays must have the same length. Got s: {len(s_arr)} and t: {len(t_arr)}"
            )

        if self.n_points > 0:
            max_idx = self.n_points
            if s_arr.size > 0 and (s_arr.max() >= max_idx or t_arr.max() >= max_idx):
                raise ValueError(
                    f"Edge indices must be < n_points ({max_idx}). "
                    f"Got max(s)={s_arr.max()}, max(t)={t_arr.max()}"
                )

        self._edges_s = s_arr
        self._edges_t = t_arr

    def set_edge_style(
        self,
        color: str | None = None,
        width: float | None = None,
        opacity: float | None = None,
    ) -> None:
        """Set edge rendering style for visualization templates.

        Args:
            color: Hex color string for edges (``#rgb`` or ``#rrggbb``).
            width: Edge line width in CSS pixels. Must be > 0.
            opacity: Edge opacity in ``[0, 1]``.
        """
        if color is not None:
            self.edge_color = _normalize_hex_color(color)

        if width is not None:
            width_value = float(width)
            if not np.isfinite(width_value) or width_value <= 0:
                raise ValueError(f"Edge width must be > 0. Got {width!r}")
            self.edge_width = width_value

        if opacity is not None:
            opacity_value = float(opacity)
            if not np.isfinite(opacity_value) or not 0.0 <= opacity_value <= 1.0:
                raise ValueError(f"Edge opacity must be in [0, 1]. Got {opacity!r}")
            self.edge_opacity = opacity_value

    def set_points(
        self,
        x: list[np.floating] | NDArray[np.floating],
        y: list[np.floating] | NDArray[np.floating],
    ) -> None:
        """
        Store and normalize point coordinates.

        x: X coordinates
        y: Y coordinates
        """
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        # TODO(ISS-015): Profile list→ndarray→list round-trip cost for large inputs

        if x_arr.shape != y_arr.shape:
            raise ValueError("x and y must have the same shape")
        if x_arr.ndim != 1 or y_arr.ndim != 1:
            raise ValueError(
                f"Both x and y should be 1 dimensional arrays,\
                Got x: {x_arr.ndim}D and y: {y_arr.ndim}D"
            )

        normalized_coords = _normalize_coords(x_arr, y_arr)
        self._points_array = normalized_coords  # Keep numpy array for binary mode
        self._points = normalized_coords.tolist()  # Keep list for JSON mode

        for col in self._columns.values():
            if len(col.values) != len(self._points):
                raise ValueError(
                    f"Column '{col.name}' has {len(col.values)} values but there are "
                    f"{len(self._points)} points"
                )

    def to_jupyter(
        self,
        layout: str | None = None,
        *,
        width: int | str = 800,
        height: int = 420,
    ) -> Any:
        """Create a jupyter-scatter view from the current ``TmapViz`` state.

        Args:
            layout: Optional color layout name to visualize. If omitted and at
                least one layout exists, the first layout is used.
            width: Widget width in pixels or ``"auto"``.
            height: Widget height in pixels.

        Returns:
            Configured ``jscatter.Scatter`` instance.
        """
        if self._points_array is None:
            raise ValueError("Call set_points() before converting to a notebook widget.")

        n_points = len(self._points_array)
        for col in self._columns.values():
            if len(col.values) != n_points:
                raise ValueError(
                    f"Column '{col.name}' has {len(col.values)} values but there are "
                    f"{n_points} points"
                )

        if layout is not None and layout not in self._layout_keys:
            raise ValueError(f"Unknown layout '{layout}'. Available layouts: {self._layout_keys}")

        from tmap.visualization.jupyter import to_jscatter

        import pandas as pd  # type: ignore[import-untyped]

        selected_layout = layout
        if selected_layout is None and self._layout_keys:
            selected_layout = self._layout_keys[0]

        data_df: pd.DataFrame | None = None
        if self._columns:
            data_df = pd.DataFrame({name: col.values for name, col in self._columns.items()})

        color_map: str | list[str] | dict[str, str] | None = None
        if selected_layout is not None:
            color_map = self._columns[selected_layout].color

        tooltip_properties = [name for name in self._labels_keys if name in self._columns]

        scatter = to_jscatter(
            self._points_array.astype(np.float32, copy=False),
            color_by=selected_layout,
            color_map=color_map,
            data=data_df,
            tooltip_properties=tooltip_properties or None,
            point_size=self.point_size,
            opacity=self.opacity,
            width=width,
            height=height,
        )

        scatter.background(self.background_color)
        if selected_layout is None:
            scatter.color(default=self.point_color)

        if self._edges_s is not None and self._edges_t is not None and len(self._edges_s) > 0:
            warnings.warn(
                "Edges are not supported in notebook mode yet and will be ignored.",
                UserWarning,
                stacklevel=2,
            )

        return scatter

    def render(self, template_name: str = "base.html.j2") -> str:
        """Return the full HTML string for the visualization.

        Note: Most users should use save() instead, which handles file I/O
        and automatically selects the optimal encoding. This method is useful
        for advanced use cases like serving HTML directly in web applications.

        Args:
            template_name: Name of the Jinja2 template to use.
                           Default is "base.html.j2". Other options include
                           future templates like "smiles.html.j2".

        Returns:
            HTML string ready to be written to a file or served
        """
        if not self._points:
            raise ValueError("Call set_points() before rendering.")

        # Auto-switch to the SMILES template when a SMILES column is present
        # and the caller didn't request a custom template.
        if template_name == "base.html.j2" and self._smiles_column:
            template_name = "smiles.html.j2"

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
                colormaps_payload[colormap_name] = _colormap_to_hex(colormap_name)

        layout_options = list(self._layout_keys)
        label_options = [name for name in self._labels_keys if name in self._columns]
        initial_color = layout_options[0] if layout_options else None

        edges_payload: dict[str, Any] = {}
        if self._edges_s is not None and self._edges_t is not None:
            edges_payload = {
                "s": self._edges_s.tolist(),
                "t": self._edges_t.tolist(),
            }

        payload = {
            "title": self.title,
            "points": self._points,
            "pointColor": self.point_color,
            "pointSize": self.point_size,
            "opacity": self.opacity,
            "edgeStrokeStyle": _hex_to_css_rgba(self.edge_color, self.edge_opacity),
            "edgeWidth": self.edge_width,
            "backgroundColor": _hex_to_rgba(self.background_color),
            "columns": columns_payload,
            "layoutOptions": layout_options,
            "labelOptions": label_options,
            "initialColor": initial_color,
            "colormaps": colormaps_payload,
            "smilesColumn": self._smiles_column,
            "edges": edges_payload,
        }

        runtime = _runtime_base64()
        payload_json = json.dumps(
            _to_json_safe(payload),
            separators=(",", ":"),
            allow_nan=False,
        )

        # Render using Jinja2 template
        env = _get_jinja_env()
        template = env.get_template(template_name)

        return template.render(
            title=self.title,
            background_color=self.background_color,
            payload_json=payload_json,
            runtime_regl=runtime["regl"],
            runtime_pubsub=runtime["pubsub"],
            runtime_scatterplot=runtime["scatterplot"],
        )

    def render_binary(self, template_name: str = "binary.html.j2") -> str:
        """Return HTML string using binary encoding for large datasets.

        Note: Most users should use save() instead, which automatically selects
        binary mode when appropriate. This method is for advanced use cases.

        This method uses:
        - Uint16 quantized coordinates (4x smaller than JSON)
        - Gzip-compressed typed arrays
        - WebWorker decoding (non-blocking)

        Recommended for datasets with >500K points.

        Args:
            template_name: Name of the Jinja2 template to use.

        Returns:
            HTML string with binary-encoded data
        """
        if self._points_array is None:
            raise ValueError("Call set_points() before rendering.")

        # Auto-switch to SMILES-aware binary template when needed.
        if template_name == "binary.html.j2" and self._smiles_column:
            template_name = "smiles_binary.html.j2"

        n_points = len(self._points_array)
        for col in self._columns.values():
            if len(col.values) != n_points:
                raise ValueError(
                    f"Column '{col.name}' has {len(col.values)} values but there are "
                    f"{n_points} points"
                )

        # Pack coordinates as binary
        coords_compressed = _pack_coords_binary(self._points_array, bits=16)
        coords_b64 = base64.b64encode(coords_compressed).decode("ascii")

        # Pack columns
        columns_b64: dict[str, str] = {}
        columns_meta: dict[str, dict[str, Any]] = {}
        colormaps_payload: dict[str, list[str]] = {}

        for name, col in self._columns.items():
            colormap_name = col.color if col.role in ("layout", "layout+label") else None

            if col.dtype == "categorical":
                compressed, dictionary = _pack_categorical_binary(col.values)
                columns_b64[name] = base64.b64encode(compressed).decode("ascii")
                columns_meta[name] = {
                    "dtype": "uint32",
                    "role": col.role,
                    "colormap": colormap_name,
                    "dictionary": dictionary,
                }
            elif col.dtype == "continuous":
                arr = np.array(col.values, dtype=np.float32)
                compressed = _pack_numeric_binary(arr, "float32")
                columns_b64[name] = base64.b64encode(compressed).decode("ascii")
                columns_meta[name] = {
                    "dtype": "float32",
                    "role": col.role,
                    "colormap": colormap_name,
                }
            else:
                # Labels and SMILES - keep as strings for now (will be optimized later)
                # For now, we skip binary encoding for string columns
                # They'll be included in metadata
                columns_meta[name] = {
                    "dtype": "string",
                    "role": col.role,
                    "values": col.values,  # Include directly in metadata for now
                }

            if colormap_name and colormap_name not in colormaps_payload:
                colormaps_payload[colormap_name] = _colormap_to_hex(colormap_name)

        # Pack edges if present
        edges_b64 = ""
        n_edges = 0
        if self._edges_s is not None and self._edges_t is not None:
            n_edges = len(self._edges_s)
            edges_combined = np.concatenate([self._edges_s, self._edges_t]).astype(np.uint32)
            edges_compressed = gzip.compress(edges_combined.tobytes(), compresslevel=6)
            edges_b64 = base64.b64encode(edges_compressed).decode("ascii")

        # Build header/metadata
        layout_options = list(self._layout_keys)
        label_options = [name for name in self._labels_keys if name in self._columns]

        header = {
            "version": 1,
            "nPoints": n_points,
            "coordDtype": "uint16",
            "columns": columns_meta,
            "metadata": {
                "title": self.title,
                "pointColor": self.point_color,
                "pointSize": self.point_size,
                "opacity": self.opacity,
                "edgeStrokeStyle": _hex_to_css_rgba(self.edge_color, self.edge_opacity),
                "edgeWidth": self.edge_width,
                "backgroundColor": _hex_to_rgba(self.background_color),
                "layoutOptions": layout_options,
                "labelOptions": label_options,
                "colormaps": colormaps_payload,
                "smilesColumn": self._smiles_column,
                "nEdges": n_edges,
            },
        }

        runtime = _runtime_base64()
        header_json = json.dumps(
            _to_json_safe(header),
            separators=(",", ":"),
            allow_nan=False,
        )

        # Filter out string columns from binary data (they're in header)
        columns_b64_filtered = {
            k: v for k, v in columns_b64.items() if columns_meta[k]["dtype"] != "string"
        }

        env = _get_jinja_env()
        template = env.get_template(template_name)

        return template.render(
            title=self.title,
            background_color=self.background_color,
            header_json=header_json,
            coords_b64=coords_b64,
            columns_b64=columns_b64_filtered,
            edges_b64=edges_b64,
            runtime_regl=runtime["regl"],
            runtime_pubsub=runtime["pubsub"],
            runtime_scatterplot=runtime["scatterplot"],
        )

    def save(
        self,
        path: str | Path,
        binary_threshold: int = BINARY_THRESHOLD,
        force_binary: bool = False,
    ) -> Path:
        """Write HTML to disk and return the path.

        This is the primary method for saving visualizations. It automatically
        selects binary mode for large datasets (>500k points by default).

        Args:
            path: Either a full file path (ending in .html) or a directory path.
                  - If a file path: saves to that exact location
                  - If a directory: uses self.title as the filename
            binary_threshold: Point count above which binary mode is used.
                              Default is 500,000 points.
            force_binary: If True, always use binary mode regardless of size.

        Returns:
            Path to the saved file

        Examples:
            >>> viz.save("output.html")  # Saves to output.html
            >>> viz.save("results/")     # Saves to results/{title}.html
            >>> viz.save("results/viz.html")  # Saves to results/viz.html
        """
        path = Path(path)

        # Determine if path is a file or directory
        if str(path).endswith(".html"):
            # Full file path provided
            output_path = path
        elif path.is_dir() or (not path.exists() and not str(path).endswith(".html")):
            # Directory provided (existing or will be created) - use title as filename
            if not self.title.endswith(".html"):
                filename = self.title + ".html"
            else:
                filename = self.title
            output_path = path / filename
        else:
            # Assume it's a file path without .html extension
            output_path = Path(str(path) + ".html")

        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Auto-select rendering mode based on dataset size
        n_points = len(self._points) if self._points else 0
        use_binary = force_binary or n_points > binary_threshold

        if use_binary:
            html = self.render_binary()
        else:
            html = self.render()

        output_path.write_text(html, encoding="utf-8")
        return output_path
