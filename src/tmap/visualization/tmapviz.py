from __future__ import annotations

import base64
import gzip
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Literal, Optional, Sequence

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
    """TODO: remove this once develop is done
    This is the node_modules that `npm install regl-scatterplot`
    creates. Not really something to commit, now the vendors take care 
    of that but in weird cases a fallback is okay
    """
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
def _get_jinja_env() -> "Environment":
    """Get or create a cached Jinja2 environment for templates."""
    if not _JINJA_AVAILABLE:
        raise ImportError(
            "Jinja2 is required for template rendering. "
            "Install with: pip install tmap[viz]"
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
    return np.stack([x_norm, y_norm], axis=1).astype(np.float64)


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> list[float]:
    """Convert #RRGGBB to [r, g, b, a] floats in [0, 1]."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color!r}")
    return [int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4)] + [alpha]


# TODO: Implement categorical=True preserves listed colors when available
def _colormap_to_hex(name: str) -> list[str]:
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
    values: Sequence[int | np.floating | str]
    role: Literal["layout", "label", "layout+label", "smiles"]
    dtype: Literal["continuous", "categorical", "label", "smiles"]
    color: Optional[str] = None


def _pack_coords_binary(points: np.ndarray, bits: int = 16) -> bytes:
    """Pack normalized [-1,1] coordinates as gzip-compressed quantized integers."""
    if bits == 16:
        max_val = 65535
        dtype = np.uint16
    else:
        max_val = 4294967295
        dtype = np.uint32

    # Quantize: [-1, 1] -> [0, max_val]
    quantized = ((points.astype(np.float64) + 1.0) * (max_val / 2.0)).astype(dtype)
    raw = quantized.flatten().tobytes()
    return gzip.compress(raw, compresslevel=6)


def _pack_numeric_binary(values: np.ndarray, dtype: str = "float32") -> bytes:
    """Pack numeric column as gzip-compressed typed array."""
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
    indices = np.empty(len(values), dtype=np.uint32)

    for i, v in enumerate(values):
        s = str(v)
        if s not in value_to_idx:
            value_to_idx[s] = len(unique_values)
            unique_values.append(s)
        indices[i] = value_to_idx[s]

    compressed = gzip.compress(indices.tobytes(), compresslevel=6)
    return compressed, unique_values


class TmapViz:
    def __init__(self) -> None:
        self.title: str = "MyTMAP"
        self.background_color: str = "#7A7A7A"
        self.point_color: str = "#4a9eff"
        self.point_size: float = 4.0
        self.opacity: float = 0.85

        # Store both formats for flexibility
        self._points: List[list[float]] = []
        self._points_array: Optional[np.ndarray] = None  # Shape: (n, 2)
        self._layout_keys: List[str] = []
        self._labels_keys: List[str] = []
        self._smiles_column: Optional[str] = None
        self._columns: dict[str, Column] = {}

    def add_color_layout(
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

        # Default to continuous because it will give less issues and having to pass
        # always the type can be annoying...
        _column_dtype = "categorical" if categorical else "continuous"

        # Default colors
        if color is None:
            color = "tab10" if categorical else "viridis"

        if color not in COLORMAPS:
            raise ValueError(
                f"Color option not found. Choose from {list(matplotlib.colormaps)}"
            )

        if color not in set(matplotlib.colormaps).difference(
            set(matplotlib.color_sequences)
        ) and not categorical:
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
        if name not in self._layout_keys:
            self._layout_keys.append(name)

        if add_as_label:
            if name not in self._labels_keys:
                self._labels_keys.append(name)
            role = "layout+label"
        else:
            if name in self._labels_keys:
                self._labels_keys.remove(name)
            role = "layout"

        self._columns[name] = Column(name, values, role, _column_dtype, color=color)

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

    def add_smiles(
        self,
        name: str,
        values: List[str],
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
        self._points_array = normalized_coords  # Keep numpy array for binary mode
        self._points = normalized_coords.tolist()  # Keep list for JSON mode

        for col in self._columns.values():
            if len(col.values) != len(self._points):
                raise ValueError(
                    f"Column '{col.name}' has {len(col.values)} values but there are "
                    f"{len(self._points)} points"
                )

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
            "smilesColumn": self._smiles_column,
        }

        runtime = _runtime_base64()
        payload_json = json.dumps(payload, separators=(",", ":"))

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
                "backgroundColor": _hex_to_rgba(self.background_color),
                "layoutOptions": layout_options,
                "labelOptions": label_options,
                "colormaps": colormaps_payload,
                "smilesColumn": self._smiles_column,
            },
        }

        runtime = _runtime_base64()
        header_json = json.dumps(header, separators=(",", ":"))

        # Filter out string columns from binary data (they're in header)
        columns_b64_filtered = {
            k: v for k, v in columns_b64.items()
            if columns_meta[k]["dtype"] != "string"
        }

        env = _get_jinja_env()
        template = env.get_template(template_name)

        return template.render(
            title=self.title,
            background_color=self.background_color,
            header_json=header_json,
            coords_b64=coords_b64,
            columns_b64=columns_b64_filtered,
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
        if str(path).endswith('.html'):
            # Full file path provided
            output_path = path
        elif path.is_dir() or (not path.exists() and not str(path).endswith('.html')):
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
