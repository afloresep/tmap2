"""jupyter-scatter integration for interactive TMAP visualization in notebooks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd
    from jscatter.jscatter import Scatter


def _check_jscatter() -> None:
    """Raise a clear error if jupyter-scatter is not installed."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        version("jupyter-scatter")
    except PackageNotFoundError:
        raise ImportError(
            "jupyter-scatter is required for interactive notebook visualization.\n"
            "Install full dependencies with: pip install -e ."
        ) from None

    try:
        from jscatter.jscatter import Scatter as _Scatter  # noqa: F401
    except ImportError:
        raise ImportError(
            "Could not import jscatter widget bindings from jupyter-scatter.\n"
            "Reinstall with: pip install --upgrade --force-reinstall jupyter-scatter"
        ) from None


def _validate_embedding(embedding: Any) -> NDArray[np.float32]:
    """Validate and normalize embedding input."""
    arr = np.asarray(embedding, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"embedding must have shape (n_samples, 2); got {arr.shape!r}"
        )
    if arr.shape[0] == 0:
        raise ValueError("embedding must contain at least one point")
    if not np.isfinite(arr).all():
        raise ValueError("embedding contains non-finite values (NaN or inf)")
    return arr


def _is_categorical(
    values: Any,
    data: pd.DataFrame | None = None,
    col_name: str | None = None,
) -> bool:
    """Infer whether *values* should be treated as categorical."""
    import pandas as pd

    if data is not None and col_name is not None and col_name in data.columns:
        series = data[col_name]
        dtype = series.dtype
        if (
            isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
        ):
            return True
        arr = np.asarray(series)
    else:
        arr = np.asarray(values)

    if arr.dtype.kind in ("U", "S", "O"):
        return True
    if np.issubdtype(arr.dtype, np.integer) and len(np.unique(arr)) <= 30:
        return True
    return False


def _display_scatter(scatter: Scatter, *, controls: bool = False) -> None:
    """Display a scatter widget when running inside IPython/Jupyter."""
    try:
        from IPython import get_ipython
        from IPython.display import display
    except ImportError:
        return

    if get_ipython() is None:
        return

    widget = scatter.show() if controls else scatter.widget
    display(widget)


def to_jscatter(
    embedding: NDArray[np.float32] | Sequence[Sequence[float]],
    *,
    color_by: str | NDArray | Sequence | None = None,
    color_map: str | list | dict | None = None,
    data: pd.DataFrame | None = None,
    tooltip_properties: list[str] | None = None,
    point_size: float = 3,
    opacity: float = 0.8,
    width: int | str = 800,
    height: int = 420,
) -> Scatter:
    """Create an interactive ``jscatter.Scatter`` widget from an embedding."""
    _check_jscatter()
    from jscatter.jscatter import Scatter as _Scatter

    import pandas as pd

    coords = _validate_embedding(embedding)
    n = coords.shape[0]

    if isinstance(width, str) and width != "auto":
        raise ValueError("width must be an integer or 'auto'")
    if isinstance(width, int) and width <= 0:
        raise ValueError("width must be > 0")
    if height <= 0:
        raise ValueError("height must be > 0")

    # -- Build DataFrame -----------------------------------------------------
    if data is not None:
        df = data.copy()
        if len(df) != n:
            raise ValueError(f"data has {len(df)} rows but embedding has {n} points")
    else:
        df = pd.DataFrame()

    df["_tmap_x"] = coords[:, 0]
    df["_tmap_y"] = coords[:, 1]

    # -- Resolve color source ------------------------------------------------
    color_col: str | None = None
    categorical = False

    if color_by is not None:
        if isinstance(color_by, str):
            if data is None or color_by not in data.columns:
                raise ValueError(f"color_by={color_by!r} is not a column in data")
            color_col = color_by
            categorical = _is_categorical(None, data=df, col_name=color_col)
            if categorical:
                # jscatter treats numeric dtypes as continuous unless they're
                # explicitly categorical.
                df[color_col] = pd.Series(df[color_col], index=df.index).astype("category")
        else:
            arr = np.asarray(color_by)
            if len(arr) != n:
                raise ValueError(f"color_by has {len(arr)} elements but embedding has {n} points")
            categorical = _is_categorical(arr)
            if categorical:
                # Enforce categorical dtype for integer category IDs.
                df["_tmap_color"] = pd.Series(arr, index=df.index).astype("category")
            else:
                df["_tmap_color"] = arr
            color_col = "_tmap_color"

    # -- Construct Scatter ---------------------------------------------------
    # Force static opacity at construction time. jscatter defaults to
    # opacity-by-density; overriding opacity later can leave an internal map
    # around and make points effectively transparent.
    scatter = _Scatter(
        x="_tmap_x",
        y="_tmap_y",
        data=df,
        width=width,
        height=height,
        size=point_size,
        opacity=opacity,
        opacity_by=None,
        opacity_map=None,
    )

    # -- Apply visual encodings ----------------------------------------------
    if color_col is not None:
        cmap = color_map if color_map is not None else ("tab10" if categorical else "viridis")
        scatter.color(by=color_col, map=cmap)

    if tooltip_properties is not None:
        scatter.tooltip(enable=True, properties=tooltip_properties)

    if color_col is not None:
        scatter.legend(True)

    return scatter
