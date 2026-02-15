"""Tests for jupyter-scatter integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    from jscatter.jscatter import Scatter as _Scatter  # noqa: F401

    _JSCATTER_AVAILABLE = True
except ImportError:
    _JSCATTER_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _JSCATTER_AVAILABLE, reason="jscatter not installed")


@pytest.fixture
def embedding() -> np.ndarray:
    """A small 2D embedding for testing."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((50, 2)).astype(np.float32)


# ---------- to_jscatter tests ----------


class TestToJscatter:
    def test_minimal(self, embedding: np.ndarray) -> None:
        from tmap.visualization.jupyter import to_jscatter

        scatter = to_jscatter(embedding)
        assert isinstance(scatter, _Scatter)
        opacity_cfg = scatter.opacity()
        assert opacity_cfg["by"] is None
        assert opacity_cfg["map"] is None

    def test_float_color_continuous(self, embedding: np.ndarray) -> None:
        """Float array → continuous → viridis."""
        from tmap.visualization.jupyter import to_jscatter

        values = np.linspace(0, 1, len(embedding))
        scatter = to_jscatter(embedding, color_by=values)
        assert isinstance(scatter, _Scatter)

    def test_string_color_categorical(self, embedding: np.ndarray) -> None:
        """String array → categorical → tab10."""
        from tmap.visualization.jupyter import to_jscatter

        labels = np.array(["a", "b", "c"] * 16 + ["a", "b"])
        scatter = to_jscatter(embedding, color_by=labels)
        assert isinstance(scatter, _Scatter)

    def test_int_color_array_is_categorical_scale(self, embedding: np.ndarray) -> None:
        """Integer category IDs should render with categorical color scale."""
        from tmap.visualization.jupyter import to_jscatter

        labels = np.array([i % 10 for i in range(len(embedding))], dtype=np.int32)
        scatter = to_jscatter(embedding, color_by=labels, color_map="tab10")
        assert isinstance(scatter, _Scatter)
        assert scatter.widget.color_scale == "categorical"

    def test_int_color_few_unique_categorical(self, embedding: np.ndarray) -> None:
        """Int array with ≤30 unique → categorical."""
        from tmap.visualization.jupyter import _is_categorical

        arr = np.array([0, 1, 2, 3] * 12 + [0, 1])
        assert _is_categorical(arr) is True

    def test_float_color_continuous_detection(self) -> None:
        """Float array → not categorical."""
        from tmap.visualization.jupyter import _is_categorical

        arr = np.linspace(0, 1, 100)
        assert _is_categorical(arr) is False

    def test_int_many_unique_continuous(self) -> None:
        """Int array with >30 unique → continuous."""
        from tmap.visualization.jupyter import _is_categorical

        arr = np.arange(50)
        assert _is_categorical(arr) is False

    def test_dataframe_column_color(self, embedding: np.ndarray) -> None:
        """DataFrame + column name."""
        from tmap.visualization.jupyter import to_jscatter

        df = pd.DataFrame({"species": ["cat", "dog"] * 25})
        scatter = to_jscatter(embedding, data=df, color_by="species")
        assert isinstance(scatter, _Scatter)

    def test_dataframe_int_column_color_is_categorical_scale(self, embedding: np.ndarray) -> None:
        """Integer category columns should render as categorical colors."""
        from tmap.visualization.jupyter import to_jscatter

        df = pd.DataFrame({"cluster": [i % 10 for i in range(len(embedding))]})
        scatter = to_jscatter(embedding, data=df, color_by="cluster", color_map="tab10")
        assert isinstance(scatter, _Scatter)
        assert scatter.widget.color_scale == "categorical"

    def test_dataframe_category_dtype(self, embedding: np.ndarray) -> None:
        """Category dtype column detected as categorical."""
        from tmap.visualization.jupyter import _is_categorical

        df = pd.DataFrame({"x": pd.Categorical(["a", "b", "c"] * 10)})
        assert _is_categorical(None, data=df, col_name="x") is True

    def test_explicit_color_map(self, embedding: np.ndarray) -> None:
        """Explicit color_map overrides default."""
        from tmap.visualization.jupyter import to_jscatter

        values = np.linspace(0, 1, len(embedding))
        scatter = to_jscatter(embedding, color_by=values, color_map="plasma")
        assert isinstance(scatter, _Scatter)

    def test_tooltip_properties(self, embedding: np.ndarray) -> None:
        """tooltip_properties passed through."""
        from tmap.visualization.jupyter import to_jscatter

        df = pd.DataFrame(
            {
                "name": [f"pt_{i}" for i in range(len(embedding))],
                "value": np.arange(len(embedding), dtype=float),
            }
        )
        scatter = to_jscatter(embedding, data=df, tooltip_properties=["name", "value"])
        assert isinstance(scatter, _Scatter)

    def test_width_and_height(self, embedding: np.ndarray) -> None:
        """Custom width/height propagate to the Scatter object."""
        from tmap.visualization.jupyter import to_jscatter

        scatter = to_jscatter(embedding, width=640, height=360)
        assert scatter.width() == 640
        assert scatter.height() == 360

    def test_invalid_embedding_shape(self) -> None:
        """Embedding must have shape (n_samples, 2)."""
        from tmap.visualization.jupyter import to_jscatter

        with pytest.raises(ValueError, match="shape"):
            to_jscatter(np.ones((10, 3), dtype=np.float32))

    def test_non_finite_embedding(self, embedding: np.ndarray) -> None:
        """Embedding with NaN/inf is rejected."""
        from tmap.visualization.jupyter import to_jscatter

        bad = embedding.copy()
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            to_jscatter(bad)

    def test_mismatched_data_length(self, embedding: np.ndarray) -> None:
        """data with wrong row count raises ValueError."""
        from tmap.visualization.jupyter import to_jscatter

        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="rows"):
            to_jscatter(embedding, data=df)

    def test_mismatched_color_length(self, embedding: np.ndarray) -> None:
        """color_by array with wrong length raises ValueError."""
        from tmap.visualization.jupyter import to_jscatter

        with pytest.raises(ValueError, match="elements"):
            to_jscatter(embedding, color_by=[1, 2, 3])

    def test_invalid_column_name(self, embedding: np.ndarray) -> None:
        """Nonexistent column name raises ValueError."""
        from tmap.visualization.jupyter import to_jscatter

        with pytest.raises(ValueError, match="not a column"):
            to_jscatter(embedding, color_by="nonexistent")

    def test_to_jscatter_not_public_on_visualization_module(self) -> None:
        """to_jscatter should not be exposed from tmap.visualization public API."""
        import tmap.visualization as viz_mod

        with pytest.raises(AttributeError):
            _ = viz_mod.to_jscatter


# ---------- TMAP.plot() tests ----------


class TestTMAPPlot:
    def test_plot_unfitted_raises(self) -> None:
        """plot() on unfitted model raises RuntimeError."""
        from tmap.estimator import TMAP

        model = TMAP()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.plot()

    def test_plot_returns_scatter(self) -> None:
        """plot() on a fitted model returns _Scatter."""
        from tmap.estimator import TMAP

        model = TMAP()
        # Inject a fake embedding to avoid needing OGDF
        model._embedding = np.random.default_rng(0).standard_normal((20, 2)).astype(np.float32)
        scatter = model.plot(show=False)
        assert isinstance(scatter, _Scatter)

    def test_plot_with_color(self) -> None:
        """plot(color_by=...) works."""
        from tmap.estimator import TMAP

        model = TMAP()
        model._embedding = np.random.default_rng(0).standard_normal((20, 2)).astype(np.float32)
        labels = ["a", "b"] * 10
        scatter = model.plot(color_by=labels, show=False)
        assert isinstance(scatter, _Scatter)

    def test_plot_show_calls_display_helper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """plot(show=True) delegates display to jupyter helper."""
        from tmap.estimator import TMAP

        calls: list[bool] = []

        def _fake_display(scatter: _Scatter, *, controls: bool = False) -> None:
            calls.append(controls)

        monkeypatch.setattr("tmap.visualization.jupyter._display_scatter", _fake_display)

        model = TMAP()
        model._embedding = np.random.default_rng(0).standard_normal((20, 2)).astype(np.float32)

        scatter = model.plot(show=True, controls=True)
        assert isinstance(scatter, _Scatter)
        assert calls == [True]

    def test_plot_show_false_skips_display_helper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """plot(show=False) does not trigger display helper."""
        from tmap.estimator import TMAP

        def _fake_display(*args: object, **kwargs: object) -> None:
            raise AssertionError("display helper should not be called")

        monkeypatch.setattr("tmap.visualization.jupyter._display_scatter", _fake_display)

        model = TMAP()
        model._embedding = np.random.default_rng(0).standard_normal((20, 2)).astype(np.float32)

        scatter = model.plot(show=False)
        assert isinstance(scatter, _Scatter)


class TestDisplayHelper:
    def test_display_scatter_prefers_show_when_flag_is_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When requested, helper should render through show(buttons=[])."""
        from tmap.visualization.jupyter import _display_scatter

        class _DummyScatter:
            def __init__(self) -> None:
                self.widget = object()
                self._tmap_prefers_show = True
                self.show_calls: list[object] = []

            def show(self, buttons=None):
                self.show_calls.append(buttons)
                return "shown-widget"

        dummy = _DummyScatter()
        displayed: list[object] = []

        monkeypatch.setattr("IPython.get_ipython", lambda: object())
        monkeypatch.setattr("IPython.display.display", lambda widget: displayed.append(widget))

        _display_scatter(dummy, controls=False)

        assert dummy.show_calls == [[]]
        assert displayed == ["shown-widget"]
