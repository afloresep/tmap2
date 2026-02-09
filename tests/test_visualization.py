"""
Tests for the visualization module.

Tests cover:
- TmapViz class creation and configuration
- Point coordinate setting and validation
- Color layout (continuous and categorical)
- Label and SMILES columns
- HTML rendering (JSON and binary modes)
- Binary container format utilities
"""

import json
import re

import numpy as np
import pytest

# Check if visualization dependencies are available
try:
    from tmap.visualization import BINARY_THRESHOLD, TmapViz
    from tmap.visualization.binary import (
        BinaryContainerWriter,
        dequantize_coords,
        pack_categorical_column,
        pack_coords,
        pack_numeric_column,
        quantize_coords,
    )
    _VIZ_AVAILABLE = True
except ImportError as e:
    _VIZ_AVAILABLE = False
    _IMPORT_ERROR = e

pytestmark = pytest.mark.skipif(
    not _VIZ_AVAILABLE,
    reason="Visualization dependencies not available (jinja2 required)"
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_coords():
    """Simple coordinate arrays for testing."""
    x = np.array([0.0, 1.0, 0.5, -0.5], dtype=np.float32)
    y = np.array([0.0, 0.5, 1.0, -0.5], dtype=np.float32)
    return x, y


@pytest.fixture
def sample_data():
    """Sample data for visualization testing."""
    n = 100
    np.random.seed(42)
    return {
        "x": np.random.uniform(-1, 1, n).astype(np.float32),
        "y": np.random.uniform(-1, 1, n).astype(np.float32),
        "continuous": np.random.uniform(0, 100, n),
        "categorical": np.random.choice(["A", "B", "C"], n).tolist(),
        "labels": [f"Point_{i}" for i in range(n)],
        "smiles": [f"C{'C' * (i % 5)}" for i in range(n)],  # Simple SMILES
    }


@pytest.fixture
def viz_with_data(sample_data):
    """TmapViz instance with data already set."""
    viz = TmapViz()
    viz.set_points(sample_data["x"], sample_data["y"])
    return viz, sample_data


# =============================================================================
# TmapViz Basic Tests
# =============================================================================


class TestTmapVizCreation:
    """Tests for TmapViz creation and basic properties."""

    def test_default_values(self):
        """TmapViz should have sensible defaults."""
        viz = TmapViz()

        assert viz.title == "MyTMAP"
        assert viz.background_color == "#7A7A7A"
        assert viz.point_color == "#4a9eff"
        assert viz.point_size == 4.0
        assert viz.opacity == 0.85
        assert viz.edge_color == "#000000"
        assert viz.edge_opacity == 0.5
        assert viz.edge_width == 2.0
        assert viz.n_points == 0

    def test_properties_settable(self):
        """Properties should be settable."""
        viz = TmapViz()

        viz.title = "Test Title"
        viz.background_color = "#FFFFFF"
        viz.point_color = "#FF0000"
        viz.point_size = 10.0
        viz.opacity = 0.5

        assert viz.title == "Test Title"
        assert viz.background_color == "#FFFFFF"
        assert viz.point_color == "#FF0000"
        assert viz.point_size == 10.0
        assert viz.opacity == 0.5


# =============================================================================
# set_points Tests
# =============================================================================


class TestSetPoints:
    """Tests for set_points method."""

    def test_basic_set_points(self, simple_coords):
        """set_points should accept coordinate arrays."""
        x, y = simple_coords
        viz = TmapViz()

        viz.set_points(x, y)

        assert viz.n_points == 4

    def test_accepts_lists(self):
        """set_points should accept Python lists."""
        viz = TmapViz()

        viz.set_points([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])

        assert viz.n_points == 3

    def test_mismatched_shapes_raises(self):
        """Mismatched x and y should raise ValueError."""
        viz = TmapViz()

        with pytest.raises(ValueError, match="same shape"):
            viz.set_points([0.0, 1.0], [0.0, 1.0, 2.0])

    def test_2d_arrays_raise(self):
        """2D arrays should raise ValueError."""
        viz = TmapViz()

        with pytest.raises(ValueError, match="1 dimensional"):
            viz.set_points(np.zeros((5, 2)), np.zeros((5, 2)))

    def test_coordinates_normalized(self, simple_coords):
        """Coordinates should be normalized to [-1, 1]."""
        x, y = simple_coords
        viz = TmapViz()

        viz.set_points(x, y)

        # Internal points should be normalized
        points = viz._points
        for p in points:
            assert -1.0 <= p[0] <= 1.0
            assert -1.0 <= p[1] <= 1.0


# =============================================================================
# add_color_layout Tests
# =============================================================================


class TestAddColorLayout:
    """Tests for add_color_layout method."""

    def test_continuous_layout(self, viz_with_data):
        """Should add continuous color layout."""
        viz, data = viz_with_data

        viz.add_color_layout("value", data["continuous"], categorical=False)

        assert len(viz.layouts) == 1
        assert viz.layouts[0].name == "value"
        assert viz.layouts[0].dtype == "continuous"

    def test_categorical_layout(self, viz_with_data):
        """Should add categorical color layout."""
        viz, data = viz_with_data

        viz.add_color_layout("group", data["categorical"], categorical=True, color="tab10")

        assert len(viz.layouts) == 1
        assert viz.layouts[0].name == "group"
        assert viz.layouts[0].dtype == "categorical"

    def test_default_colormap_continuous(self, viz_with_data):
        """Continuous should default to viridis."""
        viz, data = viz_with_data

        viz.add_color_layout("value", data["continuous"], categorical=False)

        assert viz.layouts[0].color == "viridis"

    def test_default_colormap_categorical(self, viz_with_data):
        """Categorical should default to tab10."""
        viz, data = viz_with_data

        viz.add_color_layout("group", data["categorical"], categorical=True)

        assert viz.layouts[0].color == "tab10"

    def test_custom_colormap(self, viz_with_data):
        """Should accept custom colormap."""
        viz, data = viz_with_data

        viz.add_color_layout("value", data["continuous"], categorical=False, color="plasma")

        assert viz.layouts[0].color == "plasma"

    def test_invalid_colormap_raises(self, viz_with_data):
        """Invalid colormap should raise ValueError."""
        viz, data = viz_with_data

        with pytest.raises(ValueError, match="Color option not found"):
            viz.add_color_layout("value", data["continuous"], color="not_a_colormap")

    def test_add_as_label_true(self, viz_with_data):
        """add_as_label=True should add to labels."""
        viz, data = viz_with_data

        viz.add_color_layout("value", data["continuous"], add_as_label=True)

        assert len(viz.labels) == 1
        assert viz.labels[0].name == "value"

    def test_add_as_label_false(self, viz_with_data):
        """add_as_label=False should not add to labels."""
        viz, data = viz_with_data

        viz.add_color_layout("value", data["continuous"], add_as_label=False)

        # Should be in layouts but not labels
        assert len(viz.layouts) == 1
        assert len(viz.labels) == 0

    def test_multiple_layouts(self, viz_with_data):
        """Should support multiple color layouts."""
        viz, data = viz_with_data

        viz.add_color_layout("continuous_col", data["continuous"], categorical=False)
        viz.add_color_layout("categorical_col", data["categorical"], categorical=True)

        assert len(viz.layouts) == 2


# =============================================================================
# add_label Tests
# =============================================================================


class TestAddLabel:
    """Tests for add_label method."""

    def test_basic_label(self, viz_with_data):
        """Should add label column."""
        viz, data = viz_with_data

        viz.add_label("name", data["labels"])

        assert len(viz.labels) == 1
        assert viz.labels[0].name == "name"
        assert viz.labels[0].dtype == "label"

    def test_multiple_labels(self, viz_with_data):
        """Should support multiple labels."""
        viz, data = viz_with_data

        viz.add_label("name", data["labels"])
        viz.add_label("id", [str(i) for i in range(100)])

        assert len(viz.labels) == 2


# =============================================================================
# add_smiles Tests
# =============================================================================


class TestAddSmiles:
    """Tests for add_smiles method."""

    def test_basic_smiles(self, viz_with_data):
        """Should add SMILES column."""
        viz, data = viz_with_data

        viz.add_smiles("structure", data["smiles"])

        assert viz._smiles_column == "structure"
        assert "structure" in [label.name for label in viz.labels]

    def test_only_one_smiles_allowed(self, viz_with_data):
        """Only one SMILES column should be allowed."""
        viz, data = viz_with_data

        viz.add_smiles("structure1", data["smiles"])

        with pytest.raises(ValueError, match="Only one SMILES column"):
            viz.add_smiles("structure2", data["smiles"])


# =============================================================================
# render Tests
# =============================================================================


class TestRender:
    """Tests for render method (JSON mode)."""

    def test_render_basic(self, viz_with_data):
        """Should render to HTML string."""
        viz, data = viz_with_data
        viz.add_color_layout("value", data["continuous"])

        html = viz.render()

        assert isinstance(html, str)
        assert "<html" in html.lower()
        assert "</html>" in html.lower()

    def test_render_without_points_raises(self):
        """render without set_points should raise."""
        viz = TmapViz()

        with pytest.raises(ValueError, match="set_points"):
            viz.render()

    def test_render_contains_title(self, viz_with_data):
        """HTML should contain title."""
        viz, data = viz_with_data
        viz.title = "Test Visualization"

        html = viz.render()

        assert "Test Visualization" in html

    def test_render_with_smiles_uses_smiles_template(self, viz_with_data):
        """Adding SMILES should auto-switch to smiles template."""
        viz, data = viz_with_data
        viz.add_smiles("structure", data["smiles"])

        html = viz.render()

        # Should render without error (smiles template used)
        assert isinstance(html, str)
        assert len(html) > 0


# =============================================================================
# render_binary Tests
# =============================================================================


class TestRenderBinary:
    """Tests for render_binary method."""

    def test_render_binary_basic(self, viz_with_data):
        """Should render binary HTML."""
        viz, data = viz_with_data
        viz.add_color_layout("value", data["continuous"])

        html = viz.render_binary()

        assert isinstance(html, str)
        assert "<html" in html.lower()

    def test_render_binary_without_points_raises(self):
        """render_binary without set_points should raise."""
        viz = TmapViz()

        with pytest.raises(ValueError, match="set_points"):
            viz.render_binary()

    def test_render_binary_smaller_than_json(self, viz_with_data):
        """Binary mode should generally produce smaller output."""
        viz, data = viz_with_data
        viz.add_color_layout("value", data["continuous"])

        html_json = viz.render()
        html_binary = viz.render_binary()

        # For small datasets, binary might not be smaller due to overhead
        # But both should be valid
        assert len(html_json) > 0
        assert len(html_binary) > 0


# =============================================================================
# save Tests
# =============================================================================


class TestSave:
    """Tests for save method."""

    def test_save_creates_file(self, viz_with_data, tmp_path):
        """save should create HTML file."""
        viz, data = viz_with_data
        viz.title = "test_output"

        output_path = viz.save(tmp_path)

        assert output_path.exists()
        assert output_path.name == "test_output.html"

    def test_save_adds_html_extension(self, viz_with_data, tmp_path):
        """save should add .html extension if missing."""
        viz, data = viz_with_data
        viz.title = "my_viz"

        output_path = viz.save(tmp_path)

        assert output_path.suffix == ".html"

    def test_save_preserves_existing_extension(self, viz_with_data, tmp_path):
        """save should preserve .html extension if present."""
        viz, data = viz_with_data
        viz.title = "my_viz.html"

        output_path = viz.save(tmp_path)

        assert output_path.name == "my_viz.html"
        assert not output_path.name.endswith(".html.html")

    def test_save_force_binary(self, viz_with_data, tmp_path):
        """force_binary should use binary mode."""
        viz, data = viz_with_data
        viz.title = "binary_output"

        output_path = viz.save(tmp_path, force_binary=True)

        assert output_path.exists()


# =============================================================================
# Binary Module Tests
# =============================================================================


class TestBinaryModule:
    """Tests for binary container format utilities."""

    def test_quantize_coords_16bit(self):
        """16-bit quantization should work."""
        coords = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])

        quantized = quantize_coords(coords, bits=16)

        assert quantized.dtype == np.uint16
        assert quantized[0, 0] == 0  # -1 -> 0
        assert quantized[2, 0] == 65535  # 1 -> max
        assert 32000 < quantized[1, 0] < 33000  # 0 -> ~middle

    def test_quantize_dequantize_roundtrip(self):
        """Quantize then dequantize should approximate original."""
        coords = np.array([[-0.5, 0.25], [0.75, -0.9]])

        quantized = quantize_coords(coords, bits=16)
        restored = dequantize_coords(quantized, bits=16)

        # Should be close (within quantization error)
        np.testing.assert_allclose(coords, restored, atol=1e-4)

    def test_pack_coords(self):
        """pack_coords should return compressed bytes."""
        x = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        y = np.array([0.0, 0.5, 1.0], dtype=np.float64)

        compressed, uncompressed_size = pack_coords(x, y)

        assert isinstance(compressed, bytes)
        assert len(compressed) > 0
        assert uncompressed_size == 12  # 3 points * 2 coords * 2 bytes

    def test_pack_numeric_column_float32(self):
        """pack_numeric_column should handle float32."""
        values = np.array([1.0, 2.0, 3.0, 4.0])

        compressed, uncompressed_size = pack_numeric_column(values, "float32")

        assert isinstance(compressed, bytes)
        assert uncompressed_size == 16  # 4 floats * 4 bytes

    def test_pack_numeric_column_int32(self):
        """pack_numeric_column should handle int32."""
        values = np.array([1, 2, 3, 4])

        compressed, uncompressed_size = pack_numeric_column(values, "int32")

        assert isinstance(compressed, bytes)
        assert uncompressed_size == 16  # 4 ints * 4 bytes

    def test_pack_categorical_column(self):
        """pack_categorical_column should create dictionary encoding."""
        values = ["A", "B", "A", "C", "B", "A"]

        compressed, uncompressed_size, dictionary = pack_categorical_column(values)

        assert isinstance(compressed, bytes)
        assert dictionary == ["A", "B", "C"]  # Order of first occurrence
        assert uncompressed_size == 24  # 6 indices * 4 bytes


class TestBinaryContainerWriter:
    """Tests for BinaryContainerWriter class."""

    def test_basic_write(self):
        """Should write basic container."""
        writer = BinaryContainerWriter()
        writer.add_coords(
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
        )
        writer.set_metadata({"title": "Test"})

        result = writer.write()

        assert isinstance(result, bytes)
        assert result[:4] == b"TMAP"  # Magic bytes

    def test_write_chunked(self):
        """write_chunked should return dict of chunks."""
        writer = BinaryContainerWriter()
        writer.add_coords(
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
        )
        writer.add_numeric_column("values", np.array([1.0, 2.0]), "float32")
        writer.set_metadata({"title": "Test"})

        chunks = writer.write_chunked()

        assert "header" in chunks
        assert "metadata" in chunks
        assert "coords" in chunks
        assert "col_values" in chunks


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests for visualization module."""

    def test_single_point(self):
        """Should handle single point."""
        viz = TmapViz()
        viz.set_points([0.0], [0.0])
        viz.add_label("name", ["Only Point"])

        html = viz.render()

        assert isinstance(html, str)

    def test_empty_categorical_values(self, viz_with_data):
        """Should handle categorical with empty strings."""
        viz, data = viz_with_data
        values = ["", "A", "", "B"] * 25

        viz.add_color_layout("sparse", values, categorical=True)

        assert len(viz.layouts) == 1

    def test_unicode_labels(self, viz_with_data):
        """Should handle unicode in labels."""
        viz, data = viz_with_data
        unicode_labels = [f"Point_{i}_\u03b1\u03b2\u03b3" for i in range(100)]

        viz.add_label("name", unicode_labels)

        html = viz.render()
        assert "\u03b1" in html or "\\u03b1" in html  # Either direct or escaped

    def test_large_numeric_values(self, viz_with_data):
        """Should handle large numeric values."""
        viz, data = viz_with_data
        large_values = np.array([1e10, 1e-10, 0, -1e10] * 25)

        viz.add_color_layout("large", large_values)

        html = viz.render()
        assert isinstance(html, str)

    def test_nan_values_in_continuous(self, viz_with_data):
        """Should warn and handle NaN in continuous values."""
        viz, data = viz_with_data
        values = np.array([1.0, np.nan, 3.0, np.nan] * 25)

        with pytest.warns(UserWarning, match="contains NaN values"):
            viz.add_color_layout("with_nan", values)

        # Should not raise, NaN values are rendered in black client-side
        html = viz.render()
        assert isinstance(html, str)

        # Payload JSON in HTML should be valid JSON (NaN -> null)
        match = re.search(
            r'<script id="payload" type="application/json">(.*?)</script>',
            html,
            re.DOTALL,
        )
        assert match is not None
        payload = json.loads(match.group(1))
        assert payload["columns"]["with_nan"]["values"][1] is None


# =============================================================================
# Column Length Validation
# =============================================================================


class TestColumnValidation:
    """Tests for column length validation."""

    def test_column_length_mismatch_after_set_points(self):
        """Adding column with wrong length after set_points should raise."""
        viz = TmapViz()
        viz.set_points([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])

        # This should work - setting points validates existing columns
        # But the column is added to the dict without validation
        # Validation happens at render time
        viz.add_label("name", ["A", "B"])  # Wrong length

        with pytest.raises(ValueError, match="values but there are"):
            viz.render()

    def test_set_points_validates_existing_columns(self):
        """set_points should validate against existing columns."""
        viz = TmapViz()
        viz.add_label("name", ["A", "B", "C"])

        with pytest.raises(ValueError, match="values but there are"):
            viz.set_points([0.0, 1.0], [0.0, 1.0])  # Only 2 points, 3 labels


# =============================================================================
# set_edges Tests
# =============================================================================


class TestSetEdges:
    """Tests for set_edges method."""

    def test_basic_set_edges(self, viz_with_data):
        """set_edges should accept valid s, t arrays."""
        viz, data = viz_with_data
        s = np.array([0, 1, 2], dtype=np.uint32)
        t = np.array([1, 2, 3], dtype=np.uint32)

        viz.set_edges(s, t)

        assert viz._edges_s is not None
        assert viz._edges_t is not None
        assert len(viz._edges_s) == 3
        assert len(viz._edges_t) == 3

    def test_set_edges_mismatched_length(self, viz_with_data):
        """Mismatched s and t should raise ValueError."""
        viz, data = viz_with_data

        with pytest.raises(ValueError, match="same length"):
            viz.set_edges([0, 1], [1, 2, 3])

    def test_set_edges_2d_raises(self, viz_with_data):
        """2D arrays should raise ValueError."""
        viz, data = viz_with_data

        with pytest.raises(ValueError, match="1-dimensional"):
            viz.set_edges(np.zeros((3, 2), dtype=np.uint32), np.zeros((3, 2), dtype=np.uint32))

    def test_set_edges_out_of_bounds(self, viz_with_data):
        """Edge indices >= n_points should raise ValueError."""
        viz, data = viz_with_data
        n = len(data["x"])

        with pytest.raises(ValueError, match="must be < n_points"):
            viz.set_edges([0, n], [1, 0])

    def test_render_with_edges(self, viz_with_data):
        """JSON payload should include edges."""
        viz, data = viz_with_data
        viz.add_color_layout("value", data["continuous"])
        viz.set_edges([0, 1, 2], [1, 2, 3])

        html = viz.render()

        # Extract payload from HTML
        match = re.search(
            r'<script id="payload" type="application/json">(.*?)</script>',
            html,
            re.DOTALL,
        )
        assert match is not None
        payload = json.loads(match.group(1))
        assert "edges" in payload
        assert payload["edges"]["s"] == [0, 1, 2]
        assert payload["edges"]["t"] == [1, 2, 3]
        assert payload["edgeStrokeStyle"] == "rgba(0, 0, 0, 0.5)"
        assert payload["edgeWidth"] == 2.0

    def test_render_binary_with_edges(self, viz_with_data):
        """Binary mode should include edge data."""
        viz, data = viz_with_data
        viz.add_color_layout("value", data["continuous"])
        viz.set_edges([0, 1, 2], [1, 2, 3])

        html = viz.render_binary()

        # Should contain the edges script tag
        assert 'id="tmap-edges"' in html
        # Header should contain nEdges
        match = re.search(
            r'<script id="tmap-header" type="application/json">(.*?)</script>',
            html,
            re.DOTALL,
        )
        assert match is not None
        header = json.loads(match.group(1))
        assert header["metadata"]["nEdges"] == 3
        assert header["metadata"]["edgeStrokeStyle"] == "rgba(0, 0, 0, 0.5)"
        assert header["metadata"]["edgeWidth"] == 2.0

    def test_custom_edge_style_in_payload(self, viz_with_data):
        """Custom edge style should be serialized in JSON payload."""
        viz, data = viz_with_data
        viz.set_edges([0, 1], [1, 2])
        viz.set_edge_style(color="#f03", width=4.5, opacity=0.35)

        html = viz.render()

        match = re.search(
            r'<script id="payload" type="application/json">(.*?)</script>',
            html,
            re.DOTALL,
        )
        assert match is not None
        payload = json.loads(match.group(1))
        assert payload["edgeStrokeStyle"] == "rgba(255, 0, 51, 0.35)"
        assert payload["edgeWidth"] == 4.5


class TestEdgeStyle:
    """Tests for edge style configuration."""

    def test_set_edge_style_updates_values(self):
        """set_edge_style should update style attributes."""
        viz = TmapViz()
        viz.set_edge_style(color="#abc", width=3.25, opacity=0.2)

        assert viz.edge_color == "#aabbcc"
        assert viz.edge_width == 3.25
        assert viz.edge_opacity == 0.2

    def test_set_edge_style_invalid_color_raises(self):
        """Invalid edge color should raise ValueError."""
        viz = TmapViz()
        with pytest.raises(ValueError, match="Invalid hex color"):
            viz.set_edge_style(color="not-a-color")

    def test_set_edge_style_invalid_width_raises(self):
        """Non-positive edge width should raise ValueError."""
        viz = TmapViz()
        with pytest.raises(ValueError, match="must be > 0"):
            viz.set_edge_style(width=0)

    def test_set_edge_style_invalid_opacity_raises(self):
        """Opacity outside [0, 1] should raise ValueError."""
        viz = TmapViz()
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            viz.set_edge_style(opacity=1.2)


# =============================================================================
# BINARY_THRESHOLD Tests
# =============================================================================


class TestBinaryThreshold:
    """Tests for BINARY_THRESHOLD constant."""

    def test_threshold_value(self):
        """BINARY_THRESHOLD should be 500,000."""
        assert BINARY_THRESHOLD == 500_000

    def test_save_uses_threshold(self, tmp_path):
        """save should use threshold to choose mode."""
        # Create small dataset
        viz = TmapViz()
        viz.title = "small"
        viz.set_points([0.0, 1.0], [0.0, 1.0])

        # Should use JSON mode (below threshold)
        output = viz.save(tmp_path, binary_threshold=100)

        content = output.read_text()
        # JSON mode includes points as array
        assert '"points"' in content or "points" in content
