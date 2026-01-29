"""Tests for OGDF layout extension."""

import math
import pytest

from tmap.layout import OGDF_AVAILABLE

pytestmark = pytest.mark.skipif(not OGDF_AVAILABLE, reason="OGDF extension not built")

# Conditional imports - only available when OGDF is built
if OGDF_AVAILABLE:
    from tmap.layout import LayoutConfig, layout_from_edge_list


def _edge_set(s, t):
    """Convert edge arrays to set of frozensets for comparison."""
    return {frozenset((si, ti)) for si, ti in zip(s, t)}


def test_single_node_returns_origin():
    x, y, s, t = layout_from_edge_list(1, [])
    assert list(x) == [0.0]
    assert list(y) == [0.0]
    assert list(s) == []
    assert list(t) == []


def test_negative_weights_are_filtered():
    edges = [(0, 1, -1.0), (1, 2, 0.5)]
    x, y, s, t = layout_from_edge_list(3, edges, create_mst=True)
    assert _edge_set(s, t) == {frozenset({1, 2})}


def test_mst_reduces_edges():
    edges = [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)]
    x, y, s, t = layout_from_edge_list(3, edges, create_mst=True)
    assert _edge_set(s, t) == {frozenset({0, 1}), frozenset({1, 2})}


def test_coordinates_normalized_range():
    edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
    x, y, s, t = layout_from_edge_list(4, edges, create_mst=False)
    all_vals = list(x) + list(y)
    assert all(-0.50001 <= v <= 0.50001 for v in all_vals)
    assert any(
        abs(v - (-0.5)) < 1e-3 or abs(v - 0.5) < 1e-3 for v in all_vals
    )


def test_deterministic_mode_reproducible():
    edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)]
    cfg = LayoutConfig()
    cfg.deterministic = True
    cfg.seed = 123

    x1, y1, s1, t1 = layout_from_edge_list(5, edges, cfg, create_mst=False)
    x2, y2, s2, t2 = layout_from_edge_list(5, edges, cfg, create_mst=False)

    assert list(s1) == list(s2) and list(t1) == list(t2)
    assert len(x1) == len(x2) == 5
    assert all(math.isclose(a, b, rel_tol=0, abs_tol=1e-6) for a, b in zip(x1, x2))
    assert all(math.isclose(a, b, rel_tol=0, abs_tol=1e-6) for a, b in zip(y1, y2))
