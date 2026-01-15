"""
Layout module: Compute 2D coordinates for tree visualization.

DESIGN PATTERN: Strategy Pattern (continued)
--------------------------------------------
Layout is where multiple algorithms make sense:
- Force-directed (like OGDF, Gray et al.)
- Radial/polar layouts
- Hierarchical top-down
- Space-filling (tmap style)

Each is a "strategy" - same interface, different algorithm:

    layout = ForceDirectedLayout(seed=42)
    # or: layout = RadialLayout(seed=42)
    # or: layout = HierarchicalLayout(seed=42)

    coords = layout.compute(tree)

The TreeMap orchestrator doesn't care WHICH layout - it just calls compute().


KEY REQUIREMENT: DETERMINISM
----------------------------
Your requirement: "add determinism/seeding for reproducible layouts"

Force-directed layouts use randomness for:
1. Initial node positions
2. Randomized optimization steps

We ensure determinism by:
1. Accepting seed in __init__
2. Using numpy.random.Generator (not global random state)
3. Documenting which operations are randomized
"""

from tmap.layout.base import Layout
from tmap.layout.types import Coordinates

__all__ = ["Layout", "Coordinates"]

# Uncomment as you implement:
# from tmap.layout.force_directed import ForceDirectedLayout
# from tmap.layout.radial import RadialLayout
