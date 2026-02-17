"""Type stubs for the _tmap_ogdf C++ extension (pybind11)."""

import enum

class Placer(enum.Enum):
    """Initial placement strategy during uncoarsening."""

    Barycenter: int
    Solar: int
    Circle: int
    Median: int
    Random: int
    Zero: int

class Merger(enum.Enum):
    """Graph coarsening strategy."""

    EdgeCover: int
    LocalBiconnected: int
    Solar: int
    IndependentSet: int

class ScalingType(enum.Enum):
    """Scaling strategy for layout."""

    Absolute: int
    RelativeToAvgLength: int
    RelativeToDesiredLength: int
    RelativeToDrawing: int

class LayoutConfig:
    """Configuration for OGDF layout."""

    k: int
    """Number of nearest neighbors for k-NN graph (default: 10)."""

    kc: int
    """Query multiplier for LSH (queries k*kc, keeps k) (default: 10)."""

    fme_iterations: int
    """FastMultipoleEmbedder iterations (default: 1000)."""

    fme_precision: int
    """Multipole expansion precision (default: 4)."""

    sl_repeats: int
    """ScalingLayout repeats (default: 1)."""

    sl_extra_scaling_steps: int
    """Extra scaling steps (default: 2)."""

    sl_scaling_min: float
    """Minimum scaling (default: 1.0)."""

    sl_scaling_max: float
    """Maximum scaling (default: 1.0)."""

    sl_scaling_type: ScalingType
    """Scaling type (default: RelativeToDrawing)."""

    mmm_repeats: int
    """ModularMultilevelMixer repeats (default: 1)."""

    placer: Placer
    """Placer algorithm (default: Barycenter)."""

    merger: Merger
    """Merger algorithm (default: LocalBiconnected)."""

    merger_factor: float
    """Merger factor (default: 2.0)."""

    merger_adjustment: int
    """Edge length adjustment (default: 0)."""

    node_size: float
    """Node size for repulsion (default: 1/65)."""

    deterministic: bool
    """Enable deterministic mode (single thread, seeded RNG)."""

    seed: int | None
    """Random seed (None for unseeded)."""

    def __init__(self) -> None: ...

class LayoutResult:
    """Layout computation result."""

    x: list[float]
    """X coordinates."""

    y: list[float]
    """Y coordinates."""

    s: list[int]
    """Edge source indices."""

    t: list[int]
    """Edge target indices."""

def layout_from_edge_list(
    vertex_count: int,
    edges: list[tuple[int, int, float]],
    config: LayoutConfig = ...,
    create_mst: bool = True,
) -> LayoutResult:
    """Compute 2D layout from edge list using OGDF."""
    ...
