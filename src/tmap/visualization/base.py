"""
Base class for visualization.

DESIGN NOTE: Simple interface
-----------------------------
Visualization is the "end of the pipeline" - it consumes
coordinates and metadata, produces output. Not much abstraction needed.

The main complexity is in the HTML template, not Python code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tmap.layout.types import Coordinates


@dataclass
class NodeMetadata:
    """
    Metadata for visualization - what to show on hover/click.

    This is GENERIC - not molecule-specific.
    Users pass whatever data they want displayed.

    Example for molecules:
        metadata = NodeMetadata(
            labels=smiles_list,
            data={"activity": activities, "mw": molecular_weights},
            colors=activity_colors,
        )

    Example for papers:
        metadata = NodeMetadata(
            labels=titles,
            data={"authors": authors, "year": years, "citations": citations},
            colors=year_colors,
        )
    """
    labels: list[str] | None = None  # Primary label per node
    data: dict[str, list[Any]] = field(default_factory=dict)  # Additional columns
    colors: list[str] | None = None  # Hex colors per node
    sizes: list[float] | None = None  # Size per node

    def __post_init__(self) -> None:
        """Validate that all lists have same length."""
        lengths = []
        if self.labels:
            lengths.append(("labels", len(self.labels)))
        if self.colors:
            lengths.append(("colors", len(self.colors)))
        if self.sizes:
            lengths.append(("sizes", len(self.sizes)))
        for name, values in self.data.items():
            lengths.append((name, len(values)))

        if lengths:
            first_len = lengths[0][1]
            for name, length in lengths[1:]:
                if length != first_len:
                    raise ValueError(
                        f"All metadata must have same length. "
                        f"{lengths[0][0]} has {first_len}, {name} has {length}"
                    )


class Visualizer(ABC):
    """
    Abstract base for visualization output.

    Implementations:
    - HTMLVisualizer: Interactive HTML with pan/zoom/search
    - (Future) ImageVisualizer: Static PNG/SVG export
    """

    @abstractmethod
    def render(
        self,
        coords: Coordinates,
        metadata: NodeMetadata | None = None,
        output_path: str | Path | None = None,
    ) -> str | bytes:
        """
        Render visualization.

        Args:
            coords: Layout coordinates
            metadata: Optional node metadata for tooltips/colors
            output_path: If provided, save to file

        Returns:
            HTML string or image bytes, depending on implementation
        """
        ...


class HTMLVisualizer(Visualizer):
    """
    Interactive HTML visualization.

    Features to implement:
    - [x] Basic canvas rendering
    - [ ] WebGL for large datasets (1M+ nodes)
    - [ ] Pan and zoom
    - [ ] Hover tooltips
    - [ ] Click to select
    - [ ] Search/filter by label
    - [ ] Copy data to clipboard
    - [ ] Color by category
    - [ ] Size by value

    TEMPLATE SYSTEM
    ---------------
    The HTML is generated from a Jinja2 template.
    Template location: src/tmap/templates/tmap.html

    You customize appearance by editing the template,
    not by changing Python code.
    """

    def __init__(
        self,
        template_path: str | Path | None = None,
        title: str = "TreeMap Visualization",
    ) -> None:
        """
        Initialize HTML visualizer.

        Args:
            template_path: Custom template path (uses default if None)
            title: HTML page title
        """
        self.title = title

        if template_path:
            self.template_path = Path(template_path)
        else:
            # Default template location
            self.template_path = (
                Path(__file__).parent / "templates" / "tmap.html"
            )

    def render(
        self,
        coords: Coordinates,
        metadata: NodeMetadata | None = None,
        output_path: str | Path | None = None,
    ) -> str:
        """
        Render to HTML.

        TODO: Implement this using Jinja2
        1. Load template
        2. Prepare data (coords, metadata as JSON)
        3. Render template
        4. Save to file if output_path provided
        5. Return HTML string
        """
        # Placeholder - you'll implement this
        raise NotImplementedError(
            "HTMLVisualizer.render() not yet implemented. "
            "See docstring for implementation guide."
        )

    def _prepare_data(
        self,
        coords: Coordinates,
        metadata: NodeMetadata | None,
    ) -> dict[str, Any]:
        """
        Prepare data for template.

        Converts numpy arrays to JSON-serializable format.
        """
        data = {
            "x": coords.x.tolist(),
            "y": coords.y.tolist(),
            "n_nodes": coords.n_nodes,
        }

        if metadata:
            if metadata.labels:
                data["labels"] = metadata.labels
            if metadata.colors:
                data["colors"] = metadata.colors
            if metadata.sizes:
                data["sizes"] = metadata.sizes
            if metadata.data:
                data["metadata"] = {
                    k: v if isinstance(v, list) else list(v)
                    for k, v in metadata.data.items()
                }

        return data
