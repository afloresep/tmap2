"""
Type definitions for the layout module.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class Coordinates:
    """
    2D layout coordinates.

    Attributes:
        x: X coordinates, shape (n_nodes,)
        y: Y coordinates, shape (n_nodes,)
        scale: Optional scaling factor used during layout

    The coordinates are normalized to roughly [-1, 1] or [0, 1] range,
    depending on the layout algorithm. The visualization module
    handles final scaling to screen coordinates.
    """
    x: NDArray[np.float32]
    y: NDArray[np.float32]
    scale: float = 1.0

    @property
    def n_nodes(self) -> int:
        return len(self.x)

    def __post_init__(self) -> None:
        if len(self.x) != len(self.y):
            raise ValueError(
                f"x and y must have same length, got {len(self.x)} and {len(self.y)}"
            )

    def normalize(self, margin: float = 0.1) -> "Coordinates":
        """
        Normalize coordinates to [margin, 1-margin] range.

        Useful for ensuring all points fit in visualization bounds.
        """
        x_min, x_max = self.x.min(), self.x.max()
        y_min, y_max = self.y.min(), self.y.max()

        # Avoid division by zero for single-point or degenerate cases
        x_range = max(x_max - x_min, 1e-10)
        y_range = max(y_max - y_min, 1e-10)

        # Scale to [0, 1]
        x_norm = (self.x - x_min) / x_range
        y_norm = (self.y - y_min) / y_range

        # Apply margin
        usable = 1.0 - 2 * margin
        x_norm = x_norm * usable + margin
        y_norm = y_norm * usable + margin

        return Coordinates(
            x=x_norm.astype(np.float32),
            y=y_norm.astype(np.float32),
            scale=self.scale,
        )

    def to_array(self) -> NDArray[np.float32]:
        """Return as (n_nodes, 2) array."""
        return np.column_stack([self.x, self.y])

    @classmethod
    def from_array(cls, arr: NDArray[np.float32]) -> "Coordinates":
        """Create from (n_nodes, 2) array."""
        return cls(x=arr[:, 0], y=arr[:, 1])
