"""
Grid generation for tree trunk EIT.
"""

import numpy as np
from numpy.typing import NDArray

from ..models import Pos, Trunk


def generate_grid(trunk: Trunk, resolution: int = 128) -> NDArray[np.float64]:
    """
    Generate a 2D conductivity grid from a Trunk domain model, 
    strictly respecting the physical aspect ratio.

    Args:
        trunk: Domain model containing trunk geometry and anomalies.
        resolution: Number of pixels along the longest axis (default 128).

    Returns:
        2D numpy array of shape (ny, nx) containing conductivity values.
    """
    x_min, x_max, y_min, y_max = trunk.get_bounds()

    width = x_max - x_min
    height = y_max - y_min

    if width > height:
        nx = resolution
        ny = max(1, int(resolution * (height / width)))
    else:
        nx = max(1, int(resolution * (width / height)))
        ny = resolution

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)

    def evaluate_pixel(px: float, py: float) -> float:
        pos = Pos.from_cartesian(px, py)
        return trunk.get_conductivity(pos, mode="Sum", add_base_cond=True)

    vectorized_eval = np.vectorize(evaluate_pixel)
    grid = vectorized_eval(X, Y)

    return grid


def grid2png(
    grid: NDArray[np.float64], path: str, vmin: float | None = None, vmax: float | None = None
) -> None:
    """
    Save conductivity grid as a PNG image for visualization.

    Args:
        grid: 2D conductivity array in S/m.
        path: Output file path.
        vmin: Minimum value for color scale (auto if None).
        vmax: Maximum value for color scale (auto if None).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grid, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title("Conductivity (S/m)")
    fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=150)
    plt.close(fig)
