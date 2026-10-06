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
    """
    x_min, x_max, y_min, y_max = trunk.get_bounds()
    
    # Add 5% padding so the trunk doesn't touch the edges of the image
    padx = (x_max - x_min) * 0.05
    pady = (y_max - y_min) * 0.05
    x_min, x_max = x_min - padx, x_max + padx
    y_min, y_max = y_min - pady, y_max + pady

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


def generate_mask(trunk: Trunk, resolution: int = 128) -> NDArray[np.float64]:
    """
    Generate a 2D boolean mask from a Trunk domain model, strictly based on its physical boundary.
    """
    x_min, x_max, y_min, y_max = trunk.get_bounds()

    padx = (x_max - x_min) * 0.05
    pady = (y_max - y_min) * 0.05
    x_min, x_max = x_min - padx, x_max + padx
    y_min, y_max = y_min - pady, y_max + pady

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

    def evaluate_mask(px: float, py: float) -> float:
        pos = Pos.from_cartesian(px, py)
        return float(trunk.base.contains(pos))

    vectorized_eval = np.vectorize(evaluate_mask)
    mask = vectorized_eval(X, Y)

    return mask

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
