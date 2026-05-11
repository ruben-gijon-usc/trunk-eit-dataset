
import numpy as np
from numpy.typing import NDArray

from ..models import Trunk


def generate_grid(trunk: Trunk, resolution: int = 128) -> NDArray[np.float64]:
    """
    Generate a 2D conductivity grid from a Trunk domain model.

    Uses vectorized numpy operations to efficiently rasterize the trunk
    cross-section and its anomalies onto a uniform grid.

    Args:
        trunk: Domain model containing trunk geometry and anomalies.
        resolution: Number of pixels along each axis (default 128).

    Returns:
        2D numpy array of shape (resolution, resolution) containing
        conductivity values in S/m.
    """
    x = np.linspace(-trunk.radius, trunk.radius, resolution)
    y = np.linspace(-trunk.radius, trunk.radius, resolution)
    X, Y = np.meshgrid(x, y)

    grid = np.full((resolution, resolution), trunk.base_conductivity)

    trunk_mask = (X**2 + Y**2) <= trunk.radius**2
    grid[~trunk_mask] = 0.0

    for anomaly in trunk.anomalies:
        cx = anomaly.center.r * np.cos(anomaly.center.phi)
        cy = anomaly.center.r * np.sin(anomaly.center.phi)
        dist_sq = (X - cx)**2 + (Y - cy)**2
        anomaly_mask = dist_sq <= anomaly.radius**2
        grid[anomaly_mask] = anomaly.conductivity

    return grid


def grid2png(grid: NDArray[np.float64], path: str, vmin: float | None = None,
             vmax: float | None = None) -> None:
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
    im = ax.imshow(grid, cmap='viridis', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title('Conductivity (S/m)')
    fig.colorbar(im, ax=ax)
    fig.savefig(path, dpi=150)
    plt.close(fig)
