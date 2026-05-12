"""
Grid generation for tree trunk EIT.
"""

import numpy as np
from numpy.typing import NDArray

from ..models import Trunk, Anomaly


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
        shape = anomaly.shape
        cx = getattr(shape, "cx", 0)
        cy = getattr(shape, "cy", 0)

        if hasattr(shape, "contains"):
            for i in range(resolution):
                for j in range(resolution):
                    px = x[j]
                    py = y[i]
                    if shape.contains(px, py):
                        grid[i, j] = anomaly.conductivity
        else:
            radius = getattr(shape, "radius", 0.1)
            dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
            anomaly_mask = dist_sq <= radius**2
            grid[anomaly_mask] = anomaly.conductivity

    return grid
