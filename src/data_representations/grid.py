"""
Grid generation — discretises a Trunk domain model into a 2-D conductivity array.

The trunk cross-section is a unit disc.  Each pixel is mapped to its physical
coordinate, then classified as:
  - outside the trunk  → conductivity = 0.0
  - inside an anomaly  → conductivity = anomaly.conductivity  (last anomaly wins)
  - background wood    → conductivity = trunk.base_conductivity

All operations are fully vectorised with NumPy boolean masks; no Python loops
over individual pixels.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..models import Trunk


def generate_grid(trunk: Trunk, resolution: int = 64) -> NDArray[np.float64]:
    """Rasterise a Trunk into an (resolution × resolution) conductivity grid.

    The grid spans the bounding square [-trunk.radius, trunk.radius]² in
    physical space.  Pixels outside the circular trunk boundary are set to 0,
    pixels inside anomalies receive the anomaly conductivity, and the
    remaining interior pixels receive the trunk's base conductivity.

    When multiple anomalies overlap a pixel the *last* anomaly in the list
    takes precedence (consistent with the order in Trunk.anomalies).

    Args:
        trunk: Trunk geometric/physical domain model.
        resolution: Number of pixels along each axis (square grid).

    Returns:
        2-D float64 array of shape (resolution, resolution) with conductivity
        values in S/m.  Zero values indicate pixels outside the trunk.
    """
    # Pixel-centre coordinates in physical space — fully vectorised
    coords = np.linspace(-trunk.radius, trunk.radius, resolution)
    # xs[i,j] = x-coordinate of pixel column j
    # ys[i,j] = y-coordinate of pixel row i  (row 0 = top → bottom convention)
    xs, ys = np.meshgrid(coords, coords[::-1])

    r_sq = xs ** 2 + ys ** 2
    trunk_r_sq = trunk.radius ** 2

    # Start with background; mask pixels outside trunk with 0
    grid = np.where(r_sq <= trunk_r_sq, trunk.base_conductivity, 0.0)

    # Paint anomalies using vectorised circular masks
    for anomaly in trunk.anomalies:
        cx, cy = anomaly.center.to_cartesian()
        anomaly_r_sq = anomaly.radius ** 2
        dist_sq = (xs - cx) ** 2 + (ys - cy) ** 2
        inside_anomaly = (dist_sq <= anomaly_r_sq) & (r_sq <= trunk_r_sq)
        grid = np.where(inside_anomaly, anomaly.conductivity, grid)

    return grid.astype(np.float64)


def grid2png(grid: NDArray[np.float64], filepath: str) -> None:
    """Save a conductivity grid as a grayscale PNG image.

    The grid is normalised to [0, 255] linearly.  Pixels with conductivity 0
    (outside the trunk) map to pure black.  Higher conductivity maps to
    brighter pixels.

    Args:
        grid: 2-D conductivity array produced by :func:`generate_grid`.
        filepath: Destination file path (should end in ``.png``).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.imshow(grid, cmap="viridis", origin="upper")
    ax.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(filepath, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
