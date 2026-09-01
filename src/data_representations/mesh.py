"""
Mesh-level conductivity assignment for Trunk models.

Converts a Trunk domain model into per-element conductivity data
for a given FEM mesh (nodes + elements), mirroring the element
assignment loop in EIDORS simulate.m but in pure Python.
"""

import numpy as np
from numpy.typing import NDArray

from ..models import Pos, Trunk


def trunk_to_elem_data(
    trunk: Trunk,
    nodes: NDArray[np.float64],
    elems: NDArray[np.int32],
    mode: str = "Max",
    add_base_cond: bool = False,
    check_trunk_boundary: bool = False,
) -> NDArray[np.float64]:
    """
    Convert a Trunk model to per-element conductivity for a given mesh.

    For each triangular element, computes the centroid and assigns conductivity
    based on anomaly containment. By default, all elements start with
    ``trunk.base.conductivity`` (matching EIDORS simulate.m behavior), and
    anomalies override as needed.

    Python equivalent of the element loop in ``src/eidors/scripts/simulate.m``
    (lines 36-88).

    Args:
        trunk: The trunk model.
        nodes: (N, 2) array of node coordinates.
        elems: (M, 3) array of element node indices (MATLAB 1-based or 0-based).
        mode: ``"Max"`` or ``"Sum"`` for overlapping anomalies.
        add_base_cond: Whether to add base conductivity to anomaly values.
        check_trunk_boundary: If True, elements outside the trunk base shape
            get conductivity 0. If False (default, Octave-compatible), all
            elements start at ``trunk.base.conductivity``.

    Returns:
        (M,) array of conductivity values per element.
    """
    if elems.max() >= len(nodes):
        elems = elems - 1
    centroids = nodes[elems].mean(axis=1)

    base_cond = trunk.base.conductivity
    elem_data = np.full(len(centroids), base_cond, dtype=np.float64)

    for i, (x, y) in enumerate(centroids):
        pos = Pos.from_cartesian(float(x), float(y))

        if check_trunk_boundary and not trunk.base.contains(pos):
            elem_data[i] = 0.0
            continue

        for a in trunk.anomalies:
            if not a.contains(pos):
                continue
            val = a.get_conductivity(pos)
            if mode == "Max":
                elem_data[i] = max(elem_data[i], val + (base_cond if add_base_cond else 0.0))
            elif mode == "Sum":
                elem_data[i] += val + (base_cond if add_base_cond else 0.0)

    return elem_data
