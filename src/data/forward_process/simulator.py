"""
Forward process simulator for Trunk models.

Orchestrates the full EIT forward pipeline:
  1. Conductivity grid   — via src/data_representations
  2. EIDORS FEM solve    — via scripts/run_forward.m (Octave subprocess)
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np
import scipy.io as sio
from numpy.typing import NDArray

from ..data_representations.grid import generate_grid
from ..models import Trunk

# Directory that holds the .m scripts bundled with this package
_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")


# ---------------------------------------------------------------------------
# Result dataclasses  (no dependency on src/eidors)
# ---------------------------------------------------------------------------


@dataclass
class MeshData:
    """FEM mesh returned by EIDORS."""

    nodes: NDArray[np.float64]  # shape [N, 2] — node coordinates
    elems: NDArray[np.int32]  # shape [E, 3] — 1-indexed connectivity


@dataclass
class EITResult:
    """Full output of the EIT forward simulation."""

    voltages: NDArray[np.float64]  # boundary voltage measurements [M]
    mesh: MeshData  # FEM mesh
    elem_data: NDArray[np.float64]  # per-element conductivities [E]


@dataclass
class ForwardResult:
    """Consolidated result of the forward process."""

    trunk: Trunk
    grid: NDArray[np.float64]  # 2-D conductivity grid from data_representations
    eit_result: EITResult


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def simulate_forward_process(
    trunk: Trunk,
    resolution: int = 128,
    n_electrodes: int = 16,
    pattern: str = "adjacent",
    octave_timeout: int = 300,
) -> ForwardResult | None:
    """Run the complete EIT forward process for a Trunk model.

    Steps:
    1. Generate a 2-D conductivity grid (src/data_representations).
    2. Write the grid to a temp file and invoke scripts/run_forward.m via
       Octave, which handles model creation, conductivity assignment and the
       FEM forward solve.
    3. Read back voltages, mesh and element data and return a ForwardResult.

    Args:
        trunk:           Domain model (geometry + conductivities).
        resolution:      Grid resolution (pixels along the longest axis).
        n_electrodes:    Number of EIT surface electrodes.
        pattern:         Measurement pattern ('adjacent', 'opposite', etc.).
        octave_timeout:  Maximum seconds to wait for the Octave subprocess.

    Returns:
        ForwardResult on success, or None if the simulation fails.
    """
    # ------------------------------------------------------------------
    # Step 1 — Generate the conductivity grid (Python / data_representations)
    # ------------------------------------------------------------------
    grid = generate_grid(trunk, resolution=resolution)
    x_min, x_max, y_min, y_max = trunk.get_bounds()
    base_conductivity = trunk.base.conductivity

    # ------------------------------------------------------------------
    # Step 2 — Run scripts/run_forward.m via Octave subprocess
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        grid_path = os.path.join(tmpdir, "grid.mat")
        sio.savemat(grid_path, {"grid": grid})

        startup_m = os.path.join(_SCRIPTS_DIR, "startup_eidors.m")
        driver = "\n".join(
            [
                # run() executes a script in the caller's workspace — not affected by addpath timing
                f"run('{startup_m}');",
                f"addpath('{_SCRIPTS_DIR}');",
                "startup_eidors();",
                "run_forward(",
                f"    {n_electrodes},",
                f"    '{pattern}',",
                f"    '{grid_path}',",
                f"    {x_min}, {x_max},",
                f"    {y_min}, {y_max},",
                f"    {base_conductivity},",
                f"    '{tmpdir}'",
                ");",
            ]
        )
        driver_path = os.path.join(_SCRIPTS_DIR, "_driver.m")
        with open(driver_path, "w") as fh:
            fh.write(driver)

        proc = subprocess.run(
            ["octave-cli", "--no-gui", "--quiet", driver_path],
            capture_output=True,
            text=True,
            timeout=octave_timeout,
        )

        if proc.returncode != 0:
            print(f"[forward_process] Octave error:\n{proc.stderr}")
            return None

        # ------------------------------------------------------------------
        # Step 3 — Read back outputs written by run_forward.m
        # ------------------------------------------------------------------
        results_path = os.path.join(tmpdir, "results.mat")
        if not os.path.exists(results_path):
            print("[forward_process] results.mat not written — simulation failed.")
            return None

        data = sio.loadmat(results_path)
        voltages = data['voltages'].squeeze()
        nodes = data['nodes']
        elems = data['elems'].astype(np.int32)
        elem_data = data['elem_data'].squeeze()

        eit_result = EITResult(
            voltages=voltages,
            mesh=MeshData(nodes=nodes, elems=elems),
            elem_data=elem_data,
        )

    return ForwardResult(trunk=trunk, grid=grid, eit_result=eit_result)
