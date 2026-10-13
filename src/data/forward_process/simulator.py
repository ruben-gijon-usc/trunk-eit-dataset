"""
Forward process simulator for Trunk models.

Orchestrates the full EIT forward pipeline:
  1. Conductivity grid   — via src/data_representations
  2. EIDORS FEM solve    — via scripts/run_forward.m (Octave subprocess)
"""

from __future__ import annotations

import os
import shutil
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


def get_octave_binary() -> str | None:
    """Find the Octave executable on the system ('octave-cli' or 'octave')."""
    return shutil.which("octave-cli") or shutil.which("octave")


def check_octave_installed() -> str:
    """Verify that Octave is available in PATH, raising RuntimeError if not.

    Returns:
        The path to the Octave executable.

    Raises:
        RuntimeError: If neither 'octave-cli' nor 'octave' is installed or in PATH.
    """
    binary = get_octave_binary()
    if binary is None:
        raise RuntimeError(
            "Octave is required for the forward EIT simulation, but neither 'octave-cli' nor "
            "'octave' was found in your PATH.\n"
            "Please install GNU Octave (e.g., 'sudo apt install octave' on Ubuntu/Debian, "
            "or 'brew install octave' on macOS) and ensure it is available in PATH."
        )
    return binary


def check_eidors_available(octave_bin: str | None = None) -> None:
    """Verify that EIDORS can be initialized via Octave, raising RuntimeError if not.

    Runs startup_eidors() in a minimal Octave invocation to ensure EIDORS is accessible
    either through $EIDORS_PATH or vendor/eidors.

    Raises:
        RuntimeError: If Octave fails to initialize EIDORS.
    """
    if octave_bin is None:
        octave_bin = check_octave_installed()

    startup_m = os.path.join(_SCRIPTS_DIR, "startup_eidors.m")
    test_cmd = (
        f"run('{startup_m}'); addpath('{_SCRIPTS_DIR}'); startup_eidors(); disp('EIDORS_OK');"
    )

    try:
        proc = subprocess.run(
            [octave_bin, "--no-gui", "--quiet", "--eval", test_cmd],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to execute Octave to test EIDORS: {exc}") from exc

    if proc.returncode != 0 or "EIDORS_OK" not in proc.stdout:
        err_msg = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(
            f"EIDORS initialization failed in Octave:\n{err_msg}\n\n"
            "How to fix:\n"
            "1. Download EIDORS from http://eidors3d.sourceforge.net/ (e.g. eidors-v3.12).\n"
            "2. Set the EIDORS_PATH environment variable to the extracted folder:\n"
            "   export EIDORS_PATH=/path/to/eidors-v3.12/eidors\n"
            "   (Or place it in vendor/eidors inside the repository)."
        )


# ---------------------------------------------------------------------------
# Result dataclasses  (no dependency on src/eidors)
# ---------------------------------------------------------------------------


def interpolate_nodal_voltages_to_grid(
    nodes: NDArray[np.float64],
    nodal_voltages: NDArray[np.float64],
    trunk: Trunk,
    resolution: int = 64,
) -> NDArray[np.float64]:
    """Interpolate FEM nodal potentials onto a 2D regular grid matching the trunk bounds."""
    from scipy.interpolate import griddata

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

    if nodal_voltages.ndim == 1:
        grid_v = griddata(nodes, nodal_voltages, (X, Y), method="linear", fill_value=0.0)
        return grid_v.astype(np.float32)
    else:
        P = nodal_voltages.shape[1]
        grid_maps = np.zeros((P, ny, nx), dtype=np.float32)
        for p in range(P):
            grid_maps[p] = griddata(
                nodes, nodal_voltages[:, p], (X, Y), method="linear", fill_value=0.0
            )
        return grid_maps


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
    nodal_voltages: NDArray[np.float64] | None = None  # internal nodal potentials [N, P]
    voltage_maps: NDArray[np.float64] | None = None  # 2D voltage maps [P, H, W]


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
    save_voltage_maps: bool = False,
) -> ForwardResult | None:
    """Run the complete EIT forward process for a Trunk model.

    Steps:
    1. Generate a 2-D conductivity grid (src/data_representations).
    2. Write the grid to a temp file and invoke scripts/run_forward.m via
       Octave, which handles model creation, conductivity assignment and the
       FEM forward solve.
    3. Read back voltages, mesh and element data and return a ForwardResult.

    Args:
        trunk:               Domain model (geometry + conductivities).
        resolution:          Grid resolution (pixels along the longest axis).
        n_electrodes:        Number of EIT surface electrodes.
        pattern:             Measurement pattern ('adjacent', 'opposite', etc.).
        octave_timeout:      Maximum seconds to wait for the Octave subprocess.
        save_voltage_maps:   If True, solves and interpolates internal 2D voltage maps V(x,y).

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
                f"    '{tmpdir}',",
                f"    {'true' if save_voltage_maps else 'false'}",
                ");",
            ]
        )
        driver_path = os.path.join(tmpdir, "_driver.m")
        with open(driver_path, "w") as fh:
            fh.write(driver)

        octave_bin = get_octave_binary()
        if octave_bin is None:
            print(
                "[forward_process] 'octave' / 'octave-cli' not found in PATH. EIDORS simulation skipped."
            )
            return None

        proc = subprocess.run(
            [octave_bin, "--no-gui", "--quiet", driver_path],
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
        voltages = data["voltages"].squeeze()
        nodes = data["nodes"]
        elems = data["elems"].astype(np.int32)
        elem_data = data["elem_data"].squeeze()

        nodal_voltages = data.get("nodal_voltages")
        voltage_maps = None
        if save_voltage_maps and nodal_voltages is not None:
            voltage_maps = interpolate_nodal_voltages_to_grid(
                nodes=nodes,
                nodal_voltages=nodal_voltages,
                trunk=trunk,
                resolution=resolution,
            )

        eit_result = EITResult(
            voltages=voltages,
            mesh=MeshData(nodes=nodes, elems=elems),
            elem_data=elem_data,
            nodal_voltages=nodal_voltages,
            voltage_maps=voltage_maps,
        )

    return ForwardResult(trunk=trunk, grid=grid, eit_result=eit_result)
