"""
EIDORS Bridge - Python interface to EIDORS via Octave.
"""

import os
import tempfile
import subprocess
import json
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from numpy.typing import NDArray

from ..models import Trunk


SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")

def get_eidors_path() -> str:
    """Get EIDORS path from vendor/ directory or environment."""
    import os
    vendor_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "vendor", "eidors")
    env_path = os.path.expandvars("$EIDORS_PATH")
    if os.path.exists(vendor_path):
        return vendor_path
    elif os.path.exists(env_path):
        return env_path
    return "/home/ruben/Documentos/EIDORS_implementation/eidors_lib/eidors-v3.12"


EIDORS_PATH = get_eidors_path()


@dataclass
class MeshData:
    nodes: NDArray[np.float64]
    elems: NDArray[np.int32]


@dataclass
class EIDORSResult:
    voltages: NDArray[np.float64]
    mesh: MeshData
    elem_data: NDArray[np.float64]


def _anomalies_to_json(trunk: Trunk) -> str:
    """Convert trunk anomalies to JSON string for Octave."""
    anomalies = []
    for a in trunk.anomalies:
        anomalies.append({
            'cx': a.center.r * np.cos(a.center.phi),
            'cy': a.center.r * np.sin(a.center.phi),
            'radius': a.radius,
            'conductivity': a.conductivity
        })
    return json.dumps(anomalies) if anomalies else '[]'


def run_eidors_simulation(
    trunk: Trunk,
    n_electrodes: int = 16
) -> Optional[EIDORSResult]:
    """
    Run EIDORS forward simulation.

    Args:
        trunk: Trunk domain model.
        n_electrodes: Number of electrodes.

    Returns:
        EIDORSResult with voltages, mesh, and element data.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        anomalies_json = _anomalies_to_json(trunk)

        script = f"""
addpath('{SCRIPT_DIR}');
simulate({n_electrodes}, {trunk.base_conductivity}, '{anomalies_json}', '{tmpdir}');
"""

        script_path = os.path.join(tmpdir, "run.m")
        with open(script_path, 'w') as f:
            f.write(script)

        result = subprocess.run(
            ["octave", "--no-gui", "--quiet", script_path],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            print(f"EIDORS error: {result.stderr[:300]}")
            return None

        nodes_path = os.path.join(tmpdir, "nodes.txt")
        if not os.path.exists(nodes_path):
            print(f"Output files not created")
            return None

        nodes = np.loadtxt(nodes_path)
        elems = np.loadtxt(os.path.join(tmpdir, "elems.txt")).astype(np.int32)
        voltages = np.loadtxt(os.path.join(tmpdir, "voltages.txt"))
        elem_data = np.loadtxt(os.path.join(tmpdir, "elem_data.txt"))

        return EIDORSResult(
            voltages=voltages,
            mesh=MeshData(nodes=nodes, elems=elems),
            elem_data=elem_data
        )


def get_eidors_mesh(n_electrodes: int = 16) -> Optional[MeshData]:
    """Get mesh from EIDORS model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        script = f"""
addpath('{SCRIPT_DIR}');
[fmdl, img] = create_model({n_electrodes});
dlmwrite('{tmpdir}/nodes.txt', img.fwd_model.nodes, ' ');
dlmwrite('{tmpdir}/elems.txt', img.fwd_model.elems, ' ');
"""

        script_path = os.path.join(tmpdir, "get_mesh.m")
        with open(script_path, 'w') as f:
            f.write(script)

        result = subprocess.run(
            ["octave", "--no-gui", "--quiet", script_path],
            capture_output=True,
            text=True,
            timeout=120
        )

        nodes_path = os.path.join(tmpdir, "nodes.txt")
        if not os.path.exists(nodes_path):
            return None

        nodes = np.loadtxt(nodes_path)
        elems = np.loadtxt(os.path.join(tmpdir, "elems.txt")).astype(np.int32)

        return MeshData(nodes=nodes, elems=elems)