"""
Dataset generation pipeline using src/forward_process.
Generates random trunks via StochasticTrunkFactory and saves:
  - grid/<i>.png    — conductivity image
  - json/<i>.json   — serialized Trunk model
  - voltages/<i>.npy — boundary voltage measurements
  - elem_data/<i>.npy — per-element conductivities
  - mesh/nodes.npy, mesh/elems.npy — shared FEM mesh (saved once)
"""

import sys
import json
import random
from pathlib import Path

# Fix python path to allow importing 'src' from the parent directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src.forward_process import ForwardResult, simulate_forward_process
from src.models import Trunk
from src.stochastic.generate_harmonic import StochasticTrunkFactory

# ---------------------------------------------------------------------------
# Dataset factories
# ---------------------------------------------------------------------------

def get_dataset(n: int) -> list[Trunk]:
    class PaperTrunkFactory(StochasticTrunkFactory):
        def _poisson_sample(self, lam: float) -> int:
            return random.randint(1, 3)

    factory = PaperTrunkFactory(
        trunk_radius_mu=1.0,
        trunk_radius_sigma=0.1,    # Varia el tamaño del tronco
        base_cond_mu=1.0,
        base_cond_sigma=0.1,       # Varia la conductividad de la madera sana
        wet_cond_mu=100.0,
        wet_cond_sigma=25.0,       # Varia la conductividad de la anomalía (distribución normal)
        propagation_modes=["Linear", "Cos"],
        anomaly_scale=0.25         # Tamaño mediano de la anomalía
    )
    return [factory.generate() for _ in range(n)]


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_sample(result: ForwardResult, i: int, base_path: Path) -> None:
    """Persist all outputs for one sample."""
    # Grid image
    plt.imsave(base_path / f"grid/{i}.png", result.grid, cmap="viridis")

    # Trunk model as JSON
    with open(base_path / f"json/{i}.json", "w", encoding="utf-8") as fh:
        json.dump(result.trunk.to_dict(), fh, indent=2)

    # EIT measurements
    np.save(base_path / f"voltages/{i}.npy", result.eit_result.voltages)
    np.save(base_path / f"elem_data/{i}.npy", result.eit_result.elem_data)


def save_mesh_once(result: ForwardResult, base_path: Path) -> None:
    """Save the shared FEM mesh (same for all samples with same n_electrodes)."""
    np.save(base_path / "mesh/nodes.npy", result.eit_result.mesh.nodes)
    np.save(base_path / "mesh/elems.npy", result.eit_result.mesh.elems)
    print(
        f"  Mesh saved: {len(result.eit_result.mesh.nodes)} nodes, "
        f"{len(result.eit_result.mesh.elems)} elements"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Set random seeds for reproducibility (generates the exact same dataset every run)
    random.seed(42)
    np.random.seed(42)

    N_SAMPLES = 10_220
    N_ELECTRODES = 16
    GRID_RESOLUTION = 64

    base_path = PROJECT_ROOT / "dataset"
    for sub in ("grid", "json", "voltages", "elem_data", "mesh"):
        (base_path / sub).mkdir(parents=True, exist_ok=True)

    print(f"🌲 Generating {N_SAMPLES} random trunks...")
    dataset = get_dataset(N_SAMPLES)

    mesh_saved = False
    failed = 0

    for i, trunk in enumerate(dataset):
        print(f"  [{i + 1}/{N_SAMPLES}] Simulating...", end=" ", flush=True)
        result = simulate_forward_process(
            trunk, resolution=GRID_RESOLUTION, n_electrodes=N_ELECTRODES
        )

        if result is None:
            print("FAILED — skipping")
            failed += 1
            continue

        if not mesh_saved:
            save_mesh_once(result, base_path)
            mesh_saved = True

        save_sample(result, i, base_path)
        print(f"✓  voltages={result.eit_result.voltages.shape}, grid={result.grid.shape}")

    ok = N_SAMPLES - failed
    print(f"\n✅ Done: {ok}/{N_SAMPLES} samples saved to '{base_path}/'")
