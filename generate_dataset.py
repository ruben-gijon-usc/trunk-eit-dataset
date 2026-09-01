"""
Dataset generation pipeline using src/forward_process.
Generates random trunks via StochasticTrunkFactory and saves:
  - grid/<i>.png    — conductivity image
  - json/<i>.json   — serialized Trunk model
  - voltages/<i>.npy — boundary voltage measurements
  - elem_data/<i>.npy — per-element conductivities
  - mesh/nodes.npy, mesh/elems.npy — shared FEM mesh (saved once)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.forward_process import ForwardResult, simulate_forward_process
from src.models import Trunk
from src.stochastic.generate_harmonic import StochasticTrunkFactory


# ---------------------------------------------------------------------------
# Dataset factories
# ---------------------------------------------------------------------------

def get_dataset(n: int) -> list[Trunk]:
    healthy_factory = StochasticTrunkFactory(
        trunk_radius_mu=0.3,
        expected_anomalies=1.5,
        pos_alpha=1.0,
        pos_beta=3.0,  # biased toward center
    )
    decay_factory = StochasticTrunkFactory(
        trunk_radius_mu=1.2,
        trunk_radius_sigma=0.3,
        expected_anomalies=8.0,
        anomaly_scale=0.15,
        pos_alpha=3.0,
        pos_beta=1.0,  # biased toward bark
    )

    dataset = [decay_factory.generate() for _ in range(n // 2)]
    dataset += [healthy_factory.generate() for _ in range(n - n // 2)]
    return dataset


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
    N_SAMPLES = 6
    N_ELECTRODES = 16
    GRID_RESOLUTION = 64

    base_path = Path("dataset")
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
