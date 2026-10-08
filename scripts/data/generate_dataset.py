"""
Dataset generation pipeline using src/forward_process.
Generates random trunks via StochasticTrunkFactory and saves:
  - grid/<i>.png    — conductivity image
  - json/<i>.json   — serialized Trunk model
  - voltages/<i>.npy — boundary voltage measurements
  - elem_data/<i>.npy — per-element conductivities
  - mesh/nodes.npy, mesh/elems.npy — shared FEM mesh (saved once)
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Fix python path to allow importing 'src' from the parent directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import math

import matplotlib.pyplot as plt
import numpy as np

from src.data.data_representations.grid import generate_mask
from src.data.forward_process import ForwardResult, simulate_forward_process
from src.data.models import Anomaly, Pos, SimpleTrunk, Trunk
from src.data.models.shape import Harmonic
from src.data.stochastic.generate_harmonic import StochasticTrunkFactory

from scripts.data.config import (
    ANOMALIES_MAX,
    ANOMALIES_MIN,
    ANOMALY_RADIUS_MAX,
    ANOMALY_RADIUS_MIN,
    COND_ANOMALY_MAX,
    COND_ANOMALY_MIN,
    COND_BASE_MAX,
    COND_BASE_MIN,
    TRUNK_RADIUS_MAX,
    TRUNK_RADIUS_MIN,
)

# ---------------------------------------------------------------------------
# Dataset factories
# ---------------------------------------------------------------------------


def get_dataset(n: int) -> list[Trunk]:
    class UniformMCFactory(StochasticTrunkFactory):
        def _get_random_trunk_radius(self) -> float:
            return random.uniform(TRUNK_RADIUS_MIN, TRUNK_RADIUS_MAX)

        def _poisson_sample(self, lam: float) -> int:
            return random.randint(ANOMALIES_MIN, ANOMALIES_MAX)

        def _get_random_pos(self, trunk_radius: float) -> Pos:
            # PERFECT spatial uniformity in a circle (area is proportional to r^2)
            r_normalized = math.sqrt(random.uniform(0.0, 1.0))
            r = trunk_radius * r_normalized
            phi = random.uniform(0, 2 * math.pi)
            return Pos(r=r, phi=phi)

        def _get_random_harmonic(
            self, trunk_radius: float, pos: Pos, max_attempts: int = 50
        ) -> Harmonic:
            r0 = random.uniform(ANOMALY_RADIUS_MIN, ANOMALY_RADIUS_MAX)
            r0 = max(0.01, min(r0, trunk_radius * 0.8))  # Sanity clamp
            degree = random.randint(1, self.max_harmonic_degree)

            for _ in range(max_attempts):
                harmonics = []
                for k in range(1, degree + 1):
                    sigma = (r0 * self.harmonic_sigma_base) / k
                    a = random.gauss(0, sigma)
                    b = random.gauss(0, sigma)
                    harmonics.append((a, b))

                shape = Harmonic(center=pos, base_radius=r0, harmonics=harmonics)
                if shape.is_valid(n_points=180):
                    return shape
            return Harmonic(center=pos, base_radius=r0, harmonics=[(0.0, 0.0)])

        def generate(self) -> Trunk:
            trunk_radius = self._get_random_trunk_radius()
            base_conductivity = random.uniform(COND_BASE_MIN, COND_BASE_MAX)

            n_anomalies = self._poisson_sample(0)
            anomalies = []
            for _ in range(n_anomalies):
                pos = self._get_random_pos(trunk_radius)
                shape = self._get_random_harmonic(trunk_radius, pos)
                cond = random.uniform(COND_ANOMALY_MIN, COND_ANOMALY_MAX)
                prop = random.choice(self.propagation_modes)
                anomalies.append(Anomaly(shape, cond, prop))

            return SimpleTrunk.create(
                radius=trunk_radius, base_conductivity=base_conductivity, anomalies=anomalies
            )

    factory = UniformMCFactory(propagation_modes=["Linear", "Cos"], max_harmonic_degree=5)
    return [factory.generate() for _ in range(n)]


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def save_sample(result: ForwardResult, i: int, base_path: Path) -> None:
    """Persist all outputs for one sample."""
    # 1. Targets (Datos crudos para Machine Learning)
    mask = generate_mask(result.trunk, resolution=result.grid.shape[0])
    np.save(base_path / f"targets/grid/{i}.npy", result.grid)
    np.save(base_path / f"targets/mask/{i}.npy", mask)
    np.save(base_path / f"targets/elem_data/{i}.npy", result.eit_result.elem_data)

    # 2. Metadata
    with open(base_path / f"metadata/{i}.json", "w", encoding="utf-8") as fh:
        json.dump(result.trunk.to_dict(), fh, indent=2)

    # 3. Visualizations (Imagen a color para humanos)
    valid_pixels = result.grid[mask > 0]
    g_min = valid_pixels.min() if len(valid_pixels) > 0 else 0.0
    g_max = valid_pixels.max() if len(valid_pixels) > 0 else 1.0
    vmin_adj = g_min - (g_max - g_min) * 0.8
    norm = plt.Normalize(vmin=vmin_adj, vmax=g_max)
    rgba_image = plt.get_cmap("viridis")(norm(result.grid))
    rgba_image[mask == 0] = [1.0, 1.0, 1.0, 1.0]
    plt.imsave(base_path / f"visualizations/grid/{i}.png", rgba_image, origin="lower")


def save_mesh_once(result: ForwardResult, base_path: Path) -> None:
    """Save the shared FEM mesh."""
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
    parser = argparse.ArgumentParser(description="Generador de Datasets EIT")
    parser.add_argument("--samples", type=int, default=10220, help="Número de muestras a generar")
    parser.add_argument(
        "--electrodes",
        type=str,
        default="16",
        choices=["8", "16", "32", "all"],
        help="Número de electrodos",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="all",
        choices=["adjacent", "opposite", "all"],
        help="Patrón de estimulación",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    patterns = ["adjacent", "opposite"] if args.pattern == "all" else [args.pattern]
    electrode_list = [8, 16, 32] if args.electrodes == "all" else [int(args.electrodes)]

    GRID_RESOLUTION = 64
    base_paths = {}
    mesh_saved = dict.fromkeys(electrode_list, False)

    # Preparamos la nueva arquitectura de carpetas
    for e in electrode_list:
        bp = PROJECT_ROOT / "dataset" / f"dataset_{e}e_{args.pattern}"

        # Mesh
        (bp / "mesh").mkdir(parents=True, exist_ok=True)
        # Targets
        for sub in ("grid", "mask", "elem_data"):
            (bp / f"targets/{sub}").mkdir(parents=True, exist_ok=True)
        # Inputs
        for p in patterns:
            (bp / f"inputs/voltages_{p}").mkdir(parents=True, exist_ok=True)
        # Metadata & Visualizations
        (bp / "metadata").mkdir(parents=True, exist_ok=True)
        (bp / "visualizations/grid").mkdir(parents=True, exist_ok=True)

        base_paths[e] = bp

    print(
        f"🌲 Generando {args.samples} troncos para {args.electrodes} electrodos ({args.pattern})..."
    )
    dataset = get_dataset(args.samples)
    failed = 0

    for i, trunk in enumerate(dataset):
        print(f"  [{i + 1}/{args.samples}] Simulando...", end=" ", flush=True)
        trunk_failed = False

        for e in electrode_list:
            bp = base_paths[e]
            results = {}
            for p in patterns:
                res = simulate_forward_process(
                    trunk, resolution=GRID_RESOLUTION, n_electrodes=e, pattern=p
                )
                if res is None:
                    break
                results[p] = res

            if len(results) != len(patterns):
                trunk_failed = True
                break

            first_res = results[patterns[0]]

            if not mesh_saved[e]:
                save_mesh_once(first_res, bp)
                mesh_saved[e] = True

            # Guarda shared targets, metadata y visualizations (incluye UN elem_data del primer resultado)
            save_sample(first_res, i, bp)

            # Guarda EXCLUSIVAMENTE los voltajes por patrón
            for p in patterns:
                np.save(bp / f"inputs/voltages_{p}/{i}.npy", results[p].eit_result.voltages)

        if trunk_failed:
            print("FAILED — saltando")
            failed += 1
        else:
            print("✓")

    ok = args.samples - failed
    print(f"\n✅ Terminado: {ok}/{args.samples} muestras generadas con éxito.")
