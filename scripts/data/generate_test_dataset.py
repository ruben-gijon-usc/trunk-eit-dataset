import argparse
import json
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.models.harmonics import Harmonic, Pos

# Importamos la configuración original para mantener los límites físicos (S/m) idénticos
from scripts.data.config import COND_ANOMALY_MAX, COND_ANOMALY_MIN, COND_BASE_MAX, COND_BASE_MIN

# Importamos el generador original para el Test In-Distribution
from scripts.data.generate_dataset import get_dataset as get_id_dataset
from src.data.data_representations.grid import generate_mask
from src.data.forward_process.simulator import ForwardResult, simulate_forward_process
from src.data.models.trunk import Anomaly, SimpleTrunk, Trunk
from src.data.stochastic.generate_harmonic import StochasticTrunkFactory

# ---------------------------------------------------------------------------
# Out-Of-Distribution (OOD) Dataset Factory
# ---------------------------------------------------------------------------
# Mantenemos las conductividades idénticas al entrenamiento, pero forzamos
# que la topología (forma geométrica, cantidad y tamaño) sea inédita.

OOD_ANOMALIES_MIN = 3
OOD_ANOMALIES_MAX = 4
OOD_ANOMALY_RADIUS_MIN = 0.05
OOD_ANOMALY_RADIUS_MAX = 0.15  # Anomalías más pequeñas pero más numerosas
OOD_MAX_HARMONIC = 7  # Geometrías más puntiagudas y complejas


def get_ood_dataset(n: int) -> list[Trunk]:
    class OODFactory(StochasticTrunkFactory):
        def _get_random_trunk_radius(self) -> float:
            return 1.0

        def _poisson_sample(self, lam: float) -> int:
            return random.randint(OOD_ANOMALIES_MIN, OOD_ANOMALIES_MAX)

        def _get_random_pos(self, trunk_radius: float) -> Pos:
            # Posiciones extremas (muy en el centro o pegadas a la corteza)
            r_normalized = random.choice([random.uniform(0.0, 0.2), random.uniform(0.75, 0.9)])
            r = trunk_radius * r_normalized
            phi = random.uniform(0, 2 * math.pi)
            return Pos(r=r, phi=phi)

        def _get_random_harmonic(
            self, trunk_radius: float, pos: Pos, max_attempts: int = 50
        ) -> Harmonic:
            r0 = random.uniform(OOD_ANOMALY_RADIUS_MIN, OOD_ANOMALY_RADIUS_MAX)
            r0 = max(0.01, min(r0, trunk_radius * 0.8))
            degree = random.randint(3, self.max_harmonic_degree)

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

    factory = OODFactory(propagation_modes=["Linear", "Cos"], max_harmonic_degree=OOD_MAX_HARMONIC)
    return [factory.generate() for _ in range(n)]


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------
def save_sample(result: ForwardResult, i: int, base_path: Path) -> None:
    mask = generate_mask(result.trunk, resolution=result.grid.shape[0])
    np.save(base_path / f"targets/grid/sample_{i:04d}.npy", result.grid)
    np.save(base_path / f"targets/mask/sample_{i:04d}.npy", mask)
    np.save(base_path / f"targets/elem_data/sample_{i:04d}.npy", result.eit_result.elem_data)
    with open(base_path / f"metadata/sample_{i:04d}.json", "w", encoding="utf-8") as fh:
        json.dump(result.trunk.to_dict(), fh, indent=2)
    valid_pixels = result.grid[mask > 0]
    g_min = valid_pixels.min() if len(valid_pixels) > 0 else 0.0
    g_max = valid_pixels.max() if len(valid_pixels) > 0 else 1.0
    vmin_adj = g_min - (g_max - g_min) * 0.8
    norm = plt.Normalize(vmin=vmin_adj, vmax=g_max)
    rgba_image = plt.get_cmap("viridis")(norm(result.grid))
    rgba_image[mask == 0] = [1.0, 1.0, 1.0, 1.0]
    plt.imsave(base_path / f"visualizations/grid/sample_{i:04d}.png", rgba_image, origin="lower")


def save_mesh_once(result: ForwardResult, base_path: Path) -> None:
    np.save(base_path / "mesh/nodes.npy", result.eit_result.mesh.nodes)
    np.save(base_path / "mesh/elems.npy", result.eit_result.mesh.elems)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de Datasets de TESTEO EIT (ID / OOD)")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["id", "ood"],
        help="Modo: 'id' (misma distribución) o 'ood' (topología compleja)",
    )
    parser.add_argument("--samples", type=int, default=1000, help="Número de muestras a generar")
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
        default="adjacent",
        choices=["adjacent", "opposite", "all"],
        help="Patrón de estimulación",
    )
    parser.add_argument(
        "--seed", type=int, default=999, help="Semilla aleatoria (distinta al entrenamiento)"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    patterns = ["adjacent", "opposite"] if args.pattern == "all" else [args.pattern]
    electrode_list = [8, 16, 32] if args.electrodes == "all" else [int(args.electrodes)]

    GRID_RESOLUTION = 64
    base_paths = {}
    mesh_saved = dict.fromkeys(electrode_list, False)

    # Creación de carpetas con sufijo dinámico
    suffix = "test_id" if args.mode == "id" else "test_ood"
    for e in electrode_list:
        bp = PROJECT_ROOT / "dataset" / f"dataset_{e}e_{args.pattern}_{suffix}"
        (bp / "mesh").mkdir(parents=True, exist_ok=True)
        for sub in ("grid", "mask", "elem_data"):
            (bp / f"targets/{sub}").mkdir(parents=True, exist_ok=True)
        for p in patterns:
            (bp / f"inputs/voltages_{p}").mkdir(parents=True, exist_ok=True)
        (bp / "metadata").mkdir(parents=True, exist_ok=True)
        (bp / "visualizations/grid").mkdir(parents=True, exist_ok=True)
        base_paths[e] = bp

    print(
        f"🔬 Generando {args.samples} troncos de TEST ({args.mode.upper()}) para {args.electrodes} electrodos ({args.pattern})..."
    )

    if args.mode == "id":
        dataset = get_id_dataset(args.samples)
    else:
        dataset = get_ood_dataset(args.samples)

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

            save_sample(first_res, i, bp)
            for p in patterns:
                np.save(
                    bp / f"inputs/voltages_{p}/sample_{i:04d}.npy", results[p].eit_result.voltages
                )

        if trunk_failed:
            print("FAILED — saltando")
            failed += 1
        else:
            print("✓")

    ok = args.samples - failed
    print(
        f"\n✅ Terminado Test Dataset ({args.mode.upper()}): {ok}/{args.samples} muestras generadas con éxito."
    )
