import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.stochastic.generate_harmonic import StochasticTrunkFactory
from src.data_representations.grid import generate_grid
from src.models import Trunk

def get_dataset(n: int) -> list:
    healthy_factory = StochasticTrunkFactory(
        trunk_radius_mu=0.3, 
        expected_anomalies=1.5,
        pos_alpha=1.0, pos_beta=3.0 # Biased toward the center
    )

    decay_factory = StochasticTrunkFactory(
        trunk_radius_mu=1.2, 
        trunk_radius_sigma=0.3,
        expected_anomalies=8.0, 
        anomaly_scale=0.15,
        pos_alpha=3.0, pos_beta=1.0 # Biased toward the bark
    )

    dataset = [decay_factory.generate() for _ in range(n // 2)]
    dataset += [healthy_factory.generate() for _ in range(n // 2)]
    return dataset


def save_json(data: dict, path: Path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)


def save_grid(grid: np.ndarray, png_path: Path):
    plt.imsave(png_path, grid, cmap='viridis')


def save(trunk: Trunk, i: int, base_path: Path):
    grid = generate_grid(trunk)
    data = trunk.to_dict()

    save_grid(grid, base_path / f"grid/{i}.png")
    save_json(data, base_path / f"json/{i}.json")


if __name__ == "__main__":
    N_SAMPLES = 5_000
    
    # Setup directory structure automatically
    base_path = Path("dataset")
    (base_path / "grid").mkdir(parents=True, exist_ok=True)
    (base_path / "json").mkdir(parents=True, exist_ok=True)
    
    print(f"🌲 Generating dataset of {N_SAMPLES} trees...")
    dataset = get_dataset(N_SAMPLES)
    
    print("💾 Saving to disk...")
    for i, trunk in enumerate(dataset):
        data_trunk = trunk.to_dict()
        save_json(data_trunk, base_path / f"json/{i}.json")
        
        # Print progress every 10 iterations
        if (i + 1) % 10 == 0:
            print(f"Saved {i + 1}/{N_SAMPLES}...")
            
    print("✅ Dataset generation complete!")
