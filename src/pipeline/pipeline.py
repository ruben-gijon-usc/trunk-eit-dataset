from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..models import Anomaly, Circle, Pos, Trunk
from ..eidors.bridge import run_eidors_simulation
from ..simulation import generate_grid


@dataclass
class DatasetSample:
    voltages: NDArray[np.float64]
    conductivity_map: NDArray[np.float64]
    trunk_config: Trunk
    mesh_voltages: NDArray[np.float64] | None = None
    mesh_elem_data: NDArray[np.float64] | None = None
    mesh_nodes: NDArray[np.float64] | None = None
    mesh_elems: NDArray[np.int32] | None = None


@dataclass
class PipelineConfig:
    num_samples: int = 1000
    grid_resolution: int = 128
    trunk_radius: float = 1.0
    base_conductivity: float = 0.1
    anomaly_conductivity_range: tuple = (0.01, 0.5)
    num_anomalies_range: tuple = (1, 3)
    anomaly_radius_range: tuple = (0.1, 0.4)
    use_eidors: bool = True
    n_electrodes: int = 16


def generate_random_trunk(config: PipelineConfig) -> Trunk:
    """
    Generate a random trunk configuration with random anomalies.

    Args:
        config: Pipeline configuration.

    Returns:
        Randomly configured Trunk object.
    """
    num_anomalies = np.random.randint(
        config.num_anomalies_range[0], config.num_anomalies_range[1] + 1
    )

    anomalies = []
    for _ in range(num_anomalies):
        anomaly_radius = np.random.uniform(
            config.anomaly_radius_range[0], config.anomaly_radius_range[1]
        )

        center_r = np.random.uniform(0, config.trunk_radius - anomaly_radius)
        center_phi = np.random.uniform(0, 2 * np.pi)

        conductivity = np.random.uniform(
            config.anomaly_conductivity_range[0], config.anomaly_conductivity_range[1]
        )

        cx = center_r * np.cos(center_phi)
        cy = center_r * np.sin(center_phi)

        anomalies.append(
            Anomaly(
                shape=Circle(center=Pos(r=center_r, phi=center_phi), radius=anomaly_radius),
                conductivity=conductivity,
            )
        )

    return Trunk(
        radius=config.trunk_radius, base_conductivity=config.base_conductivity, anomalies=anomalies
    )


def generate_dataset(config: PipelineConfig) -> list[DatasetSample]:
    """
    Generate a dataset of EIT samples.

    Args:
        config: Pipeline configuration.

    Returns:
        List of dataset samples containing voltages and ground truth.
    """
    samples = []

    for i in range(config.num_samples):
        trunk = generate_random_trunk(config)
        grid = generate_grid(trunk, config.grid_resolution)

        sample = DatasetSample(voltages=np.array([]), conductivity_map=grid, trunk_config=trunk)

        if config.use_eidors:
            try:
                eidors_result = run_eidors_simulation(trunk, n_electrodes=config.n_electrodes)
                if eidors_result:
                    sample.voltages = eidors_result.voltages
                    sample.mesh_voltages = eidors_result.voltages
                    sample.mesh_elem_data = eidors_result.elem_data
                    sample.mesh_nodes = eidors_result.mesh.nodes
                    sample.mesh_elems = eidors_result.mesh.elems
            except Exception as e:
                print(f"EIDORS failed for sample {i}: {e}")

        samples.append(sample)

        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{config.num_samples} samples")

    return samples


def save_dataset(samples: list[DatasetSample], output_path: str, format: str = "hdf5") -> None:
    """
    Save dataset to file.

    Args:
        samples: List of dataset samples.
        output_path: Output file path.
        format: Output format ("hdf5", "numpy", or "json").
    """
    if format == "hdf5":
        import h5py

        with h5py.File(output_path, "w") as f:
            f.create_dataset("voltages", data=[s.voltages for s in samples])
            f.create_dataset("conductivity_maps", data=[s.conductivity_map for s in samples])
            f.create_dataset("trunk_radii", data=[s.trunk_config.radius for s in samples])
            f.create_dataset(
                "base_conductivities", data=[s.trunk_config.base_conductivity for s in samples]
            )

            has_mesh = samples[0].mesh_voltages is not None
            if has_mesh:
                f.create_dataset(
                    "mesh_voltages",
                    data=[s.mesh_voltages for s in samples if s.mesh_voltages is not None],
                )
                f.create_dataset(
                    "mesh_elem_data",
                    data=[s.mesh_elem_data for s in samples if s.mesh_elem_data is not None],
                )
                max_nodes = max(len(s.mesh_nodes) for s in samples if s.mesh_nodes is not None)
                nodes_arr = np.zeros((len(samples), max_nodes, 3))
                for i, s in enumerate(samples):
                    if s.mesh_nodes is not None:
                        nodes_arr[i, : len(s.mesh_nodes)] = s.mesh_nodes
                f.create_dataset("mesh_nodes", data=nodes_arr)
                max_elems = max(len(s.mesh_elems) for s in samples if s.mesh_elems is not None)
                elems_arr = np.zeros((len(samples), max_elems, 3), dtype=np.int32)
                for i, s in enumerate(samples):
                    if s.mesh_elems is not None:
                        elems_arr[i, : len(s.mesh_elems)] = s.mesh_elems
                f.create_dataset("mesh_elems", data=elems_arr)

    elif format == "numpy":
        np.savez(
            output_path,
            voltages=np.array([s.voltages for s in samples]),
            conductivity_maps=np.array([s.conductivity_map for s in samples]),
        )

    elif format == "json":
        import json

        data = {
            "voltages": [s.voltages.tolist() for s in samples],
            "conductivity_maps": [s.conductivity_map.tolist() for s in samples],
        }
        has_mesh = samples[0].mesh_voltages is not None
        if has_mesh:
            data["mesh_voltages"] = [
                s.mesh_voltages.tolist() if s.mesh_voltages is not None else [] for s in samples
            ]
            data["mesh_elem_data"] = [
                s.mesh_elem_data.tolist() if s.mesh_elem_data is not None else [] for s in samples
            ]
            data["mesh_nodes"] = [
                s.mesh_nodes.tolist() if s.mesh_nodes is not None else [] for s in samples
            ]
            data["mesh_elems"] = [
                s.mesh_elems.tolist() if s.mesh_elems is not None else [] for s in samples
            ]
        with open(output_path, "w") as f:
            json.dump(data, f)

    else:
        raise ValueError(f"Unknown format: {format}")

    print(f"Saved {len(samples)} samples to {output_path}")


def run_pipeline(
    output_path: str, config: PipelineConfig | None = None, format: str = "hdf5"
) -> list[DatasetSample]:
    """
    Run the complete EIT dataset generation pipeline.

    Args:
        output_path: Output file path.
        config: Pipeline configuration (uses defaults if None).
        format: Output format.

    Returns:
        Generated dataset samples.
    """
    if config is None:
        config = PipelineConfig()

    print(f"Generating {config.num_samples} EIT samples...")
    samples = generate_dataset(config)
    save_dataset(samples, output_path, format)

    return samples
