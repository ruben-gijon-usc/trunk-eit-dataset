"""
Example: Run EIT simulation for tree trunk with anomalies.

Usage:
    uv run example.py

Output:
    out/<timestamp>/ - Contains all simulation results
"""

import os
import json
import math
from datetime import datetime
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.patches import Circle as MplCircle, Polygon
from matplotlib.collections import PolyCollection

from src.models import Trunk, Anomaly, Circle
from src.eidors.bridge import run_eidors_simulation
from src.simulation import generate_grid, grid2png


OUTPUT_DIR = "out"


def create_output_folder() -> str:
    """Create output folder with timestamp."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(folder, exist_ok=True)
    return folder


def save_json(data: dict, filepath: str) -> None:
    """Save data as JSON."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def save_readable_grid(grid: np.ndarray, filepath: str) -> None:
    """Save grid as readable text format."""
    header = f"# Conductivity Grid\n# Shape: {grid.shape}\n# Values: S/m\n"
    with open(filepath, 'w') as f:
        f.write(header)
        for row in grid:
            f.write(' '.join(f'{v:.6f}' for v in row) + '\n')


def save_mesh_readable(nodes: np.ndarray, elems: np.ndarray, folder: str) -> None:
    """Save mesh data in readable formats."""
    nodes_path = os.path.join(folder, "mesh_nodes.csv")
    with open(nodes_path, 'w') as f:
        f.write("id,x,y\n")
        for i, (x, y) in enumerate(nodes):
            f.write(f"{i},{x:.6f},{y:.6f}\n")

    elems_path = os.path.join(folder, "mesh_elements.csv")
    with open(elems_path, 'w') as f:
        f.write("id,node1,node2,node3\n")
        for i, (n1, n2, n3) in enumerate(elems):
            f.write(f"{i},{n1},{n2},{n3}\n")


def save_voltages_readable(voltages: np.ndarray, filepath: str) -> None:
    """Save voltages as readable text."""
    with open(filepath, 'w') as f:
        f.write(f"# Voltage Measurements\n# Count: {len(voltages)}\n# Unit: V\n\n")
        for i, v in enumerate(voltages):
            f.write(f"{i:03d}: {v:.6f}\n")


def save_conductivity_map(elem_data: np.ndarray, filepath: str) -> None:
    """Save element conductivity as readable text."""
    with open(filepath, 'w') as f:
        f.write(f"# Element Conductivity Map\n# Elements: {len(elem_data)}\n# Unit: S/m\n\n")
        for i, c in enumerate(elem_data):
            f.write(f"{i:04d}: {c:.6f}\n")


def get_electrode_positions(n_elec: int, radius: float = 1.0) -> np.ndarray:
    """Get electrode positions around the boundary."""
    angles = np.linspace(0, 2 * np.pi, n_elec, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def get_injection_pattern(n_elec: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """Get adjacent injection pattern (electrode pairs for current injection).

    Returns:
        injection_pairs: Array of (source_elec, sink_elec) pairs
        measurement_pairs: Array of (meas_positive, meas_negative) pairs
    """
    injection_pairs = []
    measurement_pairs = []

    for inj_idx in range(n_elec):
        source = inj_idx
        sink = (inj_idx + 1) % n_elec
        injection_pairs.append((source, sink))

        for meas_idx in range(n_elec):
            if meas_idx == source or meas_idx == sink:
                continue
            for m2_idx in range(meas_idx + 1, n_elec):
                if m2_idx == source or m2_idx == sink:
                    continue
                measurement_pairs.append((meas_idx, m2_idx))

    return np.array(injection_pairs), np.array(measurement_pairs)


def plot_mesh_visualization(
    nodes: np.ndarray,
    elems: np.ndarray,
    elem_data: np.ndarray,
    output_folder: str
) -> None:
    """Create mesh visualization with conductivity."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Mesh wireframe
    ax1 = axes[0]
    ax1.triplot(nodes[:, 0], nodes[:, 1], elems - 1, color='gray', linewidth=0.3, alpha=0.5)
    ax1.set_title('FEM Mesh (Wireframe)')
    ax1.set_aspect('equal')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.grid(True, alpha=0.3)

    # 2. Mesh with conductivity heatmap
    ax2 = axes[1]

    triang = Triangulation(nodes[:, 0], nodes[:, 1], elems - 1)
    tcf = ax2.tripcolor(triang, elem_data, cmap='viridis', shading='flat')
    ax2.set_title('FEM Mesh (Conductivity)')
    ax2.set_aspect('equal')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    plt.colorbar(tcf, ax=ax2, label='Conductivity (S/m)')

    # 3. Mesh with electrodes
    ax3 = axes[2]
    ax3.triplot(nodes[:, 0], nodes[:, 1], elems - 1, color='lightgray', linewidth=0.2, alpha=0.3)

    n_elec = 16
    elec_pos = get_electrode_positions(n_elec, radius=1.0)
    for i, (x, y) in enumerate(elec_pos):
        circle = MplCircle((x, y), 0.06, color='red', alpha=0.7)
        ax3.add_patch(circle)
        ax3.annotate(str(i + 1), (x, y), ha='center', va='center', fontsize=8, color='white', fontweight='bold')

    ax3.set_title('FEM Mesh (Electrodes)')
    ax3.set_aspect('equal')
    ax3.set_xlim(-1.3, 1.3)
    ax3.set_ylim(-1.3, 1.3)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "output_mesh_visualization.png"), dpi=150)
    plt.close()


def plot_electrode_voltages(
    voltages: np.ndarray,
    n_elec: int,
    output_folder: str
) -> None:
    """Plot electrode voltages - input (injection) and output (measurement)."""
    inj_pairs, meas_pairs = get_injection_pattern(n_elec)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Injection pattern visualization
    ax1 = axes[0, 0]
    elec_pos = get_electrode_positions(n_elec, radius=1.0)

    for i, (src, sink) in enumerate(inj_pairs):
        ax1.plot([elec_pos[src, 0], elec_pos[sink, 0]],
                 [elec_pos[src, 1], elec_pos[sink, 1]],
                 'r-', linewidth=2, alpha=0.7)

    for i, (x, y) in enumerate(elec_pos):
        circle = MplCircle((x, y), 0.05, color='blue', alpha=0.8)
        ax1.add_patch(circle)
        ax1.annotate(str(i + 1), (x, y), ha='center', va='center', fontsize=9, color='white')

    ax1.set_title(f'Current Injection Pattern (Adjacent)\n{n_elec} electrodes, {len(inj_pairs)} patterns')
    ax1.set_aspect('equal')
    ax1.set_xlim(-1.3, 1.3)
    ax1.set_ylim(-1.3, 1.3)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')

    # 2. Input voltage per injection (contact voltage)
    ax2 = axes[0, 1]
    input_voltages = []
    pattern_labels = []
    for i, (src, sink) in enumerate(inj_pairs):
        v_src = voltages[i * (n_elec - 2) * (n_elec - 3) // 2] if i * (n_elec - 2) * (n_elec - 3) // 2 < len(voltages) else 0
        input_voltages.append(v_src)
        pattern_labels.append(f'{src+1}-{sink+1}')

    ax2.bar(range(len(input_voltages)), input_voltages, color='blue', alpha=0.7)
    ax2.set_xlabel('Injection Pattern')
    ax2.set_ylabel('Contact Voltage (V)')
    ax2.set_title('Input Contact Voltages\n(Voltage at injection electrodes)')
    ax2.set_xticks(range(len(input_voltages)))
    ax2.set_xticklabels(pattern_labels, rotation=45, ha='right', fontsize=7)
    ax2.grid(True, alpha=0.3)

    # 3. Output voltages distribution
    ax3 = axes[1, 0]
    ax3.plot(range(len(voltages)), voltages, 'b-', alpha=0.7, linewidth=0.8)
    ax3.axhline(y=voltages.mean(), color='red', linestyle='--', label=f'Mean: {voltages.mean():.2f} V')
    ax3.set_xlabel('Measurement Index')
    ax3.set_ylabel('Voltage (V)')
    ax3.set_title(f'Output Measured Voltages\n{len(voltages)} measurements')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Voltage histogram
    ax4 = axes[1, 1]
    ax4.hist(voltages, bins=30, color='blue', alpha=0.7, edgecolor='black')
    ax4.axvline(x=voltages.mean(), color='red', linestyle='--', label=f'Mean: {voltages.mean():.2f} V')
    ax4.set_xlabel('Voltage (V)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Output Voltage Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "output_electrode_voltages.png"), dpi=150)
    plt.close()


def save_electrode_info(n_elec: int, output_folder: str) -> dict:
    """Save electrode and injection pattern info."""
    inj_pairs, meas_pairs = get_injection_pattern(n_elec)
    elec_pos = get_electrode_positions(n_elec)

    info = {
        "n_electrodes": n_elec,
        "electrode_positions": [
            {"id": i, "x": float(elec_pos[i, 0]), "y": float(elec_pos[i, 1])}
            for i in range(n_elec)
        ],
        "injection_pattern": {
            "type": "adjacent",
            "n_patterns": len(inj_pairs),
            "pairs": [
                {"source": int(src), "sink": int(sink)}
                for src, sink in inj_pairs
            ]
        },
        "measurement_pattern": {
            "type": "adjacent_all",
            "n_measurements_per_pattern": len(meas_pairs),
            "total_measurements": len(meas_pairs) * len(inj_pairs)
        }
    }

    save_json(info, os.path.join(output_folder, "input_electrode_config.json"))
    return info


def visualize_grid(grid: np.ndarray, output_folder: str) -> None:
    """Create visualization of the conductivity grid."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    im1 = ax1.imshow(grid, cmap='viridis', origin='lower')
    ax1.set_title('Conductivity Grid (Full View)')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    fig.colorbar(im1, ax=ax1, label='Conductivity (S/m)')

    ax2 = axes[1]
    center = grid.shape[0] // 2
    slice_data = grid[center, :]
    ax2.plot(range(len(slice_data)), slice_data)
    ax2.set_title('Conductivity Slice (Center Y)')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Conductivity (S/m)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "output_conductivity_visualization.png"), dpi=150)
    plt.close()


def save_summary(data: dict, filepath: str) -> None:
    """Save simulation summary as JSON."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def main():
    print("=== EIT Trunk Simulation ===\n")

    output_folder = create_output_folder()
    print(f"Output folder: {output_folder}\n")

    # ==================== INPUTS ====================
    print("[INPUTS]")

    trunk = Trunk(
        radius=1.0,
        base_conductivity=0.1,
        anomalies=[
            Anomaly(
                shape=Circle(cx=0.3, cy=0.0, radius=0.15),
                conductivity=0.5
            ),
            Anomaly(
                shape=Circle(cx=-0.2, cy=0.0, radius=0.1),
                conductivity=0.8
            )
        ]
    )

    print(f"  Trunk radius: {trunk.radius} m")
    print(f"  Base conductivity: {trunk.base_conductivity} S/m")
    print(f"  Anomalies: {len(trunk.anomalies)}")

    # Save trunk config
    trunk_config = {
        "radius_m": trunk.radius,
        "base_conductivity_S_m": trunk.base_conductivity,
        "anomalies": [
            {
                "shape_type": a.shape.shape_type,
                "conductivity_S_m": a.conductivity,
                **a.shape.to_dict()
            }
            for a in trunk.anomalies
        ]
    }
    save_json(trunk_config, os.path.join(output_folder, "input_trunk_config.json"))

    # Simulation parameters
    n_elec = 16
    sim_params = {
        "grid_resolution": 64,
        "n_electrodes": n_elec,
        "injection_method": "adjacent"
    }
    save_json(sim_params, os.path.join(output_folder, "input_simulation_params.json"))

    # Save electrode configuration
    electrode_info = save_electrode_info(n_elec, output_folder)

    print(f"  Saved: input_trunk_config.json, input_simulation_params.json, input_electrode_config.json\n")

    # ==================== GRID SIMULATION ====================
    print("[GRID SIMULATION]")

    grid = generate_grid(trunk, resolution=64)
    print(f"  Grid shape: {grid.shape}")
    print(f"  Unique conductivities: {len(set(grid[grid > 0]))}")

    save_readable_grid(grid, os.path.join(output_folder, "output_conductivity_grid.txt"))
    np.save(os.path.join(output_folder, "output_conductivity_grid.npy"), grid)
    grid2png(grid, os.path.join(output_folder, "output_conductivity_grid.png"))
    visualize_grid(grid, output_folder)

    print("  Saved: output_conductivity_grid.*\n")

    # ==================== EIDORS SIMULATION ====================
    print("[EIDORS SIMULATION]")

    result = run_eidors_simulation(trunk, n_electrodes=n_elec)
    print(f"  Voltages: {result.voltages.shape}")
    print(f"  Mesh nodes: {result.mesh.nodes.shape}")
    print(f"  Mesh elements: {result.mesh.elems.shape}")
    print(f"  Unique elem conductivities: {len(set(result.elem_data))}")

    # Save EIDORS outputs
    save_voltages_readable(result.voltages, os.path.join(output_folder, "output_eidors_voltages.txt"))
    save_mesh_readable(result.mesh.nodes, result.mesh.elems, output_folder)
    save_conductivity_map(result.elem_data, os.path.join(output_folder, "output_elem_conductivity.txt"))

    np.save(os.path.join(output_folder, "output_eidors_voltages.npy"), result.voltages)
    np.save(os.path.join(output_folder, "output_mesh_nodes.npy"), result.mesh.nodes)
    np.save(os.path.join(output_folder, "output_mesh_elems.npy"), result.mesh.elems)
    np.save(os.path.join(output_folder, "output_elem_conductivity.npy"), result.elem_data)

    # Visualizations
    plot_mesh_visualization(result.mesh.nodes, result.mesh.elems, result.elem_data, output_folder)
    plot_electrode_voltages(result.voltages, n_elec, output_folder)

    print("  Saved: output_eidors_*, output_mesh_*, output_elem_conductivity.*, *_visualization.png\n")

    # ==================== SUMMARY ====================
    print("[SUMMARY]")

    # Calculate contact voltages (input)
    inj_pairs, _ = get_injection_pattern(n_elec)
    n_meas_per_pattern = (n_elec - 2) * (n_elec - 3) // 2

    summary = {
        "timestamp": datetime.now().isoformat(),
        "grid": {
            "resolution": grid.shape[0],
            "unique_conductivities": len(set(grid[grid > 0])),
            "min_conductivity": float(grid[grid > 0].min()),
            "max_conductivity": float(grid[grid > 0].max())
        },
        "eidors": {
            "n_electrodes": n_elec,
            "n_measurements": len(result.voltages),
            "n_mesh_nodes": len(result.mesh.nodes),
            "n_mesh_elements": len(result.mesh.elems),
            "voltage_mean": float(result.voltages.mean()),
            "voltage_std": float(result.voltages.std()),
            "voltage_min": float(result.voltages.min()),
            "voltage_max": float(result.voltages.max()),
            "unique_conductivities": len(set(result.elem_data))
        },
        "electrodes": {
            "n_electrodes": n_elec,
            "injection_patterns": len(inj_pairs),
            "measurements_per_pattern": n_meas_per_pattern
        },
        "files": {
            "inputs": ["input_trunk_config.json", "input_simulation_params.json", "input_electrode_config.json"],
            "grid": ["output_conductivity_grid.npy", "output_conductivity_grid.png", "output_conductivity_grid.txt", "output_conductivity_visualization.png"],
            "mesh": ["output_mesh_nodes.npy", "output_mesh_elems.npy", "output_mesh_nodes.csv", "output_mesh_elements.csv", "output_mesh_visualization.png"],
            "eidors": ["output_eidors_voltages.npy", "output_eidors_voltages.txt", "output_elem_conductivity.npy", "output_elem_conductivity.txt"],
            "electrodes": ["output_electrode_voltages.png"]
        }
    }
    save_summary(summary, os.path.join(output_folder, "summary.json"))

    print(f"  Saved: summary.json\n")
    print(f"Output saved to: {output_folder}")
    print("\nDone!")


if __name__ == "__main__":
    main()