"""
Example: Run EIT simulation for tree trunk with anomalies.

Usage:
    uv run example.py
"""

from src.base import Trunk, Anomaly, Pos
from src.eidors.bridge import run_eidors_simulation
from src.grid import generate_grid, grid2png


def main():
    print("=== EIT Trunk Simulation ===")

    # Create trunk model with anomaly
    trunk = Trunk(
        radius=1.0,
        base_conductivity=0.1,
        anomalies=[
            Anomaly(
                radius=0.15,
                center=Pos(r=0.3, phi=0.0),
                conductivity=0.5
            )
        ]
    )

    print(f"Trunk: radius={trunk.radius}m, base_conductivity={trunk.base_conductivity}S/m")
    print(f"Anomalies: {len(trunk.anomalies)}")

    # 1. Grid-based simulation (N x N matrix)
    print("\n[1] Grid-based simulation...")
    grid = generate_grid(trunk, resolution=64)
    print(f"    Grid shape: {grid.shape}")
    print(f"    Unique conductivities: {len(set(grid[grid > 0]))}")

    # Save grid visualization
    grid2png(grid, "conductivity_grid.png")
    print("    Saved: conductivity_grid.png")

    # 2. EIDORS simulation (FEM mesh)
    print("\n[2] EIDORS FEM simulation...")
    result = run_eidors_simulation(trunk, n_electrodes=16)
    print(f"    Voltages: {result.voltages.shape}")
    print(f"    Mesh nodes: {result.mesh.nodes.shape}")
    print(f"    Mesh elements: {result.mesh.elems.shape}")
    print(f"    Unique elem conductivities: {len(set(result.elem_data))}")

    # Summary
    print("\n=== Results ===")
    print(f"Grid resolution: {grid.shape}")
    print(f"EIDORS measurements: {len(result.voltages)}")
    print(f"FEM elements: {len(result.elem_data)}")
    print("\nDone!")


if __name__ == "__main__":
    main()