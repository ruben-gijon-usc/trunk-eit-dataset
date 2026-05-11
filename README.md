# Trunk EIT Dataset

Automated pipeline for generating ML datasets for Electrical Impedance Tomography (EIT) applied to tree trunks. Translates geometric models into conductivity grids and runs forward simulations.

**Tech Stack**: Python 3.10+, `uv` (package manager), EIDORS (EIT simulation via Octave)

---

## Quick Start

```bash
uv sync                 # Install dependencies
```

---

## Project Structure

```
src/
├── base.py             # Domain models (Trunk, Anomaly, Pos)
├── grid.py             # Grid generation (N x N matrices)
├── eit_simulation.py   # Analytical forward solver
├── pipeline.py         # Dataset orchestration
└── eidors/
    ├── bridge.py       # Python wrapper for EIDORS
    ├── simulate.m     # Complete EIDORS simulation
    ├── create_model.m # Create circular FEM model
    └── forward_solve.m # Run forward solve

tests/
├── test_pipeline.py
└── test_eidors.py
```

---

## Commands

### Development

```bash
uv sync                 # Install dependencies
uv sync --dev           # Install dev dependencies (ruff)
uv run ruff check .     # Lint code
uv run ruff format .    # Format code
```

### Testing

```bash
uv run pytest                 # Run all tests
uv run pytest -v              # Run with verbose output
uv run pytest tests/          # Run specific test directory
uv run pytest tests/test_eidors.py -v  # Run specific test file
```

### Generate Dataset

```bash
# Generate dataset with grid-only (no EIDORS)
uv run python -c "
from src.pipeline import run_pipeline, PipelineConfig
config = PipelineConfig(num_samples=10, use_eidors=False)
run_pipeline('dataset.npz', config, 'numpy')
"

# Generate dataset with EIDORS (grid + mesh)
uv run python -c "
from src.pipeline import run_pipeline, PipelineConfig
config = PipelineConfig(num_samples=10, use_eidors=True)
run_pipeline('dataset.h5', config, 'hdf5')
"
```

---

## Utilities

### Grid Generation (N x N matrices)

```python
from src.base import Trunk, Anomaly, Pos
from src.grid import generate_grid, grid2png

# Create trunk model
trunk = Trunk(
    radius=1.0,
    base_conductivity=0.1,
    anomalies=[
        Anomaly(radius=0.15, center=Pos(r=0.3, phi=0.0), conductivity=0.5),
    ]
)

# Generate N x N conductivity grid
grid = generate_grid(trunk, resolution=128)

# Save as PNG
grid2png(grid, 'conductivity.png')
```

### EIDORS Forward Simulation (Mesh-based FEM)

```python
from src.eidors.bridge import run_eidors_simulation
from src.base import Trunk, Anomaly, Pos

trunk = Trunk(
    radius=1.0,
    base_conductivity=1.0,
    anomalies=[
        Anomaly(radius=0.15, center=Pos(r=0.3, phi=0.0), conductivity=0.5),
    ]
)

result = run_eidors_simulation(trunk, n_electrodes=16)

print(f"Voltages: {result.voltages.shape}")  # (208,)
print(f"Nodes: {result.mesh.nodes.shape}")  # (1564, 2)
print(f"Elements: {result.mesh.elems.shape}")  # (2943, 3)
print(f"Elem data: {result.elem_data.shape}")  # (2943,)
```

### Analytical Forward Solver

```python
from src.eit_simulation import eit_simulation, EITProtocol
from src.grid import generate_grid
from src.base import Trunk

trunk = Trunk(radius=1.0, base_conductivity=0.1, anomalies=[])
grid = generate_grid(trunk, resolution=64)

protocol = EITProtocol(num_electrodes=16, injection_method="adjacent")
voltages = eit_simulation(grid, protocol, trunk_radius=1.0)

print(f"Voltages: {voltages.shape}")  # (1456,)
```

### Pipeline Configuration

```python
from src.pipeline import PipelineConfig, run_pipeline

config = PipelineConfig(
    num_samples=1000,         # Number of samples
    grid_resolution=128,       # Grid resolution
    trunk_radius=1.0,         # Trunk radius (m)
    base_conductivity=0.1,    # Base conductivity (S/m)
    anomaly_conductivity_range=(0.01, 0.5),  # Random anomaly conductivity range
    num_anomalies_range=(1, 3),   # Random number of anomalies
    anomaly_radius_range=(0.1, 0.4),  # Random anomaly radius range
    use_eidors=True,          # Include EIDORS mesh data
    n_electrodes=16,          # Number of electrodes
)

# Run pipeline
samples = run_pipeline('output.h5', config, 'hdf5')
```

---

## Testing Examples

### Run all tests
```bash
uv run pytest -v
```

### Run specific test file
```bash
uv run pytest tests/test_eidors.py -v
```

### Run specific test function
```bash
uv run pytest tests/test_eidors.py::TestEIDORSBridge::test_run_eidors_simulation_with_anomaly -v
```

### Run with coverage
```bash
uv run pytest --cov=src --cov-report=term-missing
```

---

## Output Formats

| Format | Description | Extension |
|--------|-------------|-----------|
| `numpy` | NumPy archive with arrays | `.npz` |
| `hdf5` | HDF5 file with all data | `.h5` / `.hdf5` |
| `json` | JSON with all data | `.json` |

### Dataset Structure

```python
{
    'voltages': np.array,           # Grid voltages (N, 1456)
    'conductivity_maps': np.array,  # Grid conductivity (N, 128, 128)
    'mesh_voltages': np.array,      # EIDORS voltages (N, 208)
    'mesh_elem_data': np.array,     # Element conductivities (N, 2943)
    'mesh_nodes': np.array,         # Mesh nodes (N, 1564, 2)
    'mesh_elems': np.array,         # Mesh elements (N, 2943, 3)
}
```