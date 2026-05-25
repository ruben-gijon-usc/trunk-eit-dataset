# 🌲 Trunk EIT Dataset - AI Agent Context & Rules

## 🎯 Project Goal
Generate ML datasets for Electrical Impedance Tomography (EIT) applied to tree trunks. Translates geometric anomaly models into conductivity grids, runs forward simulations (EIDORS via Octave), and packages data for ML training.

## 🛠️ Tech Stack
- **Language**: Python 3.10+
- **Package Manager**: `uv` — always use `uv run`, `uv sync`, `uv pip install`
- **Math**: `numpy` (vectorized grid operations)
- **EIT Simulation**: [EIDORS](http://eidors3d.sourceforge.net/) via Octave subprocess calls

## 🏗️ Architecture

Strict layer separation: **models → data_representations → simulation/eidors → pipeline**

### 1. `src/models/` — Domain Models (pure math, no grids)
- `Pos` — polar coords (`r`, `phi`), has `to_cartesian()`, `from_cartesian()`, `x`/`y` properties
- `Shape` (ABC) — `Circle`, `Ellipse`, `Harmonic`. All take `center: Pos` (not cx/cy kwargs)
- `Anomaly` — `shape + conductivity + propagation_fn` ("Constant"|"Linear"|"Cos")
- `Trunk` — `base: Anomaly` (the trunk body) + `anomalies: list[Anomaly]`
- `SimpleTrunk` — factory: `SimpleTrunk.create(radius, base_conductivity, anomalies=[])`
- `Serializable` (ABC) — `to_dict()` / `from_dict()` for JSON round-trips

### 2. `src/data_representations/grid.py` — Discretization
- `generate_grid(trunk, resolution)` → `NDArray[np.float64]`
- Uses `trunk.get_conductivity(pos)` with `np.vectorize` — respects physical aspect ratio
- `grid2png(grid, path)` — visualization helper

### 3. `src/simulation/grid.py` — Re-exports from `data_representations` for backward compat

### 4. `src/eidors/bridge.py` — EIDORS (Octave FEM)
- `run_eidors_simulation(trunk, n_electrodes=16)` → `EIDORSResult(voltages, mesh, elem_data)`
- Spawns Octave subprocess with temp dir; reads back `.txt` files
- `_anomalies_to_json()` flattens model shapes to `{shape: {cx, cy, radius, ...}, conductivity}` format expected by `simulate.m`
- EIDORS path: checks `vendor/eidors/`, `$EIDORS_PATH`, then hardcoded fallback

### 5. `src/pipeline/` — Dataset Orchestration
- `PipelineConfig` (dataclass) — num_samples, grid_resolution, trunk_radius, etc.
- `generate_random_trunk(config)` → uses `SimpleTrunk.create()`
- `run_pipeline(output_path, config, format)` → generates + saves (hdf5|numpy|json)

## 📝 Commands

```bash
uv sync                 # Install deps
uv sync --dev           # Install dev deps (ruff, pytest)
uv run pytest -v        # Run all tests
uv run pytest tests/test_pipeline.py -v  # Single test file
uv run ruff check .     # Lint
uv run ruff format .    # Format
```

## ⚠️ Agent Pitfalls

1. **`Trunk` API changed**: It now takes `base: Anomaly` + `anomalies: list[Anomaly]`, NOT `radius/base_conductivity` kwargs. Use `SimpleTrunk.create(radius, base_conductivity, anomalies=[])` for simple cases.
2. **`Circle` takes `center: Pos`**, not `cx`/`cy` keyword args: `Circle(center=Pos(r=0.3, phi=0.0), radius=0.15)`
3. **EIDORS JSON format**: The Octave `simulate.m` expects nested `{shape: {cx, cy, ...}, conductivity}` — the bridge handles this conversion. Don't pass raw `to_dict()` output.
4. **Anomaly has `propagation_fn`**: Third param controls conductivity falloff ("Constant" (default), "Linear", "Cos").
