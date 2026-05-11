# 🌲 Trunk EIT Dataset - AI Agent Context & Rules

## 🎯 Project Goal
The goal of this project is to build an automated pipeline and interface to generate a Machine Learning dataset for Electrical Impedance Tomography (EIT) applied to tree trunks. 
The system translates abstract geometric models of wood anomalies (moisture, rot) into conductivity grids, runs forward simulations to obtain boundary voltages, and packages the data for ML training (Features: Voltages, Ground Truth: Conductivity maps).

## 🛠️ Tech Stack & Environment
- **Language**: Python 3.10+
- **Environment & Package Manager**: `uv`. Always use `uv` commands (e.g., `uv pip install`, `uv run`) for dependency management.
- **Math & Arrays**: `numpy` (essential for vectorized grid operations).
- **EIT Simulation**: [EIDORS](http://eidors3d.sourceforge.net/) (MATLAB/Octave library). Interfaced via Python scripts.

## 🏗️ Architecture & Separation of Concerns
The project strictly separates geometric abstractions from numerical representations and physical simulations. Do not mix these layers.

### 1. `src/base.py` (Domain Models)
- **Role**: Pure geometric and physical abstraction.
- **Implementation**: Strictly use Python `@dataclass`. 
- **Components**: `Pos` (Coordinates), `Anomaly` (Radius, center, conductivity), and `Trunk` (Main body + list of anomalies).
- **Rule**: No numerical grids or simulation logic here. Only mathematical definitions.

### 2. `src/grid.py` (Discretization)
- **Role**: Translates domain models into discrete representations for ML and simulation.
- **Implementation**: Contains functions like `generate_grid(trunk: Trunk, resolution: int) -> np.ndarray`.
- **Rule**: Use highly optimized, vectorized `numpy` operations (e.g., boolean masks for circles) to generate 2D arrays. Avoid `for` loops for pixel-level operations.

### 3. `src/eit_simulation.py` (Forward Problem & EIDORS Bridge)
- **Role**: Solves the EIT forward problem.
- **Implementation**: `eit_simulation(grid: np.ndarray, protocol_input) -> np.ndarray`.
- **Rule**: This layer handles the translation between our NumPy grids/meshes and the EIDORS simulation engine. It handles electrode placement, injection protocols, and voltage extraction.

### 4. `src/pipeline.py` (Dataset Orchestration)
- **Role**: Generates the final dataset.
- **Flow**: Randomly generates `Trunk` configurations -> Maps to grids -> Runs EIT simulation -> Saves (HDF5, Parquet, or JSON).

## 🧠 AI Agent Coding Guidelines
When writing or modifying code for this project, the AI must adhere to the following rules:
1. **Type Hinting Strictness**: All functions, methods, and classes MUST have accurate Python type hints (e.g., `np.ndarray`, `List[Anomaly]`).
2. **Vectorization**: Always prefer `numpy` vectorization over list comprehensions or standard loops for mathematical and grid operations.
3. **Conductivity Standard**: By default, physical properties should refer to **conductivity ($\sigma$)** in S/m, rather than generic impedance, unless specified otherwise.
4. **Documentation**: Write clear docstrings for all public methods. Explain the *physical or mathematical reasoning* behind EIT-specific calculations.
5. **Simplicity**: Keep functions small and modular. Ensure that `grid.py` can scale to different resolutions without breaking `base.py`.