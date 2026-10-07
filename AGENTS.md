# 🌲 Trunk EIT Dataset & Inverse Problem Models - AI Agent Rules

## 🎯 Project Goal
This repository supports a **Research Paper** comparing different Neural Network (NN) architectures for solving the **Inverse Problem of Electrical Impedance Tomography (EIT)** applied to tree trunks. 
The pipeline first procedurally generates stochastic simulated trunks (harmonic wood shapes with internal anomalies), solves the forward problem via EIDORS (Octave), and builds a highly optimized dataset. The core research focuses on training models to reconstruct the internal conductivity grid (or FEM mesh) strictly from boundary voltage measurements.

## 🏗️ Architecture

Strict Domain-Driven Design separating data generation from ML training.

### 1. `src/data/` — Domain, Generation & Forward Process
- **`models/`**: Pure math domain (`Trunk`, `Anomaly`, `Harmonic`, `Pos`). A `Trunk` contains a base anomaly and a list of inner anomalies.
- **`data_representations/`**: Rasterization mapping (`grid.py`). Generates 2D conductivity matrices and geometric masks from domain models. 
- **`forward_process/`**: EIDORS (Octave FEM) bridge via binary `.mat` files. Translates domain models to FEM meshes, runs simulations, and yields boundary voltages.
- **`stochastic/`**: Procedural generators (Poisson/Lognormal) for natural wood variations.
- **`eit_dataset.py`**: PyTorch `Dataset` that aggressively optimizes loading. It reads pre-computed `.npy` arrays directly from the dataset architecture (no on-the-fly math).

### 2. Dataset Structure (`dataset/`)
Outputs from `scripts/data/generate_dataset.py` follow a strict ML inputs/targets separation:
- `mesh/`: Shared FEM geometry (`nodes.npy`, `elems.npy`).
- `inputs/voltages_{pattern}/`: The `X` for ML. Boundary voltage vectors.
- `targets/`: The `Y` for ML. `grid/` (2D conductivity array), `mask/` (2D boolean array), and `elem_data/` (1D vector of triangle conductivities).
- `metadata/` & `visualizations/`: Raw JSON models and human-readable `.png` grids.

### 3. `src/training/` — Machine Learning & Neural Networks
- **`arquitectures/`**: PyTorch neural network definitions comparing different approaches for the inverse problem.
- **`metrics.py`**: Evaluation logic (SSIM, RMSE, Dice, Spatial Error). *Note: Metrics apply the geometric mask to ignore "air" pixels outside the trunk.*
- **`train_utils.py`**: Standardized training/validation loops.

### 4. `src/training/` — Pipeline, Evaluation & Hyperparameters
- **Loss Functions**: 
  - For 1D Vector Models (`elem_data`): Use **Masked MSE/RMSE**.
  - For 2D Image Models (`grid`): Implement a **Hybrid Loss** combining Masked RMSE (for accurate physical conductivity values) and Masked SSIM (for structural and geometric coherence of the anomalies). *Example: `Loss = α * Masked_RMSE + β * (1 - Masked_SSIM)`*.
- **Validation Strategy**: Use **5-Fold Cross-Validation** for the final architectural comparisons. The research paper must report metrics as `mean ± std` across the folds to prove statistical significance and avoid "lucky split" bias.
- **Hyperparameter Tuning**: Use **Optuna** (TPE sampler) to guarantee fair comparisons between baseline and advanced models. 
  - Tune learning rate (log scale), weight decay, and architecture-specific dimensions (depth, channels, dropout).
  - *Workflow*: Run Optuna on a standard Train/Val split to find the optimal hyperparameters. Then, take that best configuration and run it through the 5-Fold CV to get the publishable metrics.

## 📝 Commands

```bash
uv sync                 # Install deps
uv run python scripts/data/generate_dataset.py --samples 5 --electrodes 16 # Generate
uv run pytest -v        # Run tests
uv run ruff check .     # Lint
```

## ⚠️ Agent Pitfalls
1. **Dataloader reads `.npy`, not JSON**: The ML dataloader (`EITDataset`) does not parse JSONs anymore. It directly loads arrays from `inputs/` and `targets/` for maximum GPU throughput.
2. **One `elem_data` per Trunk**: `elem_data` is the physical ground truth and is invariant to the stimulation pattern.
3. **Metric Masking**: Always pass the `mask` to spatial metrics to ensure the model is evaluated on the wood domain, not the background.
4. **Vector Graphics for Paper**: Since this is for a research paper, prioritize saving analysis plots, graphs, and charts as `.pdf` (vector format) rather than `.png` to ensure lossless quality for LaTeX integration.
5. **Strict Masking on Loss**: It is not enough to mask the final evaluation metrics. The Loss Function during training MUST also be masked. Multiply both the prediction and the ground truth by the boolean `mask` array before computing gradients, otherwise the network will waste capacity learning the "air" outside the trunk.
6. **Fair Baselines**: Never compare an untuned architecture against a tuned one. Optuna must be run for every architecture proposed in the paper before the final K-Fold evaluation.
