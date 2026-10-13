# 🌲 Trunk EIT Dataset & Inverse Problem Models - AI Agent Rules

## 🎯 Project Goal
This repository supports a **Research Paper** comparing different Neural Network (NN) architectures for solving the **Inverse Problem of Electrical Impedance Tomography (EIT)** applied to tree trunks with internal rot anomalies.
The pipeline first procedurally generates stochastic simulated trunks (harmonic wood boundaries with internal anomalies), solves the physical forward problem via EIDORS (GNU Octave FEM), and builds a highly optimized dataset. The core research focuses on training inverse models to reconstruct the internal conductivity distribution $\sigma(x, y)$ (or FEM element values) strictly from boundary voltage measurements $V$.

---

## 🏗️ Architecture & Domain-Driven Design (DDD)

Strict Domain-Driven Design separating physical data generation from Machine Learning training.

### 1. `src/data/` — Domain, Generation & Forward Process
- **`models/`**: Pure mathematical domain entities (`Trunk`, `Anomaly`, `Harmonic`, `Pos`). A `Trunk` contains an outer boundary and a collection of internal anomalies.
- **`data_representations/`**: Rasterization and meshing (`grid.py`, `mesh.py`). Generates 2D conductivity matrices $\sigma \in \mathbb{R}^{64 \times 64}$ and geometric domain masks (`mask.npy`).
- **`forward_process/`**: EIDORS (Octave FEM) bridge via binary `.mat` files. Translates domain models to 2D meshes, solves the forward PDE $\nabla \cdot (\sigma \nabla V) = 0$, and extracts boundary voltages.
- **`stochastic/`**: Procedural generators (`generate_harmonic.py`) using Monte Carlo harmonic expansions for realistic wood variations.
- **`eit_dataset.py`**: PyTorch `Dataset` (`EITDataset`) optimized for high GPU throughput. Loads precomputed `.npy` arrays directly from disk with on-the-fly Z-score normalization for voltages. Supports optional internal potential maps (`load_voltage_maps=True`) for PINNs.
- **`interactive_tools/`**: Standalone HTML visualizers (`trunk_simulation.html`, `armonic_shape.html`) for inspecting trunk morphologies.

### 2. Dataset Structure (`dataset/`)
Outputs from `scripts/data/generate_dataset.py` follow a clean separation:
- `mesh/`: Shared FEM geometry (`nodes.npy`, `elems.npy`).
- `inputs/voltages_{pattern}/`: Boundary voltage vectors $X \in \mathbb{R}^{P}$.
- `targets/`:
  - `grid/`: 2D ground truth conductivity matrices $\sigma \in \mathbb{R}^{64 \times 64}$.
  - `mask/`: 2D boolean masks ($1 = \text{wood}, 0 = \text{air}$).
  - `elem_data/`: 1D vectors of conductivity per FEM mesh triangle.
  - `voltage_maps/`: (Optional for PINNs) 2D internal electric potential maps $V(x, y) \in \mathbb{R}^{P \times 64 \times 64}$.
- `metadata/` & `visualizations/`: Raw JSON models and rendered PNG previews.

### 3. `src/training/` — Machine Learning & Neural Networks
- **`arquitectures/`**:
  - `dnn.py`: `DynamicDNN` (fully-connected layers with BatchNorm, Dropout, and ReLU).
  - `cnn.py`: `DynamicCNN` (transposed convolutional decoder $8\times 8 \to 16\times 16 \to 32\times 32 \to 64\times 64$).
  - `autoencoder.py`: Physics-informed `EITAutoencoder` comprising `ConvEncoder` ($\sigma \to \hat{V}$, forward surrogate) and `ConvDecoder` ($V \to \hat{\sigma}$, inverse solver).
- **`dbar/`**: Non-iterative analytical baseline (`DBarSolver`) implementing the Calderón $\bar{\partial}$-scattering method in 2D without FEM mesh dependence.
- **`criterion/`**: Loss functions:
  - `rmse.py`: `MaskedRMSELoss` (physical conductivity error inside the trunk).
  - `ssim.py`: `MaskedSSIMLoss` (structural anomaly coherence).
  - `combined.py`: `HybridEITLoss` ($\alpha \cdot \text{RMSE} + \beta \cdot (1 - \text{SSIM})$).
  - `pinn.py`: `TVLoss` (Total Variation spatial regularization) and PDE residuals.
- **`metrics.py`**: Masked spatial metrics (RMSE, SSIM, Dice, IoU, Position Error) and sample-wise statistical evaluation functions (`compute_samplewise_metrics`, `evaluate_dataset_samplewise`).
- **`train_utils.py`**: Standardized training loop with `EarlyStopping` and best validation checkpoint restoration.
- **`experiment_logger.py`**: Centralized logging to `experiments_results.csv` capturing git commit hash (`git_commit`), hyperparameters, validation metrics, and sample-level test statistics (`mean ± std`).

---

## 🔬 Experimental Protocol & Validation Strategy

1. **No K-Fold Overuse**: With $N \ge 10,000$ simulated Monte Carlo samples, fold variance is negligible and 5-fold training introduces an unnecessary 5x overhead.
2. **Train / Val Split (80/20)**:
   - Split the main dataset into 80% training and 20% validation.
   - Train with `EarlyStopping(patience=10)` monitorizing `val_loss`.
   - Restore the weights corresponding to the best validation epoch before final evaluation.
3. **Independent Test Sets**:
   - **Test ID (In-Distribution)**: Evaluates generalization on unseen trunks with the same statistical distribution (~1,000 samples).
   - **Test OOD (Out-Of-Distribution)**: Evaluates structural robustness under severe morphological variations (high-order harmonics, multi-anomalies) (~1,000 samples).
4. **Sample-Level Statistical Reporting**:
   - Report test metrics as **sample-level `mean ± std`** across the hold-out test samples:
     $$\mu = \frac{1}{N}\sum_{i=1}^N \text{metric}_i, \quad \sigma = \sqrt{\frac{1}{N-1}\sum_{i=1}^N (\text{metric}_i - \mu)^2}$$
   - This reflects the natural anatomical difficulty distribution across tree trunks and provides publication-grade statistical rigor.
5. **Hyperparameter Tuning with Optuna**:
   - Never compare an untuned architecture against a tuned one.
   - Tune learning rate (log scale), weight decay, channels/layers, and loss weights on the 80/20 split using Optuna before final evaluation.

---

## 📁 Organization of Training Scripts (`scripts/training/`)

Every architecture family has an independent folder containing its dedicated Optuna tuner (`tune_*.py`) and evaluation script (`evaluate_*.py`):

```text
scripts/training/
├── dnn/
│   ├── dnn_rmse/          # tune_dnn_rmse.py, evaluate_dnn_rmse.py, best_params_dnn_rmse.json
│   └── dnn_hybrid/        # tune_dnn_hybrid.py, evaluate_dnn_hybrid.py
├── cnn/
│   ├── cnn_rmse/          # tune_cnn_rmse.py, evaluate_cnn_rmse.py
│   └── cnn_hybrid/        # tune_cnn_hybrid.py, evaluate_cnn_hybrid.py
├── autoencoder/
│   ├── autoencoder_rmse/  # tune_autoencoder_rmse.py, evaluate_autoencoder_rmse.py
│   └── autoencoder_hybrid/# tune_autoencoder_hybrid.py, evaluate_autoencoder_hybrid.py
├── xgboost/               # tune_xgboost.py, evaluate_xgboost.py, best_params_xgboost.json
└── dbar/                  # evaluate_dbar.py (analytical baseline, saves vector .pdf figures)
```

---

## ⚠️ Agent Pitfalls & Golden Rules

1. **Dataloader reads `.npy`, not JSON**: `EITDataset` loads arrays directly from `inputs/` and `targets/` for maximum GPU efficiency. Never parse metadata JSONs in PyTorch loops.
2. **Strict Masking on Loss**: It is NOT enough to mask final evaluation metrics. Loss functions during training MUST multiply both predictions and targets by `mask` before calculating gradients:
   $$\mathcal{L} = f(\hat{\sigma} \odot M, \sigma \odot M)$$
   Otherwise, the network wastes capacity predicting zero conductivity in the "air" domain.
3. **Metric Masking**: Always pass `mask` to spatial evaluation metrics (RMSE, SSIM, Dice, PE) to evaluate exclusively within the wood cross-section.
4. **Vector Graphics for Paper**: Prioritize exporting qualitative comparison figures as vector `.pdf` rather than `.png` to allow lossless LaTeX integration.
5. **Fair Baselines**: Always tune hyperparameters with Optuna before logging final benchmark metrics.
6. **Centralized Logging**: Every evaluation script must log its run to `experiments_results.csv` via `log_experiment_result(...)`.
