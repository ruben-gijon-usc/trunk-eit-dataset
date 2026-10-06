# Trunk EIT Dataset - Testing TODO List 🧪

This document outlines the testing strategy for the dataset generation and machine learning pipeline. As the architecture has matured, implementing these tests will ensure reproducibility and stability.

## 1. Data Pipeline & DataLoader Tests (`test_dataset.py`)
- [ ] **Tensor Shapes:** Verify `EITDataset` returns tuples of exactly `(voltage_tensor, grid_tensor, mask_tensor)` with correct dimensions (e.g., `[B, 1, 64, 64]` for grid/mask if unflattened).
- [ ] **Deterministic Loading:** Ensure that multiple epochs load the exact same data order when the dataloader seed is fixed.
- [ ] **Missing Files Handling:** Assert that `EITDataset` correctly ignores or warns if a sample is missing its corresponding `.npy` target or input.
- [ ] **Normalization Checks:** Verify that input voltages and target conductivities do not contain `NaN` or `Inf` values when drawn from the dataset.

## 2. Geometric & Stochastic Tests (`test_geometry.py`)
- [ ] **Bounding Boxes:** Test `trunk.get_bounds()` on various `Harmonic` shapes to ensure the coordinates perfectly encapsulate the shape without clipping.
- [ ] **Containment Logic:** Generate random coordinates and assert that `trunk.base.contains(pos)` and `anomaly.contains(pos)` correctly identify inside/outside points.
- [ ] **Stochastic Ranges:** Run `StochasticTrunkFactory` 1,000 times and assert that conductivities, radii, and harmonic perturbations always stay within the defined mathematical limits.

## 3. Metric Function Tests (`test_metrics.py`)
- [ ] **Masked Metrics Integrity:** Pass identical mock predictions and targets to `compute_all_metrics` and assert `RMSE == 0.0`, `SSIM == 1.0`, and `Dice == 1.0`.
- [ ] **Background Ignorance:** Modify the "air" (background) pixels of a mock prediction and assert that the metrics (which use the binary mask) **do not change**.
- [ ] **Position Error Verification:** Test the Spatial Error metric by feeding a circular anomaly offset by a known distance (e.g., `0.5` units) and assert the error calculation matches the distance.

## 4. EIDORS Bridge Tests (`test_simulator.py`)
- [ ] **IO Performance:** Mock the `simulate_forward_process` and verify that the `.mat` binary conversion properly serializes/deserializes data without loss of precision compared to text files.
- [ ] **Thread Safety (Temp Dirs):** Run 4 concurrent forward processes and ensure they execute in isolated temporary directories without overwriting each other's `.m` or `.mat` files.
- [ ] **Failure Handling:** Force an Octave crash (e.g., by passing invalid extreme mesh parameters) and assert that the Python bridge catches it and safely returns `None` instead of hanging.

## 5. Neural Network Architecture Tests (`test_models.py`)
- [ ] **Forward Pass:** Initialize `SimpleFCNN` (or future U-Nets), pass a dummy voltage batch `[B, 16]`, and assert the output matches the required grid shape `[B, 1, 64, 64]`.
- [ ] **Backward Pass:** Compute a dummy MSE loss and assert that gradients successfully flow backward to the first input layer.
