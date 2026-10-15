#!/usr/bin/env bash
set -e

# Configuración de búsqueda con Optuna
TRIALS=25
XGB_TRIALS=10
EPOCHS=50
SEED=42
DATASET_DIR="dataset/dataset_16e_adjacent"

echo "==========================================================="
echo "  Iniciando Búsqueda de Hiperparámetros (Optuna Tuning)"
echo "  Trials: $TRIALS (XGBoost: $XGB_TRIALS) | Epochs: $EPOCHS | Seed: $SEED"
echo "  Dataset: $DATASET_DIR"
echo "==========================================================="

# 1. XGBoost (árboles tabulares, no usa épocas)
echo -e "\n--- [1/7] Tuning XGBoost ---"
uv run python scripts/training/xgboost/tune_xgboost.py --trials $XGB_TRIALS --seed $SEED --dataset_dir $DATASET_DIR

# 2. DynamicDNN
echo -e "\n--- [2/7] Tuning DynamicDNN (Masked RMSE) ---"
uv run python scripts/training/dnn/dnn_rmse/tune_dnn_rmse.py --trials $TRIALS --epochs $EPOCHS --seed $SEED --dataset_dir $DATASET_DIR

echo -e "\n--- [3/7] Tuning DynamicDNN (Hybrid RMSE + SSIM) ---"
uv run python scripts/training/dnn/dnn_hybrid/tune_dnn_hybrid.py --trials $TRIALS --epochs $EPOCHS --seed $SEED --dataset_dir $DATASET_DIR

# 3. DynamicCNN
echo -e "\n--- [4/7] Tuning DynamicCNN (Masked RMSE) ---"
uv run python scripts/training/cnn/cnn_rmse/tune_cnn_rmse.py --trials $TRIALS --epochs $EPOCHS --seed $SEED --dataset_dir $DATASET_DIR

echo -e "\n--- [5/7] Tuning DynamicCNN (Hybrid RMSE + SSIM) ---"
uv run python scripts/training/cnn/cnn_hybrid/tune_cnn_hybrid.py --trials $TRIALS --epochs $EPOCHS --seed $SEED --dataset_dir $DATASET_DIR

# 4. EITAutoencoder
echo -e "\n--- [6/7] Tuning EITAutoencoder (Masked RMSE) ---"
uv run python scripts/training/autoencoder/autoencoder_rmse/tune_autoencoder_rmse.py --trials $TRIALS --epochs $EPOCHS --seed $SEED --dataset_dir $DATASET_DIR

echo -e "\n--- [7/7] Tuning EITAutoencoder (Hybrid RMSE + SSIM) ---"
uv run python scripts/training/autoencoder/autoencoder_hybrid/tune_autoencoder_hybrid.py --trials $TRIALS --epochs $EPOCHS --seed $SEED --dataset_dir $DATASET_DIR

echo -e "\n==========================================================="
echo "  ✅ Tuning de Optuna completado para todos los modelos."
echo "  Ficheros best_params_*.json actualizados."
echo "==========================================================="
