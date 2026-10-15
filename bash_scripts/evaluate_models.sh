#!/usr/bin/env bash
set -e

# Configuración de datasets y replicabilidad
SEED=42
TRAIN_DIR="dataset/dataset_16e_adjacent"
TEST_ID_DIR="dataset/dataset_16e_adjacent_test_id"
TEST_OOD_DIR="dataset/dataset_16e_adjacent_test_ood"

echo "==========================================================="
echo "  Ejecutando Pipeline de Evaluación y Benchmarks de EIT"
echo "  Seed: $SEED"
echo "  Train: $TRAIN_DIR"
echo "  Test ID: $TEST_ID_DIR | Test OOD: $TEST_OOD_DIR"
echo "==========================================================="

# 1. Baseline Analítico Directo (D-Bar) - Carga reconstrucciones precalculadas
echo -e "\n[1/8] Evaluando D-Bar..."
uv run python scripts/training/dbar/evaluate_dbar.py \
    --seed $SEED --train_dir $TRAIN_DIR --test_id_dir $TEST_ID_DIR --test_ood_dir $TEST_OOD_DIR

# 2. Baseline Machine Learning Tabular (XGBoost)
echo -e "\n[2/8] Evaluando XGBoost..."
uv run python scripts/training/xgboost/evaluate_xgboost.py \
    --seed $SEED --train_dir $TRAIN_DIR --test_id_dir $TEST_ID_DIR --test_ood_dir $TEST_OOD_DIR

# 3. Modelos Fully-Connected (DynamicDNN)
echo -e "\n[3/8] Evaluando DynamicDNN (Masked RMSE)..."
uv run python scripts/training/dnn/dnn_rmse/evaluate_dnn_rmse.py \
    --seed $SEED --train_dir $TRAIN_DIR --test_id_dir $TEST_ID_DIR --test_ood_dir $TEST_OOD_DIR

echo -e "\n[4/8] Evaluando DynamicDNN (Hybrid RMSE + SSIM)..."
uv run python scripts/training/dnn/dnn_hybrid/evaluate_dnn_hybrid.py \
    --seed $SEED --train_dir $TRAIN_DIR --test_id_dir $TEST_ID_DIR --test_ood_dir $TEST_OOD_DIR

# 4. Modelos Convolucionales Transpuestos (DynamicCNN)
echo -e "\n[5/8] Evaluando DynamicCNN (Masked RMSE)..."
uv run python scripts/training/cnn/cnn_rmse/evaluate_cnn_rmse.py \
    --seed $SEED --train_dir $TRAIN_DIR --test_id_dir $TEST_ID_DIR --test_ood_dir $TEST_OOD_DIR

echo -e "\n[6/8] Evaluando DynamicCNN (Hybrid RMSE + SSIM)..."
uv run python scripts/training/cnn/cnn_hybrid/evaluate_cnn_hybrid.py \
    --seed $SEED --train_dir $TRAIN_DIR --test_id_dir $TEST_ID_DIR --test_ood_dir $TEST_OOD_DIR

# 5. Modelos Basados en Física (EITAutoencoder)
echo -e "\n[7/8] Evaluando EITAutoencoder (Masked RMSE)..."
uv run python scripts/training/autoencoder/autoencoder_rmse/evaluate_autoencoder_rmse.py \
    --seed $SEED --train_dir $TRAIN_DIR --test_id_dir $TEST_ID_DIR --test_ood_dir $TEST_OOD_DIR

echo -e "\n[8/8] Evaluando EITAutoencoder (Hybrid RMSE + SSIM)..."
uv run python scripts/training/autoencoder/autoencoder_hybrid/evaluate_autoencoder_hybrid.py \
    --seed $SEED --train_dir $TRAIN_DIR --test_id_dir $TEST_ID_DIR --test_ood_dir $TEST_OOD_DIR

echo -e "\n==========================================================="
echo "  ✅ Todas las evaluaciones completadas con éxito."
echo "  Resultados consolidados en experiments_results.csv"
echo "==========================================================="
