import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from sklearn.model_selection import KFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.config import GROUND_TRUTH_THRESHOLD, MAX_CONDUCTIVITY
from src.data.eit_dataset import EITDataset
from src.training.metrics import EITMetricsEvaluator


def load_data(dataset_dir: str):
    dataset = EITDataset(PROJECT_ROOT / dataset_dir, target_type="grid", flatten_grid=True)
    if len(dataset) == 0:
        return None, None, None

    X_list, y_list, mask_list = [], [], []
    for i in range(len(dataset)):
        x, y, mask = dataset[i]
        X_list.append(x.numpy())
        y_list.append(y.numpy())
        mask_list.append(mask.numpy())

    return np.array(X_list), np.array(y_list), np.array(mask_list)


def evaluate_metrics(preds, targets, masks):
    # Convert numpy arrays to torch tensors (B, 1, 64, 64)
    B = preds.shape[0]
    size = int(preds.shape[1] ** 0.5)

    p_2d = torch.tensor(preds).view(B, 1, size, size)
    t_2d = torch.tensor(targets).view(B, 1, size, size)
    m_2d = torch.tensor(masks).view(B, 1, size, size)

    evaluator = EITMetricsEvaluator(threshold=GROUND_TRUTH_THRESHOLD, data_range=MAX_CONDUCTIVITY)

    metrics = {
        "rmse": evaluator.rmse(p_2d, t_2d, m_2d).item(),
        "ssim": evaluator.ssim(p_2d, t_2d, m_2d).item(),
        "pe": evaluator.position_error(p_2d, t_2d, m_2d).item(),
    }
    dice, iou = evaluator.dice_and_iou(p_2d, t_2d, m_2d)
    metrics["dice"] = dice.item()
    metrics["iou"] = iou.item()
    return metrics


def run_5_fold_cv(X, y, masks, best_params):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    output_dir = Path(__file__).resolve().parent

    fold_results = {"val_rmse": [], "val_ssim": [], "val_dice": [], "val_iou": [], "val_pe": []}

    print("\n" + "=" * 50)
    print("Iniciando 5-Fold Cross Validation (XGBoost):")
    print("=" * 50 + "\n")

    for fold, (train_ids, val_ids) in enumerate(kfold.split(X)):
        print(f"\n--- FOLD {fold + 1}/5 ---")
        X_train, y_train = X[train_ids], y[train_ids]
        X_val, y_val, m_val = X[val_ids], y[val_ids], masks[val_ids]

        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)

        # Calcular métricas espaciales en 2D
        metrics = evaluate_metrics(preds, y_val, m_val)

        fold_results["val_rmse"].append(metrics["rmse"])
        fold_results["val_ssim"].append(metrics["ssim"])
        fold_results["val_dice"].append(metrics["dice"])
        fold_results["val_iou"].append(metrics["iou"])
        fold_results["val_pe"].append(metrics["pe"])
        print(f"RMSE: {metrics['rmse']:.4f} | SSIM: {metrics['ssim']:.4f}")

    metrics_csv_path = output_dir / "xgboost_5fold_metrics.csv"
    with open(metrics_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Mean", "Std"])
        for k, v in fold_results.items():
            if v:
                mean_val = np.mean(v)
                std_val = np.std(v)
                writer.writerow([k, f"{mean_val:.6f}", f"{std_val:.6f}"])
                print(f"{k} (5-Fold CV): {mean_val:.4f} ± {std_val:.4f}")

    print(f"\n✅ Métricas de CV guardadas en local: {metrics_csv_path}")

    print("\n" + "=" * 50)
    print("Entrenando modelo final con el 100% del dataset...")

    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X, y)

    model_path = output_dir / "best_xgboost_model.json"
    final_model.save_model(model_path)
    print(f"✅ Modelo final XGBoost guardado en: {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="dataset/dataset_16e_all")
    parser.add_argument(
        "--params_file", type=str, required=True, help="Ruta al JSON de hiperparámetros de XGBoost"
    )
    args = parser.parse_args()

    with open(args.params_file) as f:
        best_params = json.load(f)

    X, y, masks = load_data(args.dataset_dir)
    if X is None:
        print("Dataset vacío.")
        return

    run_5_fold_cv(X, y, masks, best_params)


if __name__ == "__main__":
    main()
