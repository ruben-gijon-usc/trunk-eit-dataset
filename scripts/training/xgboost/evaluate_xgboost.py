import argparse
import json
import csv
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import KFold
import xgboost as xgb

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.eit_dataset import EITDataset
from src.training.metrics import EITMetricsEvaluator
from scripts.data.config import GROUND_TRUTH_THRESHOLD, MAX_CONDUCTIVITY

def load_data(dataset_dir: Path):
    if not dataset_dir.exists():
        return None, None, None
        
    dataset = EITDataset(dataset_dir, target_type="grid", flatten_grid=True)
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
    B = preds.shape[0]
    size = int(preds.shape[1] ** 0.5)
    
    p_2d = torch.tensor(preds).view(B, 1, size, size)
    t_2d = torch.tensor(targets).view(B, 1, size, size)
    m_2d = torch.tensor(masks).view(B, 1, size, size)

    evaluator = EITMetricsEvaluator(threshold=GROUND_TRUTH_THRESHOLD, data_range=MAX_CONDUCTIVITY)
    
    metrics = {
        "rmse": evaluator.rmse(p_2d, t_2d, m_2d).item(),
        "ssim": evaluator.ssim(p_2d, t_2d, m_2d).item(),
        "pe": evaluator.position_error(p_2d, t_2d, m_2d).item()
    }
    dice, iou = evaluator.dice_and_iou(p_2d, t_2d, m_2d)
    metrics["dice"] = dice.item()
    metrics["iou"] = iou.item()
    return metrics

def evaluate_on_test_set(model, X_test, y_test, masks_test, test_name):
    print(f"\n" + "=" * 50)
    print(f"Evaluando modelo final en Dataset de Test: {test_name}")
    print("=" * 50)
    
    preds = model.predict(X_test)
    metrics = evaluate_metrics(preds, y_test, masks_test)
    
    print(f"--- Resultados {test_name} ---")
    for k, v in metrics.items():
        print(f"  {k.upper()}: {v:.4f}")
    return metrics

def run_evaluation_pipeline(X_train, y_train, masks_train, X_id, y_id, m_id, X_ood, y_ood, m_ood, best_params):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    output_dir = Path(__file__).resolve().parent
    
    fold_results = {
        "val_rmse": [], "val_ssim": [], 
        "val_dice": [], "val_iou": [], "val_pe": []
    }

    print("\n" + "="*50)
    print("Iniciando 5-Fold Cross Validation (XGBoost):")
    print("="*50 + "\n")

    for fold, (train_ids, val_ids) in enumerate(kfold.split(X_train)):
        print(f"\n--- FOLD {fold + 1}/5 ---")
        X_tr, y_tr = X_train[train_ids], y_train[train_ids]
        X_val, y_val, m_val = X_train[val_ids], y_train[val_ids], masks_train[val_ids]
        
        model = xgb.XGBRegressor(**best_params)
        model.fit(X_tr, y_tr)
        
        preds = model.predict(X_val)
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
    
    print("\n" + "="*50)
    print("Entrenando modelo final con el 100% del dataset...")
    
    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_train, y_train)
    
    model_path = output_dir / "best_xgboost_model.json"
    final_model.save_model(model_path)
    print(f"✅ Modelo final XGBoost guardado en: {model_path}")

    # ---------------------------------------------------------
    # Evaluar en Test Sets Independientes
    # ---------------------------------------------------------
    test_metrics = {}
    if X_id is not None:
        test_metrics["ID"] = evaluate_on_test_set(final_model, X_id, y_id, m_id, "Test ID (Misma Distribución)")
        
    if X_ood is not None:
        test_metrics["OOD"] = evaluate_on_test_set(final_model, X_ood, y_ood, m_ood, "Test OOD (Topologías Complejas)")

    if test_metrics:
        test_csv_path = output_dir / "xgboost_test_metrics.csv"
        with open(test_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Test_Set", "Metric", "Value"])
            for test_name, metrics in test_metrics.items():
                for k, v in metrics.items():
                    writer.writerow([test_name, k, f"{v:.6f}"])
        print(f"✅ Métricas de Testeo guardadas en: {test_csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="dataset/dataset_16e_all")
    parser.add_argument("--test_id_dir", type=str, default="dataset/dataset_16e_all_test_id")
    parser.add_argument("--test_ood_dir", type=str, default="dataset/dataset_16e_all_test_ood")
    parser.add_argument("--params_file", type=str, required=True, help="Ruta al JSON de hiperparámetros de XGBoost")
    args = parser.parse_args()

    with open(args.params_file, "r") as f:
        best_params = json.load(f)

    print(f"Cargando Dataset Train: {args.train_dir}")
    X_train, y_train, masks_train = load_data(PROJECT_ROOT / args.train_dir)
    if X_train is None:
        print("Dataset de entrenamiento vacío.")
        return

    print(f"Cargando Dataset Test ID: {args.test_id_dir}")
    X_id, y_id, m_id = load_data(PROJECT_ROOT / args.test_id_dir)

    print(f"Cargando Dataset Test OOD: {args.test_ood_dir}")
    X_ood, y_ood, m_ood = load_data(PROJECT_ROOT / args.test_ood_dir)

    run_evaluation_pipeline(X_train, y_train, masks_train, X_id, y_id, m_id, X_ood, y_ood, m_ood, best_params)


if __name__ == "__main__":
    main()
