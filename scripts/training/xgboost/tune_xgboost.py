import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.eit_dataset import EITDataset


def load_data(dataset_dir: str):
    # IMPORTANTE: Usar grid aplanado, NO elem_data (por la variabilidad del mallado)
    dataset = EITDataset(PROJECT_ROOT / dataset_dir, target_type="grid", flatten_grid=True)
    if len(dataset) == 0:
        return None, None

    X_list, y_list = [], []
    for i in range(len(dataset)):
        x, y, _ = dataset[i]
        X_list.append(x.numpy())
        y_list.append(y.numpy())

    return np.array(X_list), np.array(y_list)


def objective(trial, X, y):
    param = {
        "n_estimators": trial.suggest_categorical(
            "n_estimators", [20, 50, 100]
        ),  # Pocos para 4096 outputs
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "tree_method": "hist",
        "n_jobs": -1,
    }

    # Split rápido para Optuna
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGBRegressor natively supports multi-output regression
    model = xgb.XGBRegressor(**param)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    return mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="dataset/dataset_16e_all")
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent

    print("Cargando dataset para XGBoost (puede tardar un poco y requerir RAM)...")
    X, y = load_data(args.dataset_dir)
    if X is None:
        print("Error al cargar datos.")
        return

    print(f"Datos cargados: X={X.shape}, y={y.shape}")

    print("\nIniciando Optuna Tuning para XGBoost...")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, X, y), n_trials=args.trials)

    best_params = study.best_params
    best_params["tree_method"] = "hist"
    best_params["n_jobs"] = -1

    out_file = output_dir / "best_params_xgboost.json"
    with open(out_file, "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"\n✅ Mejores parámetros de XGBoost guardados en: {out_file}")


if __name__ == "__main__":
    main()
