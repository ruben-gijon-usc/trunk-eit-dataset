import argparse
from pathlib import Path
import json
import optuna
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.eit_dataset import EITDataset

def load_data(dataset_dir: str):
    dataset = EITDataset(PROJECT_ROOT / dataset_dir, target_type="elem_data")
    if len(dataset) == 0:
        return None, None
    
    # Extraer todos los tensores a numpy arrays
    X_list, y_list = [], []
    for i in range(len(dataset)):
        x, y, _ = dataset[i]
        X_list.append(x.numpy())
        y_list.append(y.numpy())
        
    return np.array(X_list), np.array(y_list)

def objective(trial, X, y):
    # Sugerencias de Optuna para XGBoost
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "tree_method": "hist", # Más rápido
        "n_jobs": -1
    }

    # Split estándar Train/Val (80/20)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # MultiOutputRegressor entrena un modelo por cada elemento del mesh (Agent Pitfalls)
    # Nota: Si y tiene miles de elementos, esto puede ser costoso. 
    model = MultiOutputRegressor(xgb.XGBRegressor(**param))
    model.fit(X_train, y_train)
    
    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    return mse

def run_5_fold_cv(X, y, best_params):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []

    print("\n" + "="*50)
    print("Iniciando 5-Fold Cross Validation con hiperparámetros óptimos (XGBoost):")
    print(json.dumps(best_params, indent=2))
    print("="*50 + "\n")

    for fold, (train_ids, val_ids) in enumerate(kfold.split(X)):
        print(f"--- FOLD {fold + 1}/5 ---")
        X_train, y_train = X[train_ids], y[train_ids]
        X_val, y_val = X[val_ids], y[val_ids]
        
        # Inyectar tree_method y n_jobs estáticos que no son parte del suggest
        params = best_params.copy()
        params["tree_method"] = "hist"
        params["n_jobs"] = -1

        model = MultiOutputRegressor(xgb.XGBRegressor(**params))
        model.fit(X_train, y_train)
        
        preds = model.predict(X_val)
        mse = mean_squared_error(y_val, preds)
        fold_metrics.append(mse)
        print(f"MSE Fold {fold+1}: {mse:.4f}")
        
    print("\n" + "="*50)
    print("Resultados Finales (5-Fold CV XGBoost):")
    print(f"MSE: {np.mean(fold_metrics):.4f} ± {np.std(fold_metrics):.4f}")
    print("="*50)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="dataset/dataset_16e_all")
    parser.add_argument("--trials", type=int, default=10, help="Número de trials de Optuna")
    args = parser.parse_args()

    print("Cargando dataset completo en memoria...")
    X, y = load_data(args.dataset_dir)
    if X is None:
        print("Dataset vacío o no se encontró elem_data.")
        return
        
    print(f"Datos cargados: X={X.shape}, y={y.shape}")

    # Paso 1: Optuna Tuning
    print("\nIniciando Optuna Tuning para XGBoost...")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, X, y), n_trials=args.trials)

    print("\nMejores parámetros encontrados:")
    best_params = study.best_params
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # Paso 2: 5-Fold Cross Validation
    run_5_fold_cv(X, y, best_params)

if __name__ == "__main__":
    main()
