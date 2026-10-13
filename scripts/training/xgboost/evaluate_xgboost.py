import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.eit_dataset import EITDataset
from src.training.experiment_logger import log_experiment_result
from src.training.metrics import compute_samplewise_metrics
from src.training.train_utils import set_seed



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


def evaluate_samplewise_numpy(preds_np, targets_np, masks_np):
    p = torch.from_numpy(preds_np).float()
    t = torch.from_numpy(targets_np).float()
    m = torch.from_numpy(masks_np).float() if masks_np is not None else None

    raw_metrics = compute_samplewise_metrics(p, t, m)
    summary = {}
    for k, vals in raw_metrics.items():
        summary[k] = (float(np.mean(vals)), float(np.std(vals)))
    return summary


def run_evaluation_pipeline(
    X_train,
    y_train,
    masks_train,
    X_id,
    y_id,
    m_id,
    X_ood,
    y_ood,
    m_ood,
    best_params,
    seed: int = 42,
    train_dir="",
    test_id_dir="",
    test_ood_dir="",
):
    output_dir = Path(__file__).resolve().parent

    print("\n" + "=" * 50)
    print(f"Iniciando Evaluación (XGBoost, Seed: {seed}):")
    print(json.dumps(best_params, indent=2))
    print("=" * 50 + "\n")

    X_tr, X_val, y_tr, y_val, m_tr, m_val = train_test_split(
        X_train, y_train, masks_train, test_size=0.2, random_state=seed
    )

    print(f"Entrenando XGBoost en {len(X_tr)} muestras, validando en {len(X_val)} muestras...")
    model_params = {**best_params, "random_state": seed}
    model = xgb.XGBRegressor(**model_params)
    model.fit(X_tr, y_tr)


    preds_val = model.predict(X_val)
    val_sample_summary = evaluate_samplewise_numpy(preds_val, y_val, m_val)
    val_metrics = {k: v[0] for k, v in val_sample_summary.items()}

    print("\n--- Métricas de Validación (Hold-out 20%) ---")
    for k, v in val_metrics.items():
        print(f"  {k.upper()}: {v:.4f}")

    # Guardar modelo
    model_path = output_dir / "best_xgboost_model.json"
    model.save_model(model_path)
    print(f"\n✅ Modelo XGBoost guardado en: {model_path}")

    # Test ID
    test_id_summary = None
    if X_id is not None:
        print("\n--- Evaluación en Test ID (Misma Distribución) ---")
        preds_id = model.predict(X_id)
        test_id_summary = evaluate_samplewise_numpy(preds_id, y_id, m_id)
        for k, (m_val, s_val) in test_id_summary.items():
            print(f"  {k.upper()}: {m_val:.4f} ± {s_val:.4f}")

    # Test OOD
    test_ood_summary = None
    if X_ood is not None:
        print("\n--- Evaluación en Test OOD (Topologías Complejas) ---")
        preds_ood = model.predict(X_ood)
        test_ood_summary = evaluate_samplewise_numpy(preds_ood, y_ood, m_ood)
        for k, (m_val, s_val) in test_ood_summary.items():
            print(f"  {k.upper()}: {m_val:.4f} ± {s_val:.4f}")

    log_experiment_result(
        script="scripts/training/xgboost/evaluate_xgboost.py",
        model_name="XGBoost",
        loss_type="rmse",
        train_dataset=train_dir,
        test_id_dataset=test_id_dir if test_id_summary else None,
        test_ood_dataset=test_ood_dir if test_ood_summary else None,
        val_metrics=val_metrics,
        test_id_metrics=test_id_summary,
        test_ood_metrics=test_ood_summary,
        hyperparameters=best_params,
    )


def main():
    default_params = Path(__file__).resolve().parent / "best_params_xgboost.json"
    parser = argparse.ArgumentParser(description="Evaluación de XGBoost para EIT")
    parser.add_argument("--train_dir", type=str, default="dataset/dataset_16e_all")
    parser.add_argument("--test_id_dir", type=str, default="dataset/dataset_16e_adjacent_test_id")
    parser.add_argument("--test_ood_dir", type=str, default="dataset/dataset_16e_adjacent_test_ood")
    parser.add_argument(
        "--params_file",
        type=str,
        default=str(default_params),
        help="Ruta al JSON de hiperparámetros",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para replicabilidad",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    if not Path(args.params_file).exists():
        print(f"Aviso: {args.params_file} no existe. Usando configuración base.")
        best_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "n_jobs": -1,
            "tree_method": "hist",
        }
    else:
        with open(args.params_file) as f:
            best_params = json.load(f)

    print(f"Cargando Dataset Train: {args.train_dir}")
    X_train, y_train, masks_train = load_data(PROJECT_ROOT / args.train_dir)
    if X_train is None:
        raise FileNotFoundError(f"No se pudo cargar el dataset de train en {args.train_dir}")

    print(f"Cargando Dataset Test ID: {args.test_id_dir}")
    X_id, y_id, m_id = load_data(PROJECT_ROOT / args.test_id_dir)

    print(f"Cargando Dataset Test OOD: {args.test_ood_dir}")
    X_ood, y_ood, m_ood = load_data(PROJECT_ROOT / args.test_ood_dir)

    run_evaluation_pipeline(
        X_train,
        y_train,
        masks_train,
        X_id,
        y_id,
        m_id,
        X_ood,
        y_ood,
        m_ood,
        best_params,
        seed=args.seed,
        train_dir=args.train_dir,
        test_id_dir=args.test_id_dir,
        test_ood_dir=args.test_ood_dir,
    )



if __name__ == "__main__":
    main()
