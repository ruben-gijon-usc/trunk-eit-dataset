import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.config import MAX_CONDUCTIVITY
from src.data.eit_dataset import EITDataset
from src.training.arquitectures.dnn import DynamicDNN, grid_flattened_metrics_fn
from src.training.criterion.combined import HybridEITLoss
from src.training.experiment_logger import log_experiment_result
from src.training.metrics import evaluate_dataset_samplewise
from src.training.train_utils import set_seed, train_model


def run_evaluation_pipeline(
    train_dataset,
    test_id_dataset,
    test_ood_dataset,
    best_params,
    input_dim,
    output_dim,
    device,
    epochs,
    patience: int = 15,
    seed: int = 42,
    train_dir="",
    test_id_dir="",
    test_ood_dir="",
):
    output_dir = Path(__file__).resolve().parent

    print("\n" + "=" * 50)
    print(f"Iniciando Evaluación [DNN - Hybrid Loss (RMSE + SSIM)] (Seed: {seed}):")
    print(json.dumps(best_params, indent=2))
    print("=" * 50 + "\n")

    val_size = int(len(train_dataset) * 0.2)
    train_size = len(train_dataset) - val_size
    train_sub, val_sub = random_split(
        train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_sub, batch_size=best_params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=best_params["batch_size"], shuffle=False)

    n_layers = best_params["n_layers"]
    hidden_dims = [best_params[f"hidden_dim_l{i}"] for i in range(n_layers)]
    activation = best_params.get("activation", "relu")

    model = DynamicDNN(
        input_dim, output_dim, hidden_dims, best_params.get("dropout_rate", 0.0), activation=activation
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=best_params["lr"], weight_decay=best_params.get("weight_decay", 1e-4)
    )
    alpha = best_params.get("alpha", 1.0)
    beta = best_params.get("beta", 5.0)
    criterion = HybridEITLoss(alpha=alpha, beta=beta, data_range=MAX_CONDUCTIVITY)

    print(f"Entrenando modelo en {train_size} muestras, validando en {val_size} muestras (alpha={alpha:.3f}, beta={beta:.3f}, patience={patience})...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        max_epochs=epochs,
        patience=patience,
        device=device,
        metrics_fn=grid_flattened_metrics_fn,
    )

    best_idx = history.get("best_epoch_idx", -1)
    val_loss = history.get("best_val_loss", history["val_loss"][best_idx])
    val_metrics = {
        "loss": val_loss,
        "rmse": history["val_rmse"][best_idx],
        "ssim": history["val_ssim"][best_idx],
        "dice": history["val_dice"][best_idx],
        "iou": history["val_iou"][best_idx],
        "pe": history["val_pe"][best_idx],
    }

    print("\n--- Métricas de Validación (Hold-out 20%) ---")
    for k, v in val_metrics.items():
        print(f"  {k.upper()}: {v:.4f}")

    # Guardar modelo entrenado
    model_path = output_dir / "best_dnn_hybrid_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\n✅ Pesos del modelo guardados en: {model_path}")

    # Evaluación en Test ID (muestra a muestra: media ± std)
    test_id_summary = None
    if test_id_dataset and len(test_id_dataset) > 0:
        print("\n--- Evaluación en Test ID (Misma Distribución) ---")
        test_id_summary = evaluate_dataset_samplewise(model, test_id_dataset, device=device)
        if test_id_summary:
            for k, (m_val, s_val) in test_id_summary.items():
                print(f"  {k.upper()}: {m_val:.4f} ± {s_val:.4f}")

    # Evaluación en Test OOD (muestra a muestra: media ± std)
    test_ood_summary = None
    if test_ood_dataset and len(test_ood_dataset) > 0:
        print("\n--- Evaluación en Test OOD (Topologías Complejas) ---")
        test_ood_summary = evaluate_dataset_samplewise(model, test_ood_dataset, device=device)
        if test_ood_summary:
            for k, (m_val, s_val) in test_ood_summary.items():
                print(f"  {k.upper()}: {m_val:.4f} ± {s_val:.4f}")

    log_experiment_result(
        script="scripts/training/dnn/dnn_hybrid/evaluate_dnn_hybrid.py",
        model_name="DynamicDNN (Hybrid)",
        loss_type="hybrid",
        train_dataset=train_dir,
        test_id_dataset=test_id_dir if test_id_summary else None,
        test_ood_dataset=test_ood_dir if test_ood_summary else None,
        val_metrics=val_metrics,
        test_id_metrics=test_id_summary,
        test_ood_metrics=test_ood_summary,
        hyperparameters=best_params,
    )


def main():
    default_params = Path(__file__).resolve().parent / "best_params_dnn_hybrid.json"
    parser = argparse.ArgumentParser(description="Evaluación de DNN con Hybrid Loss (RMSE + SSIM)")
    parser.add_argument("--train_dir", type=str, default="dataset/dataset_16e_all")
    parser.add_argument("--test_id_dir", type=str, default="dataset/dataset_16e_adjacent_test_id")
    parser.add_argument("--test_ood_dir", type=str, default="dataset/dataset_16e_adjacent_test_ood")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15, help="Paciencia de Early Stopping (default: 15)")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria para replicabilidad")
    parser.add_argument(
        "--params_file",
        type=str,
        default=str(default_params),
        help="Ruta al JSON de hiperparámetros",
    )
    args = parser.parse_args()

    import os
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(8)

    if not Path(args.params_file).exists():
        print(f"Aviso: {args.params_file} no existe. Usando configuración base.")
        best_params = {
            "n_layers": 3,
            "hidden_dim_l0": 512,
            "hidden_dim_l1": 512,
            "hidden_dim_l2": 256,
            "activation": "relu",
            "lr": 4e-4,
            "weight_decay": 1e-5,
            "dropout_rate": 0.05,
            "batch_size": 32,
            "alpha": 1.0,
            "beta": 5.0,
        }
    else:
        with open(args.params_file) as f:
            best_params = json.load(f)
    best_params["seed"] = args.seed

    print(f"Cargando Dataset Train: {args.train_dir}")
    train_dataset = EITDataset(PROJECT_ROOT / args.train_dir, target_type="grid", flatten_grid=True)

    test_id_dataset, test_ood_dataset = None, None
    if (PROJECT_ROOT / args.test_id_dir).exists():
        print(f"Cargando Dataset Test ID: {args.test_id_dir}")
        test_id_dataset = EITDataset(
            PROJECT_ROOT / args.test_id_dir, target_type="grid", flatten_grid=True
        )

    if (PROJECT_ROOT / args.test_ood_dir).exists():
        print(f"Cargando Dataset Test OOD: {args.test_ood_dir}")
        test_ood_dataset = EITDataset(
            PROJECT_ROOT / args.test_ood_dir, target_type="grid", flatten_grid=True
        )

    sample_x, sample_y, _ = train_dataset[0]
    input_dim = sample_x.numel()
    output_dim = sample_y.numel()

    run_evaluation_pipeline(
        train_dataset,
        test_id_dataset,
        test_ood_dataset,
        best_params,
        input_dim,
        output_dim,
        device,
        args.epochs,
        patience=args.patience,
        seed=args.seed,
        train_dir=args.train_dir,
        test_id_dir=args.test_id_dir,
        test_ood_dir=args.test_ood_dir,
    )


if __name__ == "__main__":
    main()
