import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.eit_dataset import EITDataset
from src.training.arquitectures.autoencoder import EITAutoencoder, train_autoencoder
from src.training.criterion.rmse import MaskedRMSELoss
from src.training.experiment_logger import log_experiment_result
from src.training.metrics import evaluate_dataset_samplewise
from src.training.train_utils import set_seed


def run_evaluation_pipeline(
    train_dataset,
    test_id_dataset,
    test_ood_dataset,
    best_params,
    voltage_dim,
    device,
    epochs,
    seed: int = 42,
    train_dir="",
    test_id_dir="",
    test_ood_dir="",
):
    output_dir = Path(__file__).resolve().parent

    print("\n" + "=" * 50)
    print(f"Iniciando Evaluación [Autoencoder - RMSE] (Seed: {seed}):")
    print(json.dumps(best_params, indent=2))
    print("=" * 50 + "\n")

    val_size = int(len(train_dataset) * 0.2)
    train_size = len(train_dataset) - val_size
    train_sub, val_sub = random_split(
        train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_sub, batch_size=best_params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=best_params["batch_size"], shuffle=False)

    model = EITAutoencoder(
        voltage_dim=voltage_dim,
        enc_channels=best_params.get("enc_channels", 32),
        dec_channels=best_params.get("dec_channels", 32),
        dropout_rate=best_params.get("dropout_rate", 0.1),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"]
    )
    decoder_criterion = MaskedRMSELoss()

    print(f"Entrenando modelo en {train_size} muestras, validando en {val_size} muestras...")
    history = train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        decoder_criterion=decoder_criterion,
        optimizer=optimizer,
        max_epochs=epochs,
        patience=10,
        device=device,
        w_enc=1.0,
        w_dec=1.0,
    )

    best_idx = history.get("best_epoch_idx", -1)
    val_loss = history.get("best_val_loss", history["val_total_loss"][best_idx])
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
    model_path = output_dir / "best_autoencoder_rmse_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\n✅ Pesos del modelo guardados en: {model_path}")

    # Evaluación en Test ID (muestra a muestra: media ± std)
    test_id_summary = None
    if test_id_dataset and len(test_id_dataset) > 0:
        print("\n--- Evaluación en Test ID (Misma Distribución) ---")
        test_id_summary = evaluate_dataset_samplewise(
            model, test_id_dataset, device=device, is_autoencoder=True
        )
        if test_id_summary:
            for k, (m_val, s_val) in test_id_summary.items():
                print(f"  {k.upper()}: {m_val:.4f} ± {s_val:.4f}")

    # Evaluación en Test OOD (muestra a muestra: media ± std)
    test_ood_summary = None
    if test_ood_dataset and len(test_ood_dataset) > 0:
        print("\n--- Evaluación en Test OOD (Topologías Complejas) ---")
        test_ood_summary = evaluate_dataset_samplewise(
            model, test_ood_dataset, device=device, is_autoencoder=True
        )
        if test_ood_summary:
            for k, (m_val, s_val) in test_ood_summary.items():
                print(f"  {k.upper()}: {m_val:.4f} ± {s_val:.4f}")

    log_experiment_result(
        script="scripts/training/autoencoder/autoencoder_rmse/evaluate_autoencoder_rmse.py",
        model_name="EITAutoencoder (RMSE)",
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
    default_params = Path(__file__).resolve().parent / "best_params_autoencoder_rmse.json"
    parser = argparse.ArgumentParser(description="Evaluación de Autoencoder con RMSE Loss")
    parser.add_argument("--train_dir", type=str, default="dataset/dataset_16e_all")
    parser.add_argument("--test_id_dir", type=str, default="dataset/dataset_16e_adjacent_test_id")
    parser.add_argument("--test_ood_dir", type=str, default="dataset/dataset_16e_adjacent_test_ood")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria para replicabilidad")
    parser.add_argument(
        "--params_file",
        type=str,
        default=str(default_params),
        help="Ruta al JSON de hiperparámetros",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not Path(args.params_file).exists():
        print(f"Aviso: {args.params_file} no existe. Usando configuración base.")
        best_params = {
            "enc_channels": 32,
            "dec_channels": 32,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "dropout_rate": 0.1,
            "batch_size": 32,
        }
    else:
        with open(args.params_file) as f:
            best_params = json.load(f)
    best_params["seed"] = args.seed

    print(f"Cargando Dataset Train: {args.train_dir}")
    train_dataset = EITDataset(
        PROJECT_ROOT / args.train_dir, target_type="grid", flatten_grid=False
    )

    test_id_dataset, test_ood_dataset = None, None
    if (PROJECT_ROOT / args.test_id_dir).exists():
        print(f"Cargando Dataset Test ID: {args.test_id_dir}")
        test_id_dataset = EITDataset(
            PROJECT_ROOT / args.test_id_dir, target_type="grid", flatten_grid=False
        )

    if (PROJECT_ROOT / args.test_ood_dir).exists():
        print(f"Cargando Dataset Test OOD: {args.test_ood_dir}")
        test_ood_dataset = EITDataset(
            PROJECT_ROOT / args.test_ood_dir, target_type="grid", flatten_grid=False
        )

    sample_x, _, _ = train_dataset[0]
    voltage_dim = sample_x.numel()

    run_evaluation_pipeline(
        train_dataset,
        test_id_dataset,
        test_ood_dataset,
        best_params,
        voltage_dim,
        device,
        args.epochs,
        seed=args.seed,
        train_dir=args.train_dir,
        test_id_dir=args.test_id_dir,
        test_ood_dir=args.test_ood_dir,
    )


if __name__ == "__main__":
    main()
