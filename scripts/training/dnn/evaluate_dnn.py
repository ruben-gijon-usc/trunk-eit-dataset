import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.dnn.dnn_utils import DynamicDNN, get_criterion, grid_flattened_metrics_fn
from src.data.eit_dataset import EITDataset
from src.training.train_utils import train_model


def run_5_fold_cv(dataset, best_params, input_dim, output_dim, device, epochs, loss_type):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    output_dir = Path(__file__).resolve().parent

    fold_results = {
        "val_loss": [],
        "val_rmse": [],
        "val_ssim": [],
        "val_dice": [],
        "val_iou": [],
        "val_pe": [],
    }

    print("\n" + "=" * 50)
    print("Iniciando 5-Fold Cross Validation con hiperparámetros:")
    print(json.dumps(best_params, indent=2))
    print("=" * 50 + "\n")

    n_layers = best_params["n_layers"]
    hidden_dims = [best_params[f"hidden_dim_l{i}"] for i in range(n_layers)]

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f"\n--- FOLD {fold + 1}/5 ---")
        train_sub = Subset(dataset, train_ids)
        val_sub = Subset(dataset, val_ids)

        train_loader = DataLoader(train_sub, batch_size=best_params["batch_size"], shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=best_params["batch_size"], shuffle=False)

        model = DynamicDNN(input_dim, output_dim, hidden_dims, best_params["dropout_rate"])
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"]
        )
        criterion = get_criterion(loss_type)

        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            max_epochs=epochs,
            patience=10,
            device=device,
            metrics_fn=grid_flattened_metrics_fn,
        )

        fold_results["val_loss"].append(history.get("best_val_loss", history["val_loss"][-1]))
        for m in ["val_rmse", "val_ssim", "val_dice", "val_iou", "val_pe"]:
            if m in history:
                fold_results[m].append(history[m][-1])

    metrics_csv_path = output_dir / f"dnn_{loss_type}_5fold_metrics.csv"
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

    full_loader = DataLoader(dataset, batch_size=best_params["batch_size"], shuffle=True)

    final_model = DynamicDNN(input_dim, output_dim, hidden_dims, best_params["dropout_rate"])
    optimizer = torch.optim.AdamW(
        final_model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"]
    )
    criterion = get_criterion(loss_type)

    train_model(
        model=final_model,
        train_loader=full_loader,
        val_loader=full_loader,
        criterion=criterion,
        optimizer=optimizer,
        max_epochs=epochs,
        patience=15,
        device=device,
        metrics_fn=grid_flattened_metrics_fn,
    )

    model_path = output_dir / f"best_dnn_{loss_type}_model.pt"
    torch.save(final_model.state_dict(), model_path)
    print(f"✅ Pesos del modelo final listos guardados en: {model_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="dataset/dataset_16e_all")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--params_file", type=str, required=True, help="Ruta al JSON de hiperparámetros"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.params_file) as f:
        best_params = json.load(f)

    loss_type = best_params.get("loss_type", "hybrid")

    dataset = EITDataset(PROJECT_ROOT / args.dataset_dir, target_type="grid", flatten_grid=True)
    sample_x, sample_y, _ = dataset[0]
    input_dim = sample_x.numel()
    output_dim = sample_y.numel()

    run_5_fold_cv(dataset, best_params, input_dim, output_dim, device, args.epochs, loss_type)


if __name__ == "__main__":
    main()
