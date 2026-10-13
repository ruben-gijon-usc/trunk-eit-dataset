import argparse
import json
import sys
from pathlib import Path

import optuna
import torch
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.config import MAX_CONDUCTIVITY
from src.data.eit_dataset import EITDataset
from src.training.arquitectures.dnn import DynamicDNN, grid_flattened_metrics_fn
from src.training.criterion.combined import HybridEITLoss
from src.training.train_utils import set_seed, train_model


def objective(trial, dataset, input_dim, output_dim, device, epochs, seed: int = 42):
    n_layers = trial.suggest_int("n_layers", 2, 5)

    hidden_dims = []
    for i in range(n_layers):
        hidden_dims.append(
            trial.suggest_categorical(f"hidden_dim_l{i}", [128, 256, 512, 1024, 2048])
        )

    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = DynamicDNN(input_dim, output_dim, hidden_dims, dropout_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = HybridEITLoss(alpha=1.0, beta=10.0, data_range=MAX_CONDUCTIVITY)

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
    return history.get("best_val_loss", history["val_loss"][-1])


def main():
    parser = argparse.ArgumentParser(
        description="Tuning de hiperparámetros para DNN con Hybrid Loss (RMSE + SSIM)"
    )
    parser.add_argument("--dataset_dir", type=str, default="dataset/dataset_16e_all")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria para replicabilidad")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(__file__).resolve().parent

    dataset = EITDataset(PROJECT_ROOT / args.dataset_dir, target_type="grid", flatten_grid=True)
    sample_x, sample_y, _ = dataset[0]
    input_dim = sample_x.numel()
    output_dim = sample_y.numel()

    print(f"Buscando hiperparámetros con Optuna para DNN (Seed: {args.seed}, Loss: Hybrid RMSE + SSIM)...")
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed)
    )
    study.optimize(
        lambda trial: objective(trial, dataset, input_dim, output_dim, device, args.epochs, seed=args.seed),
        n_trials=args.trials,
    )

    best_params = study.best_params
    best_params["loss_type"] = "hybrid"
    best_params["n_trials_ran"] = args.trials
    best_params["seed"] = args.seed

    out_file = output_dir / "best_params_dnn_hybrid.json"
    with open(out_file, "w") as f:
        json.dump(best_params, f, indent=4)

    print(f"✅ Búsqueda terminada. Parámetros guardados en: {out_file}")


if __name__ == "__main__":
    main()
