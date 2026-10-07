import csv
import json
import argparse
from pathlib import Path
import optuna
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import KFold

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.eit_dataset import EITDataset
from src.training.train_utils import train_model
from src.training.criterion.combined import HybridEITLoss
from scripts.data.config import GROUND_TRUTH_THRESHOLD, MAX_CONDUCTIVITY

class DynamicDNN(nn.Module):
    """DNN parametrizable para Optuna (1D output para elem_data o grid flattened)."""
    def __init__(self, input_dim: int, output_dim: int, n_layers: int, hidden_dim: int, dropout_rate: float):
        super().__init__()
        layers = []
        in_d = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_d = hidden_dim
            
        layers.append(nn.Linear(in_d, output_dim))
        layers.append(nn.ReLU()) # La conductividad es siempre > 0
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

from src.training.metrics import EITMetricsEvaluator

def grid_flattened_metrics_fn(preds: torch.Tensor, targets: torch.Tensor, threshold: float = GROUND_TRUTH_THRESHOLD, mask: torch.Tensor = None) -> dict:
    """Calcula TODAS las métricas espaciales reformateando el grid 1D a 2D."""
    B = preds.size(0)
    # Reshape de (B, 4096) a (B, 1, 64, 64)
    p_2d = preds.view(B, 1, 64, 64)
    t_2d = targets.view(B, 1, 64, 64)
    m_2d = mask.view(B, 1, 64, 64) if mask is not None else None

    evaluator = EITMetricsEvaluator(threshold=threshold, device=preds.device)
    
    rmse_val = evaluator.rmse(p_2d, t_2d, m_2d).item()
    ssim_val = evaluator.ssim(p_2d, t_2d, m_2d).item()
    dice, iou = evaluator.dice_and_iou(p_2d, t_2d, m_2d)
    pe = evaluator.position_error(p_2d, t_2d, m_2d).item()
    
    return {
        "rmse": rmse_val,
        "ssim": ssim_val,
        "dice": dice.item(),
        "iou": iou.item(),
        "pe": pe
    }

def objective(trial, dataset, input_dim, output_dim, device, epochs):
    n_layers = trial.suggest_int("n_layers", 2, 5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512, 1024])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = DynamicDNN(input_dim, output_dim, n_layers, hidden_dim, dropout_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 💥 APLICAMOS LA NUEVA LOSS HÍBRIDA 💥
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
        metrics_fn=grid_flattened_metrics_fn
    )
    return history.get("best_val_loss", history["val_loss"][-1])

def run_5_fold_cv(dataset, best_params, input_dim, output_dim, device, epochs):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Estructura para almacenar TODAS las métricas de cada fold
    fold_results = {
        "val_loss": [], "val_rmse": [], "val_ssim": [], 
        "val_dice": [], "val_iou": [], "val_pe": []
    }

    print("\n" + "="*50)
    print("Iniciando 5-Fold Cross Validation con hiperparámetros óptimos:")
    print("="*50 + "\n")

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f"\n--- FOLD {fold + 1}/5 ---")
        train_sub = Subset(dataset, train_ids)
        val_sub = Subset(dataset, val_ids)
        
        train_loader = DataLoader(train_sub, batch_size=best_params["batch_size"], shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=best_params["batch_size"], shuffle=False)
        
        model = DynamicDNN(
            input_dim, output_dim, 
            best_params["n_layers"], best_params["hidden_dim"], best_params["dropout_rate"]
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
        
        # 💥 APLICAMOS LA NUEVA LOSS HÍBRIDA 💥
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
            metrics_fn=grid_flattened_metrics_fn
        )
        
        fold_results["val_loss"].append(history.get("best_val_loss", history["val_loss"][-1]))
        for m in ["val_rmse", "val_ssim", "val_dice", "val_iou", "val_pe"]:
            if m in history:
                fold_results[m].append(history[m][-1])

    # ---------------------------------------------------------
    # Guardar métricas del 5-Fold CV
    # ---------------------------------------------------------
    metrics_csv_path = PROJECT_ROOT / "dnn_5fold_metrics.csv"
    with open(metrics_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Mean", "Std"])
        for k, v in fold_results.items():
            if v:
                mean_val = np.mean(v)
                std_val = np.std(v)
                writer.writerow([k, f"{mean_val:.6f}", f"{std_val:.6f}"])
                print(f"{k} (5-Fold CV): {mean_val:.4f} ± {std_val:.4f}")

    print(f"\n✅ Métricas de CV guardadas en: {metrics_csv_path}")
    
    # ---------------------------------------------------------
    # Entrenar Modelo Final Completo (Reconstrucción futura)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("Entrenando modelo final con el 100% del dataset...")
    
    full_loader = DataLoader(dataset, batch_size=best_params["batch_size"], shuffle=True)
    
    final_model = DynamicDNN(
        input_dim, output_dim, 
        best_params["n_layers"], best_params["hidden_dim"], best_params["dropout_rate"]
    )
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    criterion = HybridEITLoss(alpha=1.0, beta=10.0, data_range=MAX_CONDUCTIVITY)
    
    # Entrenamos por un número fijo de épocas ya que no hay early stopping sin validación
    train_model(
        model=final_model,
        train_loader=full_loader,
        val_loader=full_loader, # Usamos train loader de val solo para compatibilidad del código
        criterion=criterion,
        optimizer=optimizer,
        max_epochs=epochs, 
        patience=epochs, 
        device=device,
        metrics_fn=grid_flattened_metrics_fn
    )
    
    model_path = PROJECT_ROOT / "best_dnn_model.pt"
    torch.save(final_model.state_dict(), model_path)
    print(f"✅ Pesos del modelo final listos para inferencia guardados en: {model_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="dataset/dataset_16e_all")
    parser.add_argument("--trials", type=int, default=10, help="Número de trials de Optuna")
    parser.add_argument("--epochs", type=int, default=50, help="Épocas máximas")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # IMPORTANTE: Cambiado a target_type="grid" y flatten_grid=True
    # porque el dataset actual tiene mallas FEM de tamaño variable (ej. 3134 vs 3243)
    dataset = EITDataset(PROJECT_ROOT / args.dataset_dir, target_type="grid", flatten_grid=True)
    if len(dataset) == 0:
        print("Dataset vacío o no se encontró el target especificado.")
        return

    sample_x, sample_y, _ = dataset[0]
    input_dim = sample_x.numel()
    output_dim = sample_y.numel()
    
    print(f"Input dim (Voltage): {input_dim}, Output dim (Grid Flattened): {output_dim}")

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, dataset, input_dim, output_dim, device, args.epochs), n_trials=args.trials)

    print("\nMejores parámetros encontrados:")
    best_params = study.best_params
    
    csv_path = PROJECT_ROOT / "optuna_dnn_results.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        param_keys = list(best_params.keys())
        headers = ["trial_number", "val_loss", "state"] + param_keys
        writer.writerow(headers)
        
        for trial in study.trials:
            row = [trial.number, trial.value, trial.state.name]
            for k in param_keys:
                row.append(trial.params.get(k, ""))
            writer.writerow(row)
            
    print(f"💾 Resultados de combinaciones Optuna guardados en: {csv_path}")

    # Lanza el CV y guarda modelo
    run_5_fold_cv(dataset, best_params, input_dim, output_dim, device, args.epochs)

if __name__ == "__main__":
    main()
