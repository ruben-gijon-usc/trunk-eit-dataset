import collections
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scripts.data.config import GROUND_TRUTH_THRESHOLD
from src.training.early_stopping import EarlyStopping


def default_metrics_fn(preds, targets, threshold=GROUND_TRUTH_THRESHOLD, mask=None):
    return {}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_epochs: int = 100,
    patience: int = 5,
    anomaly_threshold: float = GROUND_TRUTH_THRESHOLD,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    metrics_fn: Callable | None = None,
) -> dict[str, Any]:
    """
    Función genérica para entrenar cualquier modelo de reconstrucción EIT usando métricas dinámicas.
    Ahora incorpora Strict Masking en la Función de Pérdida (Agent Pitfall #5).
    """
    model = model.to(device)
    early_stopper = EarlyStopping(patience=patience, restore_best_weights=True)

    if metrics_fn is None:
        metrics_fn = default_metrics_fn

    history = collections.defaultdict(list)
    print(f"Iniciando entrenamiento en {device} con Early Stopping (paciencia={patience})...")

    for epoch in range(max_epochs):
        # ----------------------
        # Fase de Entrenamiento
        # ----------------------
        model.train()
        train_loss = 0.0
        train_metrics_sum = collections.defaultdict(float)

        for batch_data in train_loader:
            voltages, targets = batch_data[0].to(device), batch_data[1].to(device)
            masks = batch_data[2].to(device) if len(batch_data) > 2 else None

            optimizer.zero_grad()
            preds = model(voltages)

            # STRICT MASKING ON LOSS (AGENTS.md)
            if masks is not None:
                # Si el modelo saca (B, C, H, W) y target es (B, H, W), ajustamos dimensiones si hace falta
                # Asumimos que preds y targets tienen la misma dimensión espacial que mask.
                # Multiplicamos ambas por la máscara binaria (1 = madera, 0 = aire).
                preds_masked = preds * masks
                targets_masked = targets * masks
                try:
                    loss = criterion(preds_masked, targets_masked, masks)
                except TypeError:
                    loss = criterion(preds_masked, targets_masked)
            else:
                loss = criterion(preds, targets)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Cálculo dinámico de métricas custom por batch
            batch_metrics = metrics_fn(preds, targets, threshold=anomaly_threshold, mask=masks)
            for k, v in batch_metrics.items():
                train_metrics_sum[k] += v

        # Promediar entrenamiento
        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)
        for k, v in train_metrics_sum.items():
            history[f"train_{k}"].append(v / len(train_loader))

        # ----------------------
        # Fase de Validación
        # ----------------------
        model.eval()
        val_loss = 0.0
        val_metrics_sum = collections.defaultdict(float)

        with torch.no_grad():
            for batch_data in val_loader:
                voltages, targets = batch_data[0].to(device), batch_data[1].to(device)
                masks = batch_data[2].to(device) if len(batch_data) > 2 else None

                preds = model(voltages)

                # STRICT MASKING ON LOSS (AGENTS.md)
                if masks is not None:
                    preds_masked = preds * masks
                    targets_masked = targets * masks
                    try:
                        loss = criterion(preds_masked, targets_masked, masks)
                    except TypeError:
                        loss = criterion(preds_masked, targets_masked)
                else:
                    loss = criterion(preds, targets)

                val_loss += loss.item()

                batch_metrics = metrics_fn(preds, targets, threshold=anomaly_threshold, mask=masks)
                for k, v in batch_metrics.items():
                    val_metrics_sum[k] += v

        # Promediar validación
        val_loss /= len(val_loader)
        history["val_loss"].append(val_loss)
        for k, v in val_metrics_sum.items():
            history[f"val_{k}"].append(v / len(val_loader))

        # ----------------------
        # Registro e Impresión
        # ----------------------
        # Buscar métricas relevantes (RMSE, SSIM, F1)
        t_metric = ""
        v_metric = ""
        if "train_rmse" in history:
            t_metric = f"[RMSE: {history['train_rmse'][-1]:.4f}]"
            v_metric = f"[RMSE: {history['val_rmse'][-1]:.4f}]"
        elif "train_ssim" in history:
            t_metric = f"[SSIM: {history['train_ssim'][-1]:.4f}]"
            v_metric = f"[SSIM: {history['val_ssim'][-1]:.4f}]"

        print(
            f"Epoch {epoch + 1:02d}/{max_epochs} | "
            f"Train Loss: {train_loss:.6f} {t_metric} | "
            f"Val Loss: {val_loss:.6f} {v_metric}"
        )

        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print(f"🛑 Early stopping activado. Entrenamiendo detenido en la época {epoch + 1}.")
            break

    print("✅ Entrenamiento completado (se han restaurado los mejores pesos).")
    history["best_val_loss"] = early_stopper.best_value
    return dict(history)
