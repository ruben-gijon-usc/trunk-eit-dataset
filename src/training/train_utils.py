import collections
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from src.training.early_stopping import EarlyStopping
from src.training.metrics import MetricFunction, compute_all_metrics

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_epochs: int = 100,
    patience: int = 5,
    anomaly_threshold: float = 1.5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    metrics_fn: Optional[MetricFunction] = None
) -> Dict[str, Any]:
    """
    Función genérica para entrenar cualquier modelo de reconstrucción EIT usando métricas dinámicas.
    
    Args:
        model: Modelo de PyTorch a entrenar.
        train_loader: DataLoader con los datos de entrenamiento.
        val_loader: DataLoader con los datos de validación.
        criterion: Función de pérdida (ej: nn.MSELoss()).
        optimizer: Optimizador configurado para los pesos del modelo.
        max_epochs: Épocas máximas de entrenamiento.
        patience: Épocas de paciencia para el EarlyStopping.
        anomaly_threshold: Umbral para el cálculo de métricas (Precision, Recall, F1, etc).
        device: 'cuda' o 'cpu'. Autodetecta GPU si está disponible.
        metrics_fn: Función abstracta que recibe (preds, targets, threshold) y devuelve dict[str, float].
                    Si es None, usa 'compute_all_metrics' por defecto.
        
    Returns:
        Un diccionario (history) con la evolución de métricas y pérdidas en cada época.
    """
    model = model.to(device)
    early_stopper = EarlyStopping(patience=patience, restore_best_weights=True)
    
    if metrics_fn is None:
        metrics_fn = compute_all_metrics
        
    # Usamos defaultdict para registrar cualquier métrica arbitraria que devuelva metrics_fn
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
            
            # Puedes optar por enmascarar también el loss, aunque aquí calculamos el general.
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Cálculo dinámico de métricas custom por batch (pasando la máscara si existe)
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
        # Formatear un resumen de las métricas principales para imprimir
        t_f1 = history.get("train_f1", [0.0])[-1]
        v_f1 = history.get("val_f1", [0.0])[-1]
        v_acc = history.get("val_accuracy", [0.0])[-1]

        print(f"Epoch {epoch+1:02d}/{max_epochs} | "
              f"Train Loss: {train_loss:.6f} [F1: {t_f1:.4f}] | "
              f"Val Loss: {val_loss:.6f} [F1: {v_f1:.4f}, Acc: {v_acc:.4f}]")

        # Comprobar Early Stopping
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print(f"🛑 Early stopping activado. Entrenamiendo detenido en la época {epoch+1}.")
            break

    print("✅ Entrenamiento completado (se han restaurado los mejores pesos).")
    # Convertir a un dict normal antes de devolver
    return dict(history)
