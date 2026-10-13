import collections
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scripts.data.config import GROUND_TRUTH_THRESHOLD
from src.training.arquitectures.cnn import cnn_metrics_fn
from src.training.early_stopping import EarlyStopping


class ConvEncoder(nn.Module):
    """
    Forward Surrogate Encoder:
    Maps 2D conductivity distribution (1, 64, 64) -> predicted boundary voltages (voltage_dim, e.g. 208).
    """

    def __init__(
        self,
        voltage_dim: int = 208,
        base_channels: int = 32,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        c = base_channels

        self.net = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(1, c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
            # 32x32 -> 16x16
            nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
            # 16x16 -> 8x8
            nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
            # 8x8 -> 4x4
            nn.Conv2d(c * 4, c * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(c * 4 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, voltage_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvDecoder(nn.Module):
    """
    Inverse Solver Decoder:
    Maps boundary voltage measurements (voltage_dim, e.g. 208) -> 2D conductivity distribution (1, 64, 64).
    """

    def __init__(
        self,
        voltage_dim: int = 208,
        base_channels: int = 128,
        output_channels: int = 1,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.init_h = 8
        self.init_w = 8

        self.fc = nn.Sequential(
            nn.Linear(voltage_dim, base_channels * self.init_h * self.init_w),
            nn.BatchNorm1d(base_channels * self.init_h * self.init_w),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        c1 = base_channels
        c2 = max(16, base_channels // 2)
        c3 = max(16, c2 // 2)

        self.deconv = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(c1, c2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(c2, c3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(c3, c3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            # Final refinement
            nn.Conv2d(c3, output_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # Physics constraint: non-negative conductivity
        )

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        feat = self.fc(v)
        feat = feat.view(-1, self.base_channels, self.init_h, self.init_w)
        return self.deconv(feat)


class EITAutoencoder(nn.Module):
    """
    Physics-informed EIT Autoencoder:
      - Encoder: Conductivity Image (1, 64, 64) -> Predicted Voltages V_hat (voltage_dim)
      - Decoder: Voltages V (voltage_dim) -> Reconstructed Conductivity Image sigma_hat (1, 64, 64)

    Inference (Inverse Problem):
      Directly invoke `reconstruct_from_voltages(measured_voltages)` to predict conductivity from measurements.
    """

    def __init__(
        self,
        voltage_dim: int = 208,
        encoder_channels: int = 32,
        decoder_channels: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.encoder = ConvEncoder(
            voltage_dim=voltage_dim, base_channels=encoder_channels, dropout_rate=dropout_rate
        )
        self.decoder = ConvDecoder(
            voltage_dim=voltage_dim, base_channels=decoder_channels, dropout_rate=dropout_rate
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v_pred = self.encoder(images)
        img_recon = self.decoder(v_pred)
        return v_pred, img_recon

    def reconstruct_from_voltages(self, voltages: torch.Tensor) -> torch.Tensor:
        """Inverse solver evaluation from measured boundary voltages."""
        return self.decoder(voltages)


def train_autoencoder(
    model: EITAutoencoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    img_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_epochs: int = 100,
    patience: int = 10,
    voltage_weight: float = 1.0,
    anomaly_threshold: float = GROUND_TRUTH_THRESHOLD,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    metrics_fn: Callable = cnn_metrics_fn,
) -> dict[str, Any]:
    """
    Training loop for EITAutoencoder.
    Optimizes:
      Loss_total = Loss_img(img_recon, targets) + voltage_weight * Loss_v(v_pred, voltages)
    """
    model = model.to(device)
    early_stopper = EarlyStopping(patience=patience, restore_best_weights=True)
    history = collections.defaultdict(list)

    print(f"Iniciando entrenamiento Autoencoder en {device} (paciencia={patience})...")

    for epoch in range(max_epochs):
        # ----------------------
        # Fase de Entrenamiento
        # ----------------------
        model.train()
        train_loss = 0.0
        train_metrics_sum = collections.defaultdict(float)

        for batch_data in train_loader:
            voltages = batch_data[0].to(device)
            targets = batch_data[1].to(device)
            masks = batch_data[2].to(device) if len(batch_data) > 2 else None

            optimizer.zero_grad()
            v_pred, img_recon = model(targets)

            # Voltage loss (RMSE)
            loss_v = torch.sqrt(F.mse_loss(v_pred, voltages) + 1e-8)

            # Image loss (Masked RMSE or Masked Hybrid)
            if masks is not None:
                preds_masked = img_recon * masks
                targets_masked = targets * masks
                try:
                    loss_img = img_criterion(preds_masked, targets_masked, masks)
                except TypeError:
                    loss_img = img_criterion(preds_masked, targets_masked)
            else:
                loss_img = img_criterion(img_recon, targets)

            total_loss = loss_img + voltage_weight * loss_v
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            batch_metrics = metrics_fn(img_recon, targets, threshold=anomaly_threshold, mask=masks)
            for k, v in batch_metrics.items():
                train_metrics_sum[k] += v

        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)
        for k, v in train_metrics_sum.items():
            history[f"train_{k}"].append(v / len(train_loader))

        # ----------------------
        # Fase de Validación (Evaluación en el modo inverso puro desde voltajes reales)
        # ----------------------
        model.eval()
        val_loss = 0.0
        val_metrics_sum = collections.defaultdict(float)

        with torch.no_grad():
            for batch_data in val_loader:
                voltages = batch_data[0].to(device)
                targets = batch_data[1].to(device)
                masks = batch_data[2].to(device) if len(batch_data) > 2 else None

                # En validación medimos cómo reconstruye el Decoder a partir de los voltajes medidos
                img_recon = model.reconstruct_from_voltages(voltages)

                if masks is not None:
                    preds_masked = img_recon * masks
                    targets_masked = targets * masks
                    try:
                        loss_img = img_criterion(preds_masked, targets_masked, masks)
                    except TypeError:
                        loss_img = img_criterion(preds_masked, targets_masked)
                else:
                    loss_img = img_criterion(img_recon, targets)

                val_loss += loss_img.item()
                batch_metrics = metrics_fn(
                    img_recon, targets, threshold=anomaly_threshold, mask=masks
                )
                for k, v in batch_metrics.items():
                    val_metrics_sum[k] += v

        val_loss /= len(val_loader)
        history["val_loss"].append(val_loss)
        for k, v in val_metrics_sum.items():
            history[f"val_{k}"].append(v / len(val_loader))

        v_metric = f"[RMSE: {history['val_rmse'][-1]:.4f} | SSIM: {history['val_ssim'][-1]:.4f}]"
        print(
            f"Epoch {epoch + 1:02d}/{max_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} {v_metric}"
        )

        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print(f"🛑 Early stopping activado en la época {epoch + 1}.")
            break

    if not early_stopper.early_stop:
        early_stopper.restore_best_model(model)

    print("✅ Entrenamiento completado (mejores pesos restaurados).")
    history["best_val_loss"] = early_stopper.best_value
    history["best_epoch_idx"] = history["val_loss"].index(min(history["val_loss"]))
    return dict(history)
