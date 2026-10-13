import torch
import torch.nn as nn

from scripts.data.config import GROUND_TRUTH_THRESHOLD, MAX_CONDUCTIVITY
from src.training.metrics import EITMetricsEvaluator


class DynamicCNN(nn.Module):
    """
    Convolutional Decoder Network for EIT Inverse Problem.
    Projects 1D boundary voltage measurements into a 2D spatial latent representation,
    then progressively upsamples via Transposed Convolutions to reconstruct
    the 64x64 internal conductivity image.
    """

    def __init__(
        self,
        input_dim: int,
        base_channels: int = 128,
        output_channels: int = 1,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.base_channels = base_channels
        self.init_h = 8
        self.init_w = 8

        # Project 1D voltage -> spatial feature volume (base_channels, 8, 8)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, base_channels * self.init_h * self.init_w),
            nn.BatchNorm1d(base_channels * self.init_h * self.init_w),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        c1 = base_channels
        c2 = max(16, base_channels // 2)
        c3 = max(16, c2 // 2)

        # Transposed Convolutional Decoder: 8x8 -> 16x16 -> 32x32 -> 64x64
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
            # Refinement convolution (64x64 -> 64x64)
            nn.Conv2d(c3, output_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # Physics constraint: non-negative conductivity (>= 0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.fc(x)
        feat = feat.view(-1, self.base_channels, self.init_h, self.init_w)
        return self.deconv(feat)


def cnn_metrics_fn(preds, targets, threshold=GROUND_TRUTH_THRESHOLD, mask=None):
    evaluator = EITMetricsEvaluator(threshold=threshold, data_range=MAX_CONDUCTIVITY)
    rmse_val = evaluator.rmse(preds, targets, mask).item()
    ssim_val = evaluator.ssim(preds, targets, mask).item()
    dice, iou = evaluator.dice_and_iou(preds, targets, mask)
    pe = evaluator.position_error(preds, targets, mask).item()

    return {"rmse": rmse_val, "ssim": ssim_val, "dice": dice.item(), "iou": iou.item(), "pe": pe}
