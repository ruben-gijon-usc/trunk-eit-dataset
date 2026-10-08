import sys
from pathlib import Path

import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.config import GROUND_TRUTH_THRESHOLD, MAX_CONDUCTIVITY
from src.training.criterion.combined import HybridEITLoss
from src.training.metrics import EITMetricsEvaluator


class DynamicDNN(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, hidden_dims: list[int], dropout_rate: float
    ):
        super().__init__()
        layers = []
        in_d = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_d, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_d = h_dim

        layers.append(nn.Linear(in_d, output_dim))
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def grid_flattened_metrics_fn(preds, targets, threshold=GROUND_TRUTH_THRESHOLD, mask=None):
    if preds.dim() == 2:
        B = preds.size(0)
        size = int(preds.size(1) ** 0.5)
        p_2d = preds.view(B, 1, size, size)
        t_2d = targets.view(B, 1, size, size)
        m_2d = mask.view(B, 1, size, size) if mask is not None else None
    else:
        p_2d, t_2d, m_2d = preds, targets, mask

    evaluator = EITMetricsEvaluator(threshold=threshold, data_range=MAX_CONDUCTIVITY)
    rmse_val = evaluator.rmse(p_2d, t_2d, m_2d).item()
    ssim_val = evaluator.ssim(p_2d, t_2d, m_2d).item()
    dice, iou = evaluator.dice_and_iou(p_2d, t_2d, m_2d)
    pe = evaluator.position_error(p_2d, t_2d, m_2d).item()

    return {"rmse": rmse_val, "ssim": ssim_val, "dice": dice.item(), "iou": iou.item(), "pe": pe}


def get_criterion(loss_type: str):
    if loss_type.lower() == "hybrid":
        return HybridEITLoss(alpha=1.0, beta=10.0, data_range=MAX_CONDUCTIVITY)
    else:
        return nn.MSELoss()
