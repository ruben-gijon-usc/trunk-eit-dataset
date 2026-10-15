import torch.nn as nn

from scripts.data.config import GROUND_TRUTH_THRESHOLD, MAX_CONDUCTIVITY
from src.training.metrics import EITMetricsEvaluator


class DynamicDNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        dropout_rate: float,
        activation: str = "relu",
    ):
        super().__init__()
        act_map = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "leaky_relu": lambda: nn.LeakyReLU(0.1),
        }
        act_fn = act_map.get(activation.lower(), nn.ReLU)

        layers = []
        in_d = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_d, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_fn())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            in_d = h_dim

        layers.append(nn.Linear(in_d, output_dim))
        layers.append(nn.ReLU())  # Conductividades físicas son estrictamente >= 0
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.net(x)


def grid_flattened_metrics_fn(
    preds, targets, threshold=GROUND_TRUTH_THRESHOLD, mask=None, threshold_mode="half_max"
):
    if preds.dim() == 2:
        B = preds.size(0)
        size = int(preds.size(1) ** 0.5)
        p_2d = preds.view(B, 1, size, size)
        t_2d = targets.view(B, 1, size, size)
        m_2d = mask.view(B, 1, size, size) if mask is not None else None
    else:
        p_2d, t_2d, m_2d = preds, targets, mask

    evaluator = EITMetricsEvaluator(
        threshold=threshold, data_range=MAX_CONDUCTIVITY, threshold_mode=threshold_mode
    )
    rmse_val = evaluator.rmse(p_2d, t_2d, m_2d).item()
    ssim_val = evaluator.ssim(p_2d, t_2d, m_2d).item()
    dice, iou = evaluator.dice_and_iou(p_2d, t_2d, m_2d)
    pe = evaluator.position_error(p_2d, t_2d, m_2d).item()

    return {"rmse": rmse_val, "ssim": ssim_val, "dice": dice.item(), "iou": iou.item(), "pe": pe}
