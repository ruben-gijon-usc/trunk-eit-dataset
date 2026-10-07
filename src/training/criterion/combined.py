import torch

from scripts.data.config import MAX_CONDUCTIVITY
from src.training.criterion.base import BaseEITLoss
from src.training.criterion.rmse import MaskedRMSELoss
from src.training.criterion.ssim import MaskedSSIMLoss


class HybridEITLoss(BaseEITLoss):
    """
    Función de Pérdida Híbrida para problemas inversos de EIT 2D (Grid).
    Combina la fidelidad física estricta (RMSE) con la coherencia geométrica (SSIM).

    Fórmula: Loss = α * RMSE + β * (1 - SSIM)
    """

    def __init__(
        self, alpha: float = 1.0, beta: float = 10.0, data_range: float = MAX_CONDUCTIVITY
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.rmse_loss = MaskedRMSELoss()
        self.ssim_loss = MaskedSSIMLoss(data_range=data_range)

    def forward(
        self, preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor = None
    ) -> torch.Tensor:
        # NO llamamos a self.apply_mask() aquí porque las subclases (RMSE y SSIM)
        # se encargan de enmascarar y dar formato a sus propios tensores internamente.

        loss_rmse = self.rmse_loss(preds, targets, masks)
        loss_ssim = self.ssim_loss(preds, targets, masks)

        return (self.alpha * loss_rmse) + (self.beta * loss_ssim)
