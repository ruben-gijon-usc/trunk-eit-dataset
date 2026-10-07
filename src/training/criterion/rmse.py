import torch
import torch.nn.functional as F
from src.training.criterion.base import BaseEITLoss

class MaskedRMSELoss(BaseEITLoss):
    """
    Error Cuadrático Medio Raíz (RMSE) adaptado para EIT.
    Calcula el error físico real de la conductividad S/m solo dentro del área del tronco.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor = None) -> torch.Tensor:
        preds, targets, masks = self.apply_mask(preds, targets, masks)
        
        if masks is not None:
            # Calculamos MSE sin reducción para ignorar los ceros del exterior
            mse = F.mse_loss(preds, targets, reduction='none')
            # Promediamos estrictamente por el número de píxeles/elementos activos
            mse = (mse * masks).sum() / (masks.sum() + self.eps)
        else:
            mse = F.mse_loss(preds, targets)
            
        return torch.sqrt(mse + self.eps)
