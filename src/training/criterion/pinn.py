import torch
from src.training.criterion.base import BaseEITLoss

class TVLoss(BaseEITLoss):
    """
    Pérdida Physics-Informed (Variación Total / Total Variation).
    Fuerza a la matriz de conductividad S/m predicha a ser espacialmente continua.
    Penaliza gradientes muy altos entre píxeles adyacentes para reducir el ruido 
    (artifacts) típico en los problemas inversos mal condicionados como el EIT.
    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor = None, masks: torch.Tensor = None) -> torch.Tensor:
        # Aquí 'targets' no es necesario porque es una regularización directa sobre la predicción.
        preds, _, masks = self.apply_mask(preds, None, masks)
        
        # Diferencias a lo largo del eje Y (Altura) y X (Anchura)
        h_diff = preds[:, :, 1:, :] - preds[:, :, :-1, :]
        w_diff = preds[:, :, :, 1:] - preds[:, :, :, :-1]
        
        if masks is not None:
            # Máscaras desplazadas para coincidir con h_diff y w_diff
            h_mask = masks[:, :, 1:, :] * masks[:, :, :-1, :]
            w_mask = masks[:, :, :, 1:] * masks[:, :, :, :-1]
            
            # Aplicamos máscara booleana a los diferenciales
            h_diff = h_diff * h_mask
            w_diff = w_diff * w_mask

        # Variación total absoluta
        tv = torch.sum(torch.abs(h_diff)) + torch.sum(torch.abs(w_diff))
        
        # Normalizamos por el tamaño del batch
        batch_size = preds.size(0)
        
        return self.weight * (tv / batch_size)
