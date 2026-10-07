import torch
from torchmetrics.functional.image import structural_similarity_index_measure

from scripts.data.config import MAX_CONDUCTIVITY
from src.training.criterion.base import BaseEITLoss


class MaskedSSIMLoss(BaseEITLoss):
    """
    Pérdida basada en Similitud Estructural (SSIM).
    Obliga a la red neuronal a generar geometrías y fronteras de anomalías realistas,
    evaluando la coherencia espacial 2D de la madera y las descomposiciones.
    """

    def __init__(self, data_range: float = MAX_CONDUCTIVITY):
        super().__init__()
        self.data_range = data_range

    def forward(
        self, preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor = None
    ) -> torch.Tensor:
        preds, targets, masks = self.apply_mask(preds, targets, masks)

        # structural_similarity_index_measure ya es diferenciable
        ssim_val = structural_similarity_index_measure(preds, targets, data_range=self.data_range)

        # Como queremos MINIMIZAR la pérdida, devolvemos (1 - SSIM)
        # SSIM = 1.0 (imágenes idénticas) -> Loss = 0.0
        return 1.0 - ssim_val
