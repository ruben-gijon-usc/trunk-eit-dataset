import torch
import torch.nn as nn


class BaseEITLoss(nn.Module):
    """
    Clase base para todas las funciones de pérdida del problema EIT.
    Maneja automáticamente el redimensionamiento de vectores aplanados a
    cuadrículas 2D (Grid) y la aplicación de máscaras booleanas.
    """

    def __init__(self):
        super().__init__()

    def _format_2d(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convierte tensores aplanados (B, 4096) a imágenes 2D (B, 1, 64, 64)."""
        if tensor is None:
            return None
        if tensor.dim() == 2:
            B = tensor.size(0)
            size = int(tensor.size(1) ** 0.5)
            return tensor.view(B, 1, size, size)
        elif tensor.dim() == 3:
            return tensor.unsqueeze(1)
        return tensor

    def apply_mask(self, preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor = None):
        """
        Adapta la dimensión y aplica la máscara booleana a predicciones y objetivos.
        Todo lo que esté fuera del tronco (mask == 0) se anula (0.0).
        """
        preds = self._format_2d(preds)
        targets = self._format_2d(targets)
        masks = self._format_2d(masks)

        if masks is not None:
            preds = preds * masks
            if targets is not None:
                targets = targets * masks

        return preds, targets, masks

    def forward(
        self, preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor = None
    ) -> torch.Tensor:
        raise NotImplementedError("Las subclases deben implementar forward(preds, targets, masks)")
