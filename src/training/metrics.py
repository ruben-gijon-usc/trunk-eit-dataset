import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure


class EITMetricsEvaluator:
    def __init__(self, threshold=0.5, data_range: float = 1.0, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range).to(self.device)
        self.threshold = threshold
        self.device = device

    def rmse(self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            mse = F.mse_loss(preds, targets, reduction='none')
            mse = (mse * mask).sum() / (mask.sum() + 1e-8)
            return torch.sqrt(mse)
        return torch.sqrt(F.mse_loss(preds, targets))

    def dice_and_iou(self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        preds_bin = (preds > self.threshold).float()
        targets_bin = (targets > self.threshold).float()

        if mask is not None:
            preds_bin = preds_bin * mask
            targets_bin = targets_bin * mask

        preds_flat = preds_bin.view(preds.size(0), -1)
        targets_flat = targets_bin.view(targets.size(0), -1)

        intersection = (preds_flat * targets_flat).sum(dim=1)
        union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2. * intersection + 1e-6) / (union + 1e-6)
        iou = (intersection + 1e-6) / (union - intersection + 1e-6)

        return dice.mean(), iou.mean()

    def accuracy(self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        preds_bin = (preds > self.threshold).float()
        targets_bin = (targets > self.threshold).float()

        if mask is not None:
            preds_bin = preds_bin * mask
            targets_bin = targets_bin * mask

    def position_error(self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        B, C, H, W = preds.shape
        
        preds_bin = (preds > self.threshold).float()
        targets_bin = (targets > self.threshold).float()

        if mask is not None:
            preds_bin = preds_bin * mask
            targets_bin = targets_bin * mask

        y_coords = torch.arange(H, device=self.device).view(1, 1, H, 1).expand(B, 1, H, W).float()
        x_coords = torch.arange(W, device=self.device).view(1, 1, 1, W).expand(B, 1, H, W).float()

        p_sum = preds_bin.sum(dim=(2, 3)) + 1e-6
        t_sum = targets_bin.sum(dim=(2, 3)) + 1e-6

        p_cy = (preds_bin * y_coords).sum(dim=(2, 3)) / p_sum
        p_cx = (preds_bin * x_coords).sum(dim=(2, 3)) / p_sum

        t_cy = (targets_bin * y_coords).sum(dim=(2, 3)) / t_sum
        t_cx = (targets_bin * x_coords).sum(dim=(2, 3)) / t_sum

        pe = torch.sqrt((p_cx - t_cx)**2 + (p_cy - t_cy)**2)
        
        return pe.mean()

    def ssim(self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            preds = preds * mask
            targets = targets * mask
            
        return self.ssim_metric(preds, targets)


def compute_classification_metrics(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.05):
    """
    Computes classification metrics for predicted grid vs target grid based on a conductivity threshold.
    Values strictly above the threshold are considered Class 1 (Anomaly).
    Returns basic metrics: (Accuracy, Precision, Recall, F1 Score).
    """
    p_class = (preds > threshold).bool()
    t_class = (targets > threshold).bool()

    tp = (p_class & t_class).sum().float()
    fp = (p_class & ~t_class).sum().float()
    fn = (~p_class & t_class).sum().float()
    tn = (~p_class & ~t_class).sum().float()

    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return float(accuracy), float(precision), float(recall), float(f1)
