import collections
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.data.config import GROUND_TRUTH_THRESHOLD, MAX_CONDUCTIVITY


class EITMetricsEvaluator:
    def __init__(
        self,
        threshold=GROUND_TRUTH_THRESHOLD,
        data_range: float = MAX_CONDUCTIVITY,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=data_range).to(self.device)
        self.threshold = threshold

    def rmse(
        self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        if mask is not None:
            mse = F.mse_loss(preds, targets, reduction="none")
            mse = (mse * mask).sum() / (mask.sum() + 1e-8)
            return torch.sqrt(mse)
        return torch.sqrt(F.mse_loss(preds, targets))

    def dice_and_iou(self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        preds_bin = (preds > self.threshold).float()
        targets_bin = (targets > self.threshold).float()

        if mask is not None:
            preds_bin = preds_bin * mask
            targets_bin = targets_bin * mask

        preds_flat = preds_bin.reshape(preds.size(0), -1)
        targets_flat = targets_bin.reshape(targets.size(0), -1)

        intersection = (preds_flat * targets_flat).sum(dim=1)
        union = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        iou = (intersection + 1e-6) / (union - intersection + 1e-6)

        return dice.mean(), iou.mean()

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

        pe = torch.sqrt((p_cx - t_cx) ** 2 + (p_cy - t_cy) ** 2)

        return pe.mean()

    def ssim(
        self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        if mask is not None:
            preds = preds * mask
            targets = targets * mask

        return self.ssim_metric(preds, targets)


def compute_classification_metrics(
    preds: torch.Tensor, targets: torch.Tensor, threshold: float = GROUND_TRUTH_THRESHOLD
):
    p_class = (preds > threshold).bool()
    t_class = (targets > threshold).bool()

    tp = (p_class & t_class).sum().float()
    fp = (p_class & ~t_class).sum().float()
    fn = (~p_class & t_class).sum().float()
    tn = (~p_class & ~t_class).sum().float()

    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return float(accuracy), float(precision), float(recall), float(f1)


def compute_samplewise_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor | None = None,
    threshold: float = GROUND_TRUTH_THRESHOLD,
    data_range: float = MAX_CONDUCTIVITY,
    device: str = "cpu",
) -> dict[str, list[float]]:
    """
    Computes RMSE, SSIM, Dice, IoU, and Position Error (PE) for each sample individually in a batch.
    Returns a dictionary mapping each metric to a list of float values.
    """
    if preds.dim() == 2:
        B = preds.size(0)
        size = int(preds.size(1) ** 0.5)
        preds = preds.view(B, 1, size, size)
        targets = targets.view(B, 1, size, size)
        masks = masks.view(B, 1, size, size) if masks is not None else None

    evaluator = EITMetricsEvaluator(threshold=threshold, data_range=data_range, device=device)
    B = preds.size(0)

    results: dict[str, list[float]] = collections.defaultdict(list)

    for i in range(B):
        p_i = preds[i : i + 1]
        t_i = targets[i : i + 1]
        m_i = masks[i : i + 1] if masks is not None else None

        rmse_val = evaluator.rmse(p_i, t_i, m_i).item()
        ssim_val = evaluator.ssim(p_i, t_i, m_i).item()
        dice_val, iou_val = evaluator.dice_and_iou(p_i, t_i, m_i)
        pe_val = evaluator.position_error(p_i, t_i, m_i).item()

        results["rmse"].append(rmse_val)
        results["ssim"].append(ssim_val)
        results["dice"].append(dice_val.item())
        results["iou"].append(iou_val.item())
        results["pe"].append(pe_val)

    return results


def evaluate_dataset_samplewise(
    model: Any,
    dataset: Any,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 32,
    is_autoencoder: bool = False,
) -> dict[str, tuple[float, float]] | None:
    """
    Evaluates a model across an entire dataset sample-by-sample and returns {metric: (mean, std)}.
    Provides statistically rigorous sample-level reporting for Test ID and Test OOD without K-Fold.
    """
    if dataset is None or len(dataset) == 0:
        return None

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if hasattr(model, "eval"):
        model.eval()

    aggregated: dict[str, list[float]] = collections.defaultdict(list)

    with torch.no_grad():
        for batch_data in loader:
            voltages, targets = batch_data[0].to(device), batch_data[1].to(device)
            masks = batch_data[2].to(device) if len(batch_data) > 2 else None

            if is_autoencoder:
                preds = model.decoder(voltages)
            elif callable(model):
                preds = model(voltages)
            else:
                raise ValueError("Model must be callable or provide decoder")

            batch_res = compute_samplewise_metrics(
                preds=preds,
                targets=targets,
                masks=masks,
                device=device,
            )
            for k, vals in batch_res.items():
                aggregated[k].extend(vals)

    summary = {}
    for k, vals in aggregated.items():
        summary[k] = (float(np.mean(vals)), float(np.std(vals)))

    return summary
