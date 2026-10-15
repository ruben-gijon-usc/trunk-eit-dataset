import math

import torch

from src.training.metrics import EITMetricsEvaluator


def test_masked_metrics_integrity():
    evaluator = EITMetricsEvaluator(threshold=1.0, data_range=200.0)

    # B, C, H, W
    preds = torch.ones(1, 1, 64, 64) * 10.0
    targets = torch.ones(1, 1, 64, 64) * 10.0
    masks = torch.ones(1, 1, 64, 64)

    rmse = evaluator.rmse(preds, targets, masks).item()
    ssim = evaluator.ssim(preds, targets, masks).item()
    dice, iou = evaluator.dice_and_iou(preds, targets, masks)

    assert math.isclose(rmse, 0.0, abs_tol=1e-5)
    assert math.isclose(ssim, 1.0, abs_tol=1e-5)
    assert math.isclose(dice.item(), 1.0, abs_tol=1e-5)


def test_background_ignorance():
    evaluator = EITMetricsEvaluator(threshold=1.0, data_range=200.0)

    preds_clean = torch.ones(1, 1, 64, 64) * 10.0
    targets = torch.ones(1, 1, 64, 64) * 10.0

    # 50% mask
    masks = torch.zeros(1, 1, 64, 64)
    masks[0, 0, 16:48, 16:48] = 1.0

    rmse_clean = evaluator.rmse(preds_clean, targets, masks).item()

    # Corrupt the background in preds
    preds_corrupted = preds_clean.clone()
    preds_corrupted[0, 0, 0:10, 0:10] = 9999.0  # Should be ignored by mask

    rmse_corrupted = evaluator.rmse(preds_corrupted, targets, masks).item()

    assert math.isclose(rmse_clean, rmse_corrupted, abs_tol=1e-5)


def test_position_error_verification():
    evaluator = EITMetricsEvaluator(threshold=1.0, data_range=200.0)

    # Create an empty grid
    targets = torch.zeros(1, 1, 64, 64)
    preds = torch.zeros(1, 1, 64, 64)
    masks = torch.ones(1, 1, 64, 64)

    # Target center at (32, 32)
    targets[0, 0, 31:34, 31:34] = 10.0

    # Pred center offset by 10 pixels in X: (32, 42)
    preds[0, 0, 31:34, 41:44] = 10.0

    # Center of mass calculation for targets should be ~ (32, 32)
    # Center of mass calculation for preds should be ~ (32, 42)
    # Expected distance: 10 pixels

    pe = evaluator.position_error(preds, targets, masks).item()

    assert math.isclose(pe, 10.0, abs_tol=1.0)


def test_samplewise_metrics_computation():
    from src.training.metrics import compute_samplewise_metrics

    B = 3
    preds = torch.ones(B, 1, 64, 64) * 2.0
    targets = torch.ones(B, 1, 64, 64) * 2.0
    masks = torch.ones(B, 1, 64, 64)

    res = compute_samplewise_metrics(preds, targets, masks, threshold=1.0, data_range=200.0)

    assert "rmse" in res
    assert "ssim" in res
    assert "dice" in res
    assert len(res["rmse"]) == B
    assert math.isclose(res["rmse"][0], 0.0, abs_tol=1e-5)
    assert math.isclose(res["ssim"][0], 1.0, abs_tol=1e-5)
    assert math.isclose(res["dice"][0], 1.0, abs_tol=1e-5)


def test_half_max_adaptive_threshold():
    # Simulates a model with compressed dynamic range (like D-Bar or regularized NN)
    # Target: wood=1.0, anomaly=100.0
    # Pred: wood=1.2, anomaly=1.6
    evaluator_fixed = EITMetricsEvaluator(threshold=6.0, data_range=200.0, threshold_mode="fixed")
    evaluator_half_max = EITMetricsEvaluator(threshold=6.0, data_range=200.0, threshold_mode="half_max")

    targets = torch.ones(1, 1, 64, 64) * 1.0
    targets[0, 0, 28:36, 28:36] = 100.0  # Anomaly at center

    preds = torch.ones(1, 1, 64, 64) * 1.2
    preds[0, 0, 28:36, 28:36] = 1.6  # Relative peak at center

    masks = torch.ones(1, 1, 64, 64)

    dice_fixed, _ = evaluator_fixed.dice_and_iou(preds, targets, masks)
    dice_hm, _ = evaluator_half_max.dice_and_iou(preds, targets, masks)

    # Fixed threshold gives 0 because preds never cross 6.0
    assert math.isclose(dice_fixed.item(), 0.0, abs_tol=1e-5)
    # Half-max detects the relative peak and achieves near-perfect segmentation
    assert dice_hm.item() > 0.95
