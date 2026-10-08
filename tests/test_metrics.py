import pytest
import torch
import math

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
    preds_corrupted[0, 0, 0:10, 0:10] = 9999.0 # Should be ignored by mask
    
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
