import pytest
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.data.eit_dataset import EITDataset

@pytest.fixture
def mock_dataset_dir(tmp_path):
    # Setup mock directory structure
    inputs_dir = tmp_path / "inputs" / "voltages"
    inputs_dir.mkdir(parents=True)
    targets_grid_dir = tmp_path / "targets" / "grid"
    targets_grid_dir.mkdir(parents=True)
    targets_mask_dir = tmp_path / "targets" / "mask"
    targets_mask_dir.mkdir(parents=True)
    
    # Create 5 valid samples
    for i in range(5):
        np.save(inputs_dir / f"{i:04d}.npy", np.random.rand(208))
        np.save(targets_grid_dir / f"{i:04d}.npy", np.random.rand(64, 64))
        np.save(targets_mask_dir / f"{i:04d}.npy", np.random.randint(0, 2, size=(64, 64)))
        
    # Create 1 missing target sample (voltage exists, grid missing)
    np.save(inputs_dir / "0005.npy", np.random.rand(208))
    
    return tmp_path

def test_dataset_tensor_shapes(mock_dataset_dir):
    dataset = EITDataset(mock_dataset_dir, target_type="grid", flatten_grid=False, normalize=True)
    assert len(dataset) == 5 # Sample 5 should be ignored
    
    v, g, m = dataset[0]
    assert isinstance(v, torch.Tensor)
    assert isinstance(g, torch.Tensor)
    assert isinstance(m, torch.Tensor)
    
    assert v.shape == (208,)
    assert g.shape == (1, 64, 64)
    assert m.shape == (1, 64, 64)

def test_missing_files_handling(mock_dataset_dir):
    dataset = EITDataset(mock_dataset_dir, target_type="grid")
    assert "0005" not in dataset.valid_indices
    assert len(dataset.valid_indices) == 5

def test_deterministic_loading(mock_dataset_dir):
    dataset = EITDataset(mock_dataset_dir, target_type="grid", normalize=False)
    
    g1 = torch.Generator()
    g1.manual_seed(42)
    loader1 = DataLoader(dataset, batch_size=2, shuffle=True, generator=g1)
    
    g2 = torch.Generator()
    g2.manual_seed(42)
    loader2 = DataLoader(dataset, batch_size=2, shuffle=True, generator=g2)
    
    # Iterate and compare
    for batch1, batch2 in zip(loader1, loader2):
        assert torch.equal(batch1[0], batch2[0])
        assert torch.equal(batch1[1], batch2[1])

def test_normalization_checks(mock_dataset_dir):
    dataset = EITDataset(mock_dataset_dir, target_type="grid", normalize=True)
    v, g, m = dataset[0]
    
    assert not torch.isnan(v).any()
    assert not torch.isinf(v).any()
    
    # Check that mean is roughly 0 and std is roughly 1 for the normalized voltage
    assert torch.isclose(v.mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.isclose(v.std(), torch.tensor(1.0), atol=1e-1)
