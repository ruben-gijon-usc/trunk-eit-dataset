import numpy as np
import torch

from src.training.dbar.dbar_solver import DBarSolver


def test_dbar_solver_output_shape():
    solver = DBarSolver(n_electrodes=16, r_cutoff=3.0, grid_resolution=64)

    dummy_voltages = np.random.randn(208)
    recon = solver.reconstruct(dummy_voltages)

    assert isinstance(recon, torch.Tensor)
    assert recon.shape == (1, 64, 64)
    # Conductivity must be non-negative
    assert (recon >= 0.0).all()


def test_dbar_solver_masking():
    solver = DBarSolver(n_electrodes=16, r_cutoff=3.0, grid_resolution=64)

    dummy_voltages = np.random.randn(208)
    # Mask with central circle only
    mask = torch.zeros(1, 64, 64)
    mask[0, 20:44, 20:44] = 1.0

    recon = solver.reconstruct(dummy_voltages, mask=mask)

    # Outside mask must be strictly 0
    assert (recon[mask == 0] == 0.0).all()
