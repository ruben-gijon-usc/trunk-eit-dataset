from pathlib import Path
from typing import Literal

import torch
import numpy as np
from torch.utils.data import Dataset

class EITDataset(Dataset):
    """
    PyTorch Dataset for EIT trunk datasets.
    Loads voltage readings (inputs) and raw numpy grids/masks (targets).
    """
    def __init__(self, dataset_dir: str | Path, flatten_grid: bool = False, pattern: Literal["adjacent", "opposite"] = None):
        self.dataset_dir = Path(dataset_dir)
        self.flatten_grid = flatten_grid

        # Set Inputs directory based on pattern
        if pattern is not None:
            self.voltages_dir = self.dataset_dir / f"inputs/voltages_{pattern}"
        elif (self.dataset_dir / "inputs/voltages_adjacent").exists():
            self.voltages_dir = self.dataset_dir / "inputs/voltages_adjacent"
        else:
            self.voltages_dir = self.dataset_dir / "inputs/voltages"

        # Set Targets directory
        self.grid_dir = self.dataset_dir / "targets/grid"
        self.mask_dir = self.dataset_dir / "targets/mask"

        # Discover all valid samples
        self.valid_indices = []
        for v_file in self.voltages_dir.glob("*.npy"):
            idx = v_file.stem
            if (self.grid_dir / f"{idx}.npy").exists() and (self.mask_dir / f"{idx}.npy").exists():
                self.valid_indices.append(idx)

        # Ensure determinism in data loading
        self.valid_indices.sort(key=int)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = self.valid_indices[i]

        # 1. Load voltage readings (Inputs)
        voltage_arr = np.load(self.voltages_dir / f"{idx}.npy")
        voltage_tensor = torch.tensor(voltage_arr, dtype=torch.float32)

        # 2. Load pre-computed targets directly
        grid_arr = np.load(self.grid_dir / f"{idx}.npy")
        mask_arr = np.load(self.mask_dir / f"{idx}.npy")

        grid_tensor = torch.tensor(grid_arr, dtype=torch.float32)
        mask_tensor = torch.tensor(mask_arr, dtype=torch.float32)
        
        if self.flatten_grid:
            grid_tensor = grid_tensor.flatten()
            mask_tensor = mask_tensor.flatten()
        else:
            # Formato imagen: (Canal, Alto, Ancho)
            grid_tensor = grid_tensor.unsqueeze(0)
            mask_tensor = mask_tensor.unsqueeze(0)
            
        return voltage_tensor, grid_tensor, mask_tensor