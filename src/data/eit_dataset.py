from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


class EITDataset(Dataset):
    """
    PyTorch Dataset for EIT trunk datasets.
    Loads voltage readings (inputs) and raw numpy grids/masks (targets).
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        flatten_grid: bool = False,
        pattern: Literal["adjacent", "opposite"] = None,
        target_type: Literal["grid", "elem_data"] = "grid",
        normalize: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.flatten_grid = flatten_grid
        self.target_type = target_type
        self.normalize = normalize

        # Set Inputs directory based on pattern
        if pattern is not None:
            self.voltages_dir = self.dataset_dir / f"inputs/voltages_{pattern}"
        elif (self.dataset_dir / "inputs/voltages_adjacent").exists():
            self.voltages_dir = self.dataset_dir / "inputs/voltages_adjacent"
        else:
            self.voltages_dir = self.dataset_dir / "inputs/voltages"

        # Set Targets directory
        self.target_dir = self.dataset_dir / f"targets/{target_type}"
        self.mask_dir = self.dataset_dir / "targets/mask"

        # Discover all valid samples
        self.valid_indices = []
        for v_file in self.voltages_dir.glob("*.npy"):
            idx = v_file.stem
            if (self.target_dir / f"{idx}.npy").exists():
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

        # Z-Score Normalization per sample para estabilizar la DNN
        if self.normalize:
            v_mean = voltage_tensor.mean()
            v_std = voltage_tensor.std() + 1e-8
            voltage_tensor = (voltage_tensor - v_mean) / v_std

        # 2. Load pre-computed targets directly
        target_arr = np.load(self.target_dir / f"{idx}.npy")
        target_tensor = torch.tensor(target_arr, dtype=torch.float32)

        if self.target_type == "grid":
            mask_arr = np.load(self.mask_dir / f"{idx}.npy")
            mask_tensor = torch.tensor(mask_arr, dtype=torch.float32)

            if self.flatten_grid:
                target_tensor = target_tensor.flatten()
                mask_tensor = mask_tensor.flatten()
            else:
                target_tensor = target_tensor.unsqueeze(0)
                mask_tensor = mask_tensor.unsqueeze(0)
        else:
            # For elem_data, mask is just a vector of 1s (all mesh elements are valid wood by definition of FEM)
            mask_tensor = torch.ones_like(target_tensor)

        return voltage_tensor, target_tensor, mask_tensor
