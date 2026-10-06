import json
import sys
from pathlib import Path

# Add project root to sys.path since this script resides in training/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data_representations.grid import generate_grid
from src.models import Trunk
from training.early_stopping import EarlyStopping


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
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return float(accuracy), float(precision), float(recall), float(f1)



class EITDataset(Dataset):
    """
    PyTorch Dataset for EIT trunk datasets.
    Loads voltage readings and converts serialized Trunk JSONs back into True conductivity grids on the fly.
    """
    def __init__(self, dataset_dir: str | Path, resolution: int = 64):
        self.dataset_dir = Path(dataset_dir)
        self.resolution = resolution

        self.voltages_dir = self.dataset_dir / "voltages"
        self.json_dir = self.dataset_dir / "json"

        # Discover all valid samples
        self.valid_indices = []
        for v_file in self.voltages_dir.glob("*.npy"):
            idx = v_file.stem
            if (self.json_dir / f"{idx}.json").exists():
                self.valid_indices.append(idx)

        # Ensure determinism in data loading
        self.valid_indices.sort(key=int)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self.valid_indices[i]

        # 1. Load voltage readings (Inputs)
        volts_path = self.voltages_dir / f"{idx}.npy"
        voltage_arr = np.load(volts_path)
        voltage_tensor = torch.tensor(voltage_arr, dtype=torch.float32)

        # 2. Re-create the Grid layout structurally (Outputs)
        json_path = self.json_dir / f"{idx}.json"
        with open(json_path, encoding="utf-8") as fh:
            trunk_dict = json.load(fh)

        trunk = Trunk.from_dict(trunk_dict)
        grid_arr = generate_grid(trunk, resolution=self.resolution)

        # Expand dims if need to treat as an image later (1, H, W)
        # But for FCNN we want it flattened
        grid_tensor = torch.tensor(grid_arr, dtype=torch.float32).flatten()
        return voltage_tensor, grid_tensor


class SimpleFCNN(nn.Module):
    """
    Very naive FCNN mapped:
    Expected inputs: 16-electrodes measurement array -> flattening out to a Dense network 
    Expected output: 64x64 grid mapped to output vector
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.ReLU() # Conductivity ranges are typically positive and unbounded above
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    # Define dataset relative to root
    dataset_path = Path(__file__).resolve().parent.parent / "dataset"

    print("Checking dataset...")
    if not dataset_path.exists():
        print(f"Error: {dataset_path} not found.")
        exit(1)

    dataset = EITDataset(dataset_path, resolution=64)
    if len(dataset) == 0:
        print(f"No valid examples found in {dataset_path}")
        exit(1)

    print(f"Loaded dataset with {len(dataset)} examples.")

    # Grab the first item to infer dims
    first_v, first_g = dataset[0]
    in_dim = first_v.shape[0]
    out_dim = first_g.shape[0]

    print(f"Voltage dimension: {in_dim}")
    print(f"Grid dimension: {out_dim} ({int(out_dim**0.5)}x{int(out_dim**0.5)})")

    print("Splitting Dataset and Building Loaders...")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print("Building FCNN...")
    model = SimpleFCNN(input_dim=in_dim, output_dim=out_dim)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    early_stopper = EarlyStopping(patience=5, restore_best_weights=True)

    print("Starting training loop with early stopping...")
    max_epochs = 100

    # Fixed threshold since base conductivity is deterministically exactly 1.0 S/m.
    # Any value > 1.0 belongs to an anomaly. The variance-based intelligent threshold 
    # would incorrectly shift classification bounds depending on batch anomaly sizes.
    anomaly_threshold = 1.5

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        train_f1_sum = 0.0
        for voltages, targets in train_loader:
            optimizer.zero_grad()
            preds = model(voltages)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, _, _, batch_f1 = compute_classification_metrics(preds, targets, threshold=anomaly_threshold)
            train_f1_sum += batch_f1

        train_loss /= len(train_loader)
        train_f1 = train_f1_sum / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_acc_sum, val_prec_sum, val_rec_sum, val_f1_sum = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            for voltages, targets in val_loader:
                preds = model(voltages)
                loss = criterion(preds, targets)
                val_loss += loss.item()

                acc, prec, rec, f1 = compute_classification_metrics(preds, targets, threshold=anomaly_threshold)
                val_acc_sum += acc
                val_prec_sum += prec
                val_rec_sum += rec
                val_f1_sum += f1

        val_loss /= len(val_loader)
        val_f1 = val_f1_sum / len(val_loader)
        val_acc = val_acc_sum / len(val_loader)

        print(f"Epoch {epoch+1:02d}/{max_epochs} | Train Loss: {train_loss:.6f} [F1: {train_f1:.4f}] | Val Loss: {val_loss:.6f} [F1: {val_f1:.4f}, Acc: {val_acc:.4f}]")

        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered, halting training.")
            break

    print("✅ PyTorch script successfully completed!")
