import json
import sys
from pathlib import Path

# Add project root to sys.path since this script resides in training/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.models import Trunk
from src.data.eit_dataset import EITDataset
from src.training.arquitectures.simple_fcc import SimpleFCNN

def train_linear():
    pass

if __name__ == "__main__":
    # Define dataset relative to root
    dataset_path = Path(__file__).resolve().parent.parent / "dataset"

    print("Checking dataset...")
    if not dataset_path.exists():
        print(f"Error: {dataset_path} not found.")
        exit(1)

    dataset = EITDataset(dataset_path, resolution=64, flatten_grid=False)
    if len(dataset) == 0:
        print(f"No valid examples found in {dataset_path}")
        exit(1)

    print(f"Loaded dataset with {len(dataset)} examples.")

    # Grab the first item to infer dims
    first_v, first_g, first_m = dataset[0]
    in_dim = first_v.shape[0]
    out_shape = tuple(first_g.shape)  # e.g., (1, 64, 64)

    print(f"Voltage dimension: {in_dim}")
    print(f"Grid dimension: {out_shape}")

    print("Splitting Dataset and Building Loaders...")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print("Building FCNN...")
    model = SimpleFCNN(input_dim=in_dim, output_shape=out_shape)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Llama a la nueva función generica de entrenamiento
    from src.training.train_utils import train_model

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        max_epochs=100,
        patience=5,
        anomaly_threshold=1.5
    )

    print("✅ Script de prueba completado exitosamente!")
