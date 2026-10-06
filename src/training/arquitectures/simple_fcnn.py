import torch
import torch.nn as nn


class SimpleFCNN(nn.Module):
    """
    Very naive FCNN mapped:
    Expected inputs: 16-electrodes measurement array -> flattening out to a Dense network 
    Expected output: 64x64 grid mapped to output vector, then reshaped to a matrix.
    """
    def __init__(self, input_dim: int, output_shape: tuple[int, int, int] = (1, 64, 64)):
        super().__init__()
        self.output_shape = output_shape
        # Calculate total flat elements for the final linear layer
        flat_output_dim = output_shape[0] * output_shape[1] * output_shape[2]
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, flat_output_dim),
            nn.ReLU() # Conductivity ranges are typically positive and unbounded above
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B x flat_dim -> B x C x H x W
        flat_out = self.net(x)
        return flat_out.view(-1, *self.output_shape)
