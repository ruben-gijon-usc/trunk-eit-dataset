import pytest
import torch
import torch.nn as nn

from scripts.training.dnn.dnn_utils import DynamicDNN

def test_dynamic_dnn_forward():
    # Simulate a batch of 8 samples, 16 electrodes (208 voltages), output 64x64=4096
    B = 8
    in_dim = 208
    out_dim = 4096
    
    model = DynamicDNN(input_dim=in_dim, output_dim=out_dim, hidden_dims=[512, 256], dropout_rate=0.1)
    
    dummy_input = torch.randn(B, in_dim)
    output = model(dummy_input)
    
    assert output.shape == (B, out_dim)

def test_dynamic_dnn_backward():
    B = 4
    in_dim = 208
    out_dim = 4096
    
    model = DynamicDNN(input_dim=in_dim, output_dim=out_dim, hidden_dims=[128], dropout_rate=0.0)
    
    # Store initial weights to check if they update
    initial_weight = model.net[0].weight.clone()
    
    dummy_input = torch.randn(B, in_dim)
    dummy_target = torch.rand(B, out_dim) * 100.0 # Random targets
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Forward
    output = model(dummy_input)
    loss = criterion(output, dummy_target)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Check that gradients were calculated for the first layer
    assert model.net[0].weight.grad is not None
    
    # Check that weights actually updated
    assert not torch.equal(model.net[0].weight, initial_weight)
