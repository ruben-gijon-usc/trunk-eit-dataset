import torch
import torch.nn as nn

from src.training.arquitectures import DynamicCNN, DynamicDNN, EITAutoencoder


def test_dynamic_dnn_forward():
    # Simulate a batch of 8 samples, 16 electrodes (208 voltages), output 64x64=4096
    B = 8
    in_dim = 208
    out_dim = 4096

    model = DynamicDNN(
        input_dim=in_dim, output_dim=out_dim, hidden_dims=[512, 256], dropout_rate=0.1
    )

    dummy_input = torch.randn(B, in_dim)
    output = model(dummy_input)

    assert output.shape == (B, out_dim)


def test_dynamic_dnn_activations():
    B = 2
    in_dim = 208
    out_dim = 4096
    dummy_input = torch.randn(B, in_dim)

    for act in ["relu", "gelu", "leaky_relu"]:
        model = DynamicDNN(
            input_dim=in_dim, output_dim=out_dim, hidden_dims=[256], dropout_rate=0.05, activation=act
        )
        out = model(dummy_input)
        assert out.shape == (B, out_dim)
        assert (out >= 0.0).all()  # Final ReLU ensures physical non-negativity


def test_dynamic_dnn_backward():
    B = 4
    in_dim = 208
    out_dim = 4096

    model = DynamicDNN(input_dim=in_dim, output_dim=out_dim, hidden_dims=[128], dropout_rate=0.0)

    # Store initial weights to check if they update
    initial_weight = model.net[0].weight.clone()

    dummy_input = torch.randn(B, in_dim)
    dummy_target = torch.rand(B, out_dim) * 100.0  # Random targets

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


def test_dynamic_cnn_forward():
    B = 4
    in_dim = 208
    model = DynamicCNN(input_dim=in_dim, base_channels=64, dropout_rate=0.1)

    dummy_input = torch.randn(B, in_dim)
    output = model(dummy_input)

    assert output.shape == (B, 1, 64, 64)


def test_dynamic_cnn_backward():
    B = 2
    in_dim = 208
    model = DynamicCNN(input_dim=in_dim, base_channels=32, dropout_rate=0.0)

    initial_fc_weight = model.fc[0].weight.clone()

    dummy_input = torch.randn(B, in_dim)
    dummy_target = torch.rand(B, 1, 64, 64) * 50.0

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    output = model(dummy_input)
    loss = criterion(output, dummy_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert model.fc[0].weight.grad is not None
    assert not torch.equal(model.fc[0].weight, initial_fc_weight)


def test_autoencoder_forward():
    B = 4
    v_dim = 208
    model = EITAutoencoder(
        voltage_dim=v_dim, encoder_channels=16, decoder_channels=32, dropout_rate=0.1
    )

    dummy_image = torch.randn(B, 1, 64, 64)
    v_pred, img_recon = model(dummy_image)

    assert v_pred.shape == (B, v_dim)
    assert img_recon.shape == (B, 1, 64, 64)

    # Test pure inverse reconstruction from voltages
    dummy_voltage = torch.randn(B, v_dim)
    inv_recon = model.reconstruct_from_voltages(dummy_voltage)
    assert inv_recon.shape == (B, 1, 64, 64)


def test_autoencoder_backward():
    B = 2
    v_dim = 208
    model = EITAutoencoder(
        voltage_dim=v_dim, encoder_channels=16, decoder_channels=32, dropout_rate=0.0
    )

    dummy_image = torch.rand(B, 1, 64, 64)
    dummy_voltage = torch.randn(B, v_dim)

    v_pred, img_recon = model(dummy_image)
    loss = nn.MSELoss()(v_pred, dummy_voltage) + nn.MSELoss()(img_recon, dummy_image)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check encoder and decoder gradients
    assert model.encoder.net[0].weight.grad is not None
    assert model.decoder.fc[0].weight.grad is not None
