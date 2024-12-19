import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE(nn.Module):
    def __init__(self, in_size, in_channels, latent_dim, hidden_channels: list, dropout_rate=0.1):
        super().__init__()
        self.in_size = in_size  # Input should be a (in_channels, in_size, in_size) image
        self.in_channels = in_channels
        self.z_dim = latent_dim

        if not hidden_channels:
            raise ValueError("hidden_channels list cannot be empty")
        if in_size % (2 ** len(hidden_channels)) != 0:
            raise ValueError(f"in_size {in_size} must be divisible by {2 ** len(hidden_channels)}")

        """Constructing the encoder"""
        encoder_layers = [
            nn.Sequential(
                nn.Conv2d(self.in_channels, hidden_channels[0], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_channels[0]),
                nn.ReLU(),
                nn.Dropout2d(dropout_rate)
            )
        ]

        for i in range(len(hidden_channels) - 1):
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(hidden_channels[i], hidden_channels[i + 1], kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_channels[i + 1]),
                    nn.ReLU(),
                    nn.Dropout2d(dropout_rate)
                )
            )

        last_ir_size = in_size // (2 ** len(hidden_channels))
        flattened_dim = hidden_channels[-1] * last_ir_size * last_ir_size

        encoder_layers.append(nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 2 * self.z_dim),
            nn.BatchNorm1d(2 * self.z_dim),
            nn.Dropout(dropout_rate)
        ))
        self.encoder = nn.Sequential(*encoder_layers)

        """Constructing the decoder"""
        decoder_layers = [
            nn.Sequential(
                nn.Linear(self.z_dim, flattened_dim),
                nn.BatchNorm1d(flattened_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
        ]

        decoder_layers.append(
            nn.Sequential(
                nn.Unflatten(1, (hidden_channels[-1], last_ir_size, last_ir_size))
            )
        )

        for i in range(len(hidden_channels) - 1, 0, -1):
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_channels[i], hidden_channels[i - 1], kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_channels[i - 1]),
                    nn.ReLU(),
                    nn.Dropout2d(dropout_rate)
                )
            )

        decoder_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_channels[0], self.in_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(self.in_channels),
                nn.Sigmoid()
            )
        )

        self.decoder = nn.Sequential(*decoder_layers)

    @staticmethod
    def generate(model, device, num_samples=10):
        model.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, model.z_dim).to(device)
            samples = model.decode(z)
        return samples

    def encode(self, x):
        x = self.encoder(x)
        mu, log_var = x.chunk(2, dim=1)
        return mu, log_var

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

import pytest
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class TestConvVAE:
    @pytest.fixture
    def model_params(self):
        return {
            'in_size': 32,
            'in_channels': 3,
            'latent_dim': 16,
            'hidden_channels': [32, 64, 128],
            'dropout_rate': 0.1
        }
    
    @pytest.fixture
    def model(self, model_params):
        return ConvVAE(**model_params)
    
    @pytest.fixture
    def sample_batch(self, model_params):
        batch_size = 4
        return torch.randn(batch_size, model_params['in_channels'], 
                         model_params['in_size'], model_params['in_size'])

    def test_initialization(self, model_params):
        # Test successful initialization
        model = ConvVAE(**model_params)
        assert isinstance(model, nn.Module)
        
        # Test invalid input size
        with pytest.raises(ValueError):
            invalid_params = model_params.copy()
            invalid_params['in_size'] = 31  # Not divisible by 2^3
            ConvVAE(**invalid_params)
        
        # Test empty hidden channels
        with pytest.raises(ValueError):
            invalid_params = model_params.copy()
            invalid_params['hidden_channels'] = []
            ConvVAE(**invalid_params)

    def test_encoder_output_shape(self, model, sample_batch):
        mu, log_var = model.encode(sample_batch)
        assert mu.shape == (sample_batch.shape[0], model.z_dim)
        assert log_var.shape == (sample_batch.shape[0], model.z_dim)

    def test_decoder_output_shape(self, model, model_params, sample_batch):
        batch_size = sample_batch.shape[0]
        z = torch.randn(batch_size, model_params['latent_dim'])
        output = model.decode(z)
        
        expected_shape = (batch_size, model_params['in_channels'], 
                         model_params['in_size'], model_params['in_size'])
        assert output.shape == expected_shape

    def test_reparameterization(self, model, sample_batch):
        mu, log_var = model.encode(sample_batch)
        z = model.reparameterize(mu, log_var)
        assert z.shape == mu.shape
        
        # Test that reparameterization is different for multiple runs
        z2 = model.reparameterize(mu, log_var)
        assert not torch.allclose(z, z2)

    def test_forward_pass(self, model, sample_batch):
        recon, mu, log_var = model(sample_batch)
        assert recon.shape == sample_batch.shape
        assert mu.shape == (sample_batch.shape[0], model.z_dim)
        assert log_var.shape == (sample_batch.shape[0], model.z_dim)

    def test_generate_samples(self, model):
        num_samples = 5
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        samples = ConvVAE.generate(model, device, num_samples)
        expected_shape = (num_samples, model.in_channels, model.in_size, model.in_size)
        assert samples.shape == expected_shape
        
        # Test values are in valid range for sigmoid output
        assert torch.all(samples >= 0) and torch.all(samples <= 1)

    def test_output_range(self, model, sample_batch):
        recon, _, _ = model(sample_batch)
        # Test if output is in [0, 1] range due to sigmoid activation
        assert torch.all(recon >= 0) and torch.all(recon <= 1)

    def test_training_mode(self, model, model_params):
        # Create a fixed latent vector
        z = torch.randn(4, model_params['latent_dim'])
        # Test train mode
        model.train()
        out1 = model.decode(z)
        out2 = model.decode(z)
        # Outputs should be different in training mode due to dropout
        assert not torch.allclose(out1, out2)
        
        # Test eval mode
        model.eval()
        out1 = model.decode(z)
        out2 = model.decode(z)
        # Outputs should be identical in eval mode when using the same z
        assert torch.allclose(out1, out2)

    def test_batch_norm_behavior(self, model, sample_batch):
        # Test that BatchNorm statistics are updated in training mode
        model.train()
        initial_mean = model.encoder[0][1].running_mean.clone()
        
        # Forward pass should update BatchNorm statistics
        _ = model(sample_batch)
        
        updated_mean = model.encoder[0][1].running_mean
        assert not torch.allclose(initial_mean, updated_mean)

    @pytest.mark.parametrize("batch_size", [2, 4, 8])
    def test_different_batch_sizes(self, model, model_params, batch_size):
        x = torch.randn(batch_size, model_params['in_channels'], 
                       model_params['in_size'], model_params['in_size'])
        recon, mu, log_var = model(x)
        assert recon.shape == x.shape
        assert mu.shape == (batch_size, model_params['latent_dim'])
        assert log_var.shape == (batch_size, model_params['latent_dim'])

if __name__ == "__main__":
    pytest.main([__file__])