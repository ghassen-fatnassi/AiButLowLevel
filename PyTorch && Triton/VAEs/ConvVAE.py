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