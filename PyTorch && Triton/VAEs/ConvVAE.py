import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, in_size, in_channels, latent_dim, hidden_channels: list):
        super(VAE, self).__init__()
        self.in_size = in_size  # Input should be a (in_channels, in_size, in_size) image
        self.in_channels = in_channels
        self.z_dim = latent_dim

        """Constructing the encoder"""
        encoder_layers = [
            nn.Sequential(
                nn.Conv2d(self.in_channels, hidden_channels[0], kernel_size=4, stride=2, padding=1),
                nn.ReLU()
            )
        ]

        for i in range(len(hidden_channels) - 1):
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(hidden_channels[i], hidden_channels[i + 1], kernel_size=4, stride=2, padding=1),
                    nn.ReLU()
                )
            )

        final_conv_size = in_size // (2 ** len(hidden_channels))
        flattened_dim = hidden_channels[-1] * final_conv_size * final_conv_size

        encoder_layers.append(nn.Sequential(nn.Flatten(), nn.Linear(flattened_dim, 2 * self.z_dim)))
        # 2*self.z_dim: half for mu and half for log_var
        self.encoder = nn.Sequential(*encoder_layers)

        """Constructing the decoder"""
        decoder_layers = [
            nn.Sequential(nn.Linear(self.z_dim, flattened_dim), nn.ReLU())
        ]

        decoder_layers.append(
            nn.Sequential(
                nn.Unflatten(1, (hidden_channels[-1], final_conv_size, final_conv_size))
            )
        )

        for i in range(len(hidden_channels) - 1, 0, -1):
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_channels[i], hidden_channels[i - 1], kernel_size=4, stride=2, padding=1),
                    nn.ReLU()
                )
            )

        decoder_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_channels[0], self.in_channels, kernel_size=4, stride=2, padding=1),
                nn.Identity()  # Identity activation for Gaussian assumption
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
        eps = torch.randn_like(log_var)
        return mu + eps * torch.exp(0.5 * log_var)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return mu, log_var, self.decode(z)
