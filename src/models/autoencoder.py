

import torch
import torch.nn as nn

from .base_model import BaseIDSModel


class Autoencoder(BaseIDSModel):
    """Feed-forward autoencoder for reconstruction-based anomaly scoring."""

    def __init__(self, input_dim: int, device="cpu"):
        """Build encoder-decoder layers for static/combined input vectors."""
        super().__init__(device=device)
        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )
        self.to(self.device)

    def forward(self, x):
        """Run reconstruction forward pass.

        Returns:
            recon: torch.Tensor
        """
        encoded = self.encoder(x)
        return self.decoder(encoded)

    def get_init_params(self) -> dict[str, int]:
        """Return params required for checkpoint reload.

        Returns:
            init_params: dict[str, int]
        """
        return {
            "input_dim": self.input_dim,
        }
