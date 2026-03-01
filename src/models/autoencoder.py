from __future__ import annotations

import torch
import torch.nn as nn

from .base_model import BaseIDSModel


class Autoencoder(BaseIDSModel):
    def __init__(self, input_dim: int, device: str = "cpu"):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        return self.decoder(encoded)

    def get_init_params(self) -> dict:
        return {
            "input_dim": self.input_dim,
        }
