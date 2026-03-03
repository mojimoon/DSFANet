from __future__ import annotations

import torch
import torch.nn as nn

from .base_model import BaseIDSModel


class LSTMClassifier(BaseIDSModel):
    def __init__(self, temporal_dim: int, n_classes: int, device: str = "cpu"):
        super().__init__(device=device)
        self.temporal_dim = temporal_dim
        self.n_classes = n_classes
        self.hidden_size = 64
        self.num_layers = 2

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.2,
        )

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )
        self.to(self.device)

    def forward(self, x_temporal: torch.Tensor) -> torch.Tensor:
        x = x_temporal.unsqueeze(-1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

    def get_init_params(self) -> dict:
        return {
            "temporal_dim": self.temporal_dim,
            "n_classes": self.n_classes,
        }
