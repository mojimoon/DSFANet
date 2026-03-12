from __future__ import annotations

import torch
import torch.nn as nn

from .base_model import BaseIDSModel


class MLPClassifier(BaseIDSModel):
    def __init__(
        self,
        temporal_dim: int,
        n_classes: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.temporal_dim = temporal_dim
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        layers: list[nn.Module] = [nn.LayerNorm(self.temporal_dim)]
        in_dim = self.temporal_dim
        hidden_layers = max(1, int(self.num_layers))
        for _ in range(hidden_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, self.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )
            in_dim = self.hidden_size
        self.feature_net = nn.Sequential(*layers)

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )
        self.to(self.device)

    def extract_features(self, x_temporal: torch.Tensor) -> torch.Tensor:
        if x_temporal.dim() == 2:
            x = x_temporal
        elif x_temporal.dim() == 3:
            if x_temporal.shape[1] == 1:
                x = x_temporal[:, 0, :]
            else:
                x = x_temporal.reshape(x_temporal.shape[0], -1)
        else:
            raise ValueError(f"Expected x_temporal dim 2 or 3, got {x_temporal.dim()}")

        if x.shape[1] != self.temporal_dim:
            raise ValueError(f"Expected temporal dim {self.temporal_dim}, got {x.shape[1]}")

        return self.feature_net(x)

    def forward(self, x_temporal: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x_temporal)
        return self.fc(features)

    def get_init_params(self) -> dict:
        return {
            "temporal_dim": self.temporal_dim,
            "n_classes": self.n_classes,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
        }
