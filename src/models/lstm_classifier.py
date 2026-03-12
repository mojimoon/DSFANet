from __future__ import annotations

import torch
import torch.nn as nn

from .base_model import BaseIDSModel
from src.runtime import resolve_device


class LSTMClassifier(BaseIDSModel):
    def __init__(
        self,
        temporal_dim: int,
        n_classes: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        device: str = "cpu",
        backbone: str = "mlp",
    ):
        super().__init__(device=device)
        self.temporal_dim = temporal_dim
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.backbone = backbone

        if self.backbone == "lstm":
            self.lstm = nn.LSTM(
                input_size=self.temporal_dim,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=0.2,
            )
            self.input_norm = None
            self.temporal_mlp = None
        else:
            self.lstm = None
            self.input_norm = nn.LayerNorm(self.temporal_dim)
            self.temporal_mlp = nn.Sequential(
                nn.Linear(self.temporal_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, self.hidden_size),
                nn.ReLU(),
            )

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )
        self.to(self.device)

    def extract_features(self, x_temporal: torch.Tensor) -> torch.Tensor:
        if x_temporal.dim() == 2:
            x = x_temporal.unsqueeze(1)
        elif x_temporal.dim() == 3:
            x = x_temporal
        else:
            raise ValueError(f"Expected x_temporal dim 2 or 3, got {x_temporal.dim()}")

        if self.backbone == "lstm":
            self.lstm.flatten_parameters()
            out, _ = self.lstm(x)
            return out[:, -1, :]

        if x.shape[1] == 1:
            x_flat = x[:, 0, :]
        else:
            x_flat = x.reshape(x.shape[0], -1)

        if x_flat.shape[1] != self.temporal_dim:
            raise ValueError(f"Expected flattened temporal dim {self.temporal_dim}, got {x_flat.shape[1]}")

        x_flat = self.input_norm(x_flat)
        return self.temporal_mlp(x_flat)

    def forward(self, x_temporal: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x_temporal)
        return self.fc(features)

    def get_init_params(self) -> dict:
        return {
            "temporal_dim": self.temporal_dim,
            "n_classes": self.n_classes,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "backbone": self.backbone,
        }

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str, device: str = "cpu") -> "LSTMClassifier":
        map_location = resolve_device(device)
        payload = torch.load(checkpoint_path, map_location=map_location)
        init_params = payload.get("init_params", {})
        if "backbone" not in init_params:
            init_params["backbone"] = "lstm"
        init_params["device"] = str(map_location)
        model = cls(**init_params)
        model.load_state_dict(payload["state_dict"])
        model.to(map_location)
        model.eval()
        return model
