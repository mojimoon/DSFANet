

import torch
import torch.nn as nn

from .base_model import BaseIDSModel


class LSTMClassifier(BaseIDSModel):
    """Temporal classifier based on stacked LSTM layers."""

    def __init__(self, temporal_dim: int, n_classes: int, hidden_size=64, num_layers=2, device="cpu"):
        """Create LSTM backbone and classification head."""
        super().__init__(device=device)
        self.temporal_dim = temporal_dim
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=self.temporal_dim,
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

    def forward(self, x_temporal):
        """Predict class logits from temporal features.

        Returns:
            logits: torch.Tensor
        """
        if x_temporal.dim() == 2:
            x = x_temporal.unsqueeze(1)
        elif x_temporal.dim() == 3:
            x = x_temporal
        else:
            raise ValueError(f"Expected x_temporal dim 2 or 3, got {x_temporal.dim()}")
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

    def get_init_params(self) -> dict[str, int]:
        """Return params required for checkpoint reload.

        Returns:
            init_params: dict[str, int]
        """
        return {
            "temporal_dim": self.temporal_dim,
            "n_classes": self.n_classes,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
        }
