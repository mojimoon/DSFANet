from __future__ import annotations

import torch
import torch.nn as nn

from .base_model import BaseIDSModel


class DSFANet(BaseIDSModel):
    def __init__(self, static_dim: int, temporal_dim: int, n_classes: int, device: str = "cpu"):
        super().__init__(device=device)
        self.static_dim = static_dim
        self.temporal_dim = temporal_dim
        self.n_classes = n_classes

        self.static_fc = nn.Sequential(
            nn.Linear(static_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.temporal_conv = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.temporal_lstm = nn.LSTM(input_size=16, hidden_size=32, batch_first=True, bidirectional=True)

        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        self.final_fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )
        self.to(self.device)

    def forward(self, x_static: torch.Tensor, x_temporal: torch.Tensor) -> torch.Tensor:
        h_static = self.static_fc(x_static)

        x_temporal_reshaped = x_temporal.unsqueeze(1)
        h_conv = self.temporal_conv(x_temporal_reshaped)
        h_conv = h_conv.permute(0, 2, 1)
        self.temporal_lstm.flatten_parameters()
        h_lstm, _ = self.temporal_lstm(h_conv)
        h_temporal = h_lstm[:, -1, :]

        combined = torch.cat((h_static, h_temporal), dim=1)
        combined_seq = combined.unsqueeze(1)
        attn_output, _ = self.attn(combined_seq, combined_seq, combined_seq)
        feat_fused = attn_output.squeeze(1)

        return self.final_fc(feat_fused)

    def extract_features(self, x_static: torch.Tensor, x_temporal: torch.Tensor) -> torch.Tensor:
        h_static = self.static_fc(x_static)
        x_temporal_reshaped = x_temporal.unsqueeze(1)
        h_conv = self.temporal_conv(x_temporal_reshaped)
        h_conv = h_conv.permute(0, 2, 1)
        self.temporal_lstm.flatten_parameters()
        h_lstm, _ = self.temporal_lstm(h_conv)
        h_temporal = h_lstm[:, -1, :]

        combined = torch.cat((h_static, h_temporal), dim=1)
        combined_seq = combined.unsqueeze(1)
        attn_output, _ = self.attn(combined_seq, combined_seq, combined_seq)
        return attn_output.squeeze(1)

    def get_init_params(self) -> dict:
        return {
            "static_dim": self.static_dim,
            "temporal_dim": self.temporal_dim,
            "n_classes": self.n_classes,
        }
