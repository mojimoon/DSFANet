import torch
import torch.nn as nn


class DSFANet(nn.Module):
    """Dual-stream model that fuses static and temporal features."""

    def __init__(self, static_dim, temporal_dim, n_classes):
        super().__init__()

        self.static_fc = nn.Sequential(
            nn.Linear(static_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.temporal_conv = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.temporal_lstm = nn.LSTM(
            input_size=16,
            hidden_size=32,
            batch_first=True,
            bidirectional=True,
        )

        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.final_fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, x_static, x_temporal):
        # Static stream: [B, static_dim] -> [B, 64]
        h_static = self.static_fc(x_static)

        # Temporal stream: [B, T] -> [B, 1, T] -> Conv1d -> BiLSTM -> [B, 64]
        x_temporal_reshaped = x_temporal.unsqueeze(1)
        h_conv = self.temporal_conv(x_temporal_reshaped)
        h_conv = h_conv.permute(0, 2, 1)
        h_lstm, _ = self.temporal_lstm(h_conv)
        h_temporal = h_lstm[:, -1, :]

        # Fusion: concat [B, 64] + [B, 64] -> [B, 128], then self-attention
        combined = torch.cat((h_static, h_temporal), dim=1)
        combined_seq = combined.unsqueeze(1)
        attn_output, _ = self.attn(combined_seq, combined_seq, combined_seq)
        feat_fused = attn_output.squeeze(1)
        return self.final_fc(feat_fused)

    def extract_features(self, x_static, x_temporal):
        """Return fused representation for active-learning diversity metrics."""
        h_static = self.static_fc(x_static)

        x_temporal_reshaped = x_temporal.unsqueeze(1)
        h_conv = self.temporal_conv(x_temporal_reshaped)
        h_conv = h_conv.permute(0, 2, 1)
        h_lstm, _ = self.temporal_lstm(h_conv)
        h_temporal = h_lstm[:, -1, :]

        combined = torch.cat((h_static, h_temporal), dim=1)
        combined_seq = combined.unsqueeze(1)
        attn_output, _ = self.attn(combined_seq, combined_seq, combined_seq)
        return attn_output.squeeze(1)


class LSTMClassifier(nn.Module):
    """Single-stream temporal classifier."""

    def __init__(self, temporal_dim, n_classes):
        super().__init__()
        self.hidden_size = 64
        self.num_layers = 2

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Sequential(nn.Linear(self.hidden_size, 32), nn.ReLU(), nn.Linear(32, n_classes))

    def forward(self, x_temporal):
        # Features are interpreted as sequence length: [B, T] -> [B, T, 1]
        x = x_temporal.unsqueeze(-1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


class Autoencoder(nn.Module):
    """Autoencoder used as anomaly detector via reconstruction error."""

    def __init__(self, input_dim):
        super().__init__()
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

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)
