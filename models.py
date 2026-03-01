import torch
import torch.nn as nn
import torch.nn.functional as F

class DSFANet(nn.Module):
    """
    DSFANet: 双流特征聚合网络 (Dual-Stream Feature Aggregation Network)
    输入: 静态特征 (Static) + 时序特征 (Temporal)
    """
    def __init__(self, static_dim, temporal_dim, n_classes):
        super(DSFANet, self).__init__()
        
        # --- 静态特征流 ---
        self.static_fc = nn.Sequential(
            nn.Linear(static_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # --- 时序特征流 ---
        # Conv1d 提取局部特征 -> BiLSTM 提取时序依赖
        self.temporal_conv = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.temporal_lstm = nn.LSTM(input_size=16, hidden_size=32, batch_first=True, bidirectional=True)
        
        # --- 特征聚合 ---
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        
        self.final_fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x_static, x_temporal):
        # Static: [B, static_dim] -> [B, 64]
        h_static = self.static_fc(x_static)
        
        # Temporal: [B, 10] -> [B, 1, 10] -> Conv -> LSTM
        x_temporal_reshaped = x_temporal.unsqueeze(1) 
        h_conv = self.temporal_conv(x_temporal_reshaped) # [B, 16, 10]
        h_conv = h_conv.permute(0, 2, 1) # [B, 10, 16]
        h_lstm, _ = self.temporal_lstm(h_conv) # [B, 10, 64]
        h_temporal = h_lstm[:, -1, :] # 取最后一个时间步 [B, 64]
        
        # Fusion
        combined = torch.cat((h_static, h_temporal), dim=1) # [B, 128]
        combined_seq = combined.unsqueeze(1) # [B, 1, 128] for Attn
        attn_output, _ = self.attn(combined_seq, combined_seq, combined_seq)
        feat_fused = attn_output.squeeze(1)
        
        return self.final_fc(feat_fused)

    def extract_features(self, x_static, x_temporal):
        """Helper for Active Learning (GD metric) - returns the fused feature vector"""
        h_static = self.static_fc(x_static)
        x_temporal_reshaped = x_temporal.unsqueeze(1) 
        h_conv = self.temporal_conv(x_temporal_reshaped)
        h_conv = h_conv.permute(0, 2, 1)
        h_lstm, _ = self.temporal_lstm(h_conv)
        h_temporal = h_lstm[:, -1, :]
        
        combined = torch.cat((h_static, h_temporal), dim=1)
        combined_seq = combined.unsqueeze(1)
        attn_output, _ = self.attn(combined_seq, combined_seq, combined_seq)
        feat_fused = attn_output.squeeze(1)
        
        return feat_fused

class LSTMClassifier(nn.Module):
    """
    基础 LSTM 模型
    输入: 仅使用时序特征 (Temporal Features)
    """
    def __init__(self, temporal_dim, n_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = 64
        self.num_layers = 2
        
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, batch_first=True, dropout=0.2)
        
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )

    def forward(self, x_temporal):
        # Input: [Batch, Feature_Dim]
        # Treat features as a sequence: [Batch, Seq_Len=Feature_Dim, Input_Size=1]
        x = x_temporal.unsqueeze(-1)
        
        # LSTM Output: [Batch, Seq_Len, Hidden]
        out, _ = self.lstm(x)
        
        # 取最后一个时间步
        out = out[:, -1, :]
        
        return self.fc(out)

class Autoencoder(nn.Module):
    """
    Autoencoder 用于异常检测
    """
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16), # Latent
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded