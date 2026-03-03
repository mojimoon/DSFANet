from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base import BaseEnsemble, UnificationLayer


class SimpleDNNMetaLearner(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class DNNStackingEnsemble(BaseEnsemble):
    def __init__(self, unifier: UnificationLayer, epochs: int = 50, lr: float = 0.01, device: str = "cpu"):
        super().__init__(unifier=unifier, device=device)
        self.epochs = epochs
        self.lr = lr
        self.meta_learner = None
        self.is_fitted = False

    def get_hparams(self) -> dict:
        return {"epochs": self.epochs, "lr": self.lr}

    def fit_meta(self, x_static_val, x_temporal_val, y_val):
        print("[DNNStackingEnsemble] Training DNN Meta-Learner...")
        base_preds = self._collect_base_scores(x_static_val, x_temporal_val)

        x_torch = torch.FloatTensor(base_preds).to(self.device)
        y_torch = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        input_dim = base_preds.shape[1]
        self.meta_learner = SimpleDNNMetaLearner(input_dim).to(self.device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.meta_learner.parameters(), lr=self.lr)

        self.meta_learner.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            out = self.meta_learner(x_torch)
            loss = criterion(out, y_torch)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")

        self.is_fitted = True

    def predict(self, x_static, x_temporal):
        if not self.is_fitted or self.meta_learner is None:
            print("Warning: Meta-learner not fitted. Returning Voting average instead.")
            base_scores = self._collect_base_scores(x_static, x_temporal)
            return np.mean(base_scores, axis=1)

        base_preds = self._collect_base_scores(x_static, x_temporal)
        x_torch = torch.FloatTensor(base_preds).to(self.device)

        self.meta_learner.eval()
        with torch.no_grad():
            preds = self.meta_learner(x_torch)
        return preds.detach().cpu().numpy().flatten()
