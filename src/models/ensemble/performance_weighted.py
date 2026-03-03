from __future__ import annotations

from sklearn.metrics import accuracy_score

from .base import UnificationLayer
from .voting import VotingEnsemble


class PerformanceWeightedEnsemble(VotingEnsemble):
    def __init__(self, unifier: UnificationLayer, device: str = "cpu"):
        super().__init__(unifier=unifier, weights=None, device=device)

    def fit_weights(self, x_static_val, x_temporal_val, y_val):
        print("[PerformanceWeightedEnsemble] Calculating weights based on validation accuracy...")
        new_weights = {}

        for wrapper in self.models:
            unified = wrapper.get_unified_score(x_static_val, x_temporal_val)
            preds = (unified > 0.5).astype(int)
            acc = accuracy_score(y_val, preds)

            weight = acc ** 2
            new_weights[wrapper.name] = weight
            print(f"  Model {wrapper.name}: Acc={acc:.4f}, Weight={weight:.4f}")

        self.set_weights(new_weights)
