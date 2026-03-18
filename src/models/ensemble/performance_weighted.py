

from sklearn.metrics import accuracy_score

from .base import UnificationLayer
from .voting import VotingEnsemble


class PerformanceWeightedEnsemble(VotingEnsemble):
    """Voting ensemble that assigns validation-performance-based weights."""

    def __init__(self, unifier, device="cpu"):
        """Initialize weighted-voting ensemble."""
        super().__init__(unifier=unifier, weights=None, device=device)

    def fit_weights(self, x_static_val, x_temporal_val, y_val):
        """Fit per-model voting weights from validation accuracy.

        Criterion:
            Weight is accuracy^2 to emphasize stronger base models.
        """
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
