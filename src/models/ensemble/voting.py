

import numpy as np

from .base import BaseEnsemble, UnificationLayer


class VotingEnsemble(BaseEnsemble):
    """Weighted average voting over unified base scores."""

    def __init__(self, unifier, weights: dict[str, float] | None = None, device="cpu"):
        """Initialize voting ensemble with optional manual weights."""
        super().__init__(unifier=unifier, device=device)
        self.weights = weights

    def set_weights(self, weights_dict: dict[str, float]) -> None:
        """Update per-model weight map used by weighted voting."""
        self.weights = weights_dict

    def get_hparams(self) -> dict[str, dict[str, float] | None]:
        """Return serializable voting hyper-parameters.

        Returns:
            hparams: dict[str, dict[str, float] | None]
        """
        return {"weights": self.weights}

    def predict(self, x_static, x_temporal):
        """Predict by weighted average of unified base scores.

        Returns:
            probs: np.ndarray
        """
        base_scores = self._collect_base_scores(x_static, x_temporal)
        if self.weights:
            weight_vec = np.array([self.weights.get(m.name, 1.0) for m in self.models])
        else:
            weight_vec = np.ones(len(self.models))

        total_weight = np.sum(weight_vec)
        if total_weight == 0:
            total_weight = 1.0

        return np.average(base_scores, axis=1, weights=weight_vec)
