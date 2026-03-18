

import numpy as np
from scipy.stats import rankdata

from .base import BaseEnsemble, UnificationLayer


class RankAveragingEnsemble(BaseEnsemble):
    """Ensemble that averages per-model rank positions."""

    def __init__(self, unifier, device="cpu"):
        """Initialize rank-averaging ensemble."""
        super().__init__(unifier=unifier, device=device)

    def predict(self, x_static, x_temporal):
        """Predict by averaging ranks of unified base scores.

        Returns:
            probs: np.ndarray
        """
        base_scores = self._collect_base_scores(x_static, x_temporal)
        n_samples = base_scores.shape[0]
        n_models = base_scores.shape[1]

        if n_samples == 0:
            return np.array([])

        ranks = np.zeros_like(base_scores)
        for i in range(n_models):
            ranks[:, i] = rankdata(base_scores[:, i], method="min")

        avg_rank = np.mean(ranks, axis=1)
        return (avg_rank - 1) / (n_samples - 1 + 1e-8)
