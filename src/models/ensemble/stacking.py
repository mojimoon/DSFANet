

import numpy as np
from sklearn.linear_model import LogisticRegression

from .base import BaseEnsemble, UnificationLayer


class StackingEnsemble(BaseEnsemble):
    """Logistic-regression stacking ensemble."""

    def __init__(self, unifier, device="cpu"):
        """Initialize stacking ensemble and linear meta-learner."""
        super().__init__(unifier=unifier, device=device)
        self.meta_learner = LogisticRegression()
        self.is_fitted = False

    def fit_meta(self, x_static_val, x_temporal_val, y_val):
        """Train logistic meta-learner on base unified predictions."""
        print("[StackingEnsemble] Training Meta-Learner...")
        base_preds = self._collect_base_scores(x_static_val, x_temporal_val)
        self.meta_learner.fit(base_preds, y_val)
        self.is_fitted = True

    def predict(self, x_static, x_temporal):
        """Predict probabilities using fitted meta-learner.

        Returns:
            probs: np.ndarray
        """
        if not self.is_fitted:
            print("Warning: Meta-learner not fitted. Returning Voting average instead.")
            base_scores = self._collect_base_scores(x_static, x_temporal)
            return np.mean(base_scores, axis=1)

        base_preds = self._collect_base_scores(x_static, x_temporal)
        return self.meta_learner.predict_proba(base_preds)[:, 1]
