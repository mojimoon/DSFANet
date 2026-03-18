

import numpy as np
from xgboost import XGBClassifier

from .base import BaseEnsemble, UnificationLayer


class XGBoostStackingEnsemble(BaseEnsemble):
    """Stacking ensemble with XGBoost as meta-learner."""

    def __init__(self, unifier, xgb_params: dict | None = None, device="cpu"):
        """Initialize XGBoost-stacking ensemble."""
        super().__init__(unifier=unifier, device=device)
        self.params = xgb_params if xgb_params else {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "eval_metric": "logloss",
            # "use_label_encoder": False,
        }
        self.meta_learner = XGBClassifier(**self.params)
        self.is_fitted = False

    def get_hparams(self) -> dict[str, dict]:
        """Return serializable XGBoost meta-learner params.

        Returns:
            hparams: dict[str, dict]
        """
        return {"xgb_params": self.params}

    def fit_meta(self, x_static_val, x_temporal_val, y_val):
        """Train XGBoost meta-learner on base unified predictions."""
        print("[XGBoostStackingEnsemble] Training XGBoost Meta-Learner...")
        base_preds = self._collect_base_scores(x_static_val, x_temporal_val)
        self.meta_learner.fit(base_preds, y_val)
        self.is_fitted = True

    def predict(self, x_static, x_temporal):
        """Predict probabilities using fitted XGBoost meta-learner.

        Returns:
            probs: np.ndarray
        """
        if not self.is_fitted:
            print("Warning: Meta-learner not fitted. Returning Voting average instead.")
            base_scores = self._collect_base_scores(x_static, x_temporal)
            return np.mean(base_scores, axis=1)

        base_preds = self._collect_base_scores(x_static, x_temporal)
        return self.meta_learner.predict_proba(base_preds)[:, 1]
