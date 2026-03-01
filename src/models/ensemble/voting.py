from __future__ import annotations

import numpy as np

from .base import BaseEnsemble, UnificationLayer


class VotingEnsemble(BaseEnsemble):
    def __init__(self, unifier: UnificationLayer, weights: dict[str, float] | None = None, device: str = "cpu"):
        super().__init__(unifier=unifier, device=device)
        self.weights = weights

    def set_weights(self, weights_dict: dict[str, float]) -> None:
        self.weights = weights_dict

    def get_hparams(self) -> dict:
        return {"weights": self.weights}

    def predict(self, x_static, x_temporal):
        base_scores = self._collect_base_scores(x_static, x_temporal)
        if self.weights:
            weight_vec = np.array([self.weights.get(m.name, 1.0) for m in self.models])
        else:
            weight_vec = np.ones(len(self.models))

        total_weight = np.sum(weight_vec)
        if total_weight == 0:
            total_weight = 1.0

        return np.average(base_scores, axis=1, weights=weight_vec)
