

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from src.runtime import resolve_device


class ActiveLearner:
    """Sample-selection helper for adaptive retraining under drift."""

    def __init__(self, model, device="cpu"):
        """Initialize learner with model and runtime device."""
        self.model = model
        self.device = resolve_device(device)
        self.train_stats = None

    def fit_train_stats(self, x_static, x_temporal):
        """Fit uncertainty statistics on reference data for p-value ranking."""
        probs, _ = self._get_probs_and_features(x_static, x_temporal)

        gini = 1 - np.sum(probs ** 2, axis=1)
        p_safe = np.clip(probs, 1e-8, 1.0)
        entropy = -np.sum(p_safe * np.log(p_safe), axis=1)

        self.train_stats = {
            "gini_mean": np.mean(gini),
            "gini_std": np.std(gini),
            "entropy_mean": np.mean(entropy),
            "entropy_std": np.std(entropy),
        }
        print(f"[ActiveLearner] Train stats fitted: {self.train_stats}")

    def _get_probs_and_features(self, x_static, x_temporal):
        """Run model inference and return softmax probabilities and features.

        Returns:
            probs: np.ndarray
            features: np.ndarray
        """
        self.model.eval()
        if hasattr(self.model, "to"):
            self.model.to(self.device)

        if isinstance(x_static, np.ndarray):
            x_static = torch.FloatTensor(x_static).to(self.device)
        if isinstance(x_temporal, np.ndarray):
            x_temporal = torch.FloatTensor(x_temporal).to(self.device)

        with torch.no_grad():
            logits = self.model(x_static, x_temporal)
            probs = F.softmax(logits, dim=1)
            if hasattr(self.model, "extract_features"):
                features = self.model.extract_features(x_static, x_temporal)
            else:
                features = logits

        return probs.cpu().numpy(), features.cpu().numpy()

    def select_random(self, x_static, x_temporal, budget_ratio):
        """Random baseline selector.

        Returns:
            indices: np.ndarray
        """
        n_samples = x_static.shape[0]
        n_select = int(n_samples * budget_ratio)
        return np.random.choice(n_samples, n_select, replace=False)

    def select_deep_gini(self, x_static, x_temporal, budget_ratio):
        """Select highest-uncertainty samples by DeepGini score.

        Criterion:
            Gini = 1 - sum(p^2), higher means more uncertainty.

        Returns:
            indices: np.ndarray
        """
        probs, _ = self._get_probs_and_features(x_static, x_temporal)
        gini_scores = 1 - np.sum(probs ** 2, axis=1)
        n_select = int(len(gini_scores) * budget_ratio)
        return np.argsort(gini_scores)[::-1][:n_select]

    def select_entropy(self, x_static, x_temporal, budget_ratio):
        """Select highest-uncertainty samples by predictive entropy.

        Criterion:
            Entropy = -sum(p * log(p)), higher means more uncertainty.

        Returns:
            indices: np.ndarray
        """
        probs, _ = self._get_probs_and_features(x_static, x_temporal)
        probs = np.clip(probs, 1e-8, 1.0)
        entropy_scores = -np.sum(probs * np.log(probs), axis=1)

        n_select = int(len(entropy_scores) * budget_ratio)
        return np.argsort(entropy_scores)[::-1][:n_select]

    def select_geometric_diversity(self, x_static, x_temporal, budget_ratio, iterations=20):
        """Select diverse samples by maximizing log-det of feature Gram matrix.

        Args:
            iterations: Number of Monte Carlo candidate subsets.

        Returns:
            indices: np.ndarray
        """
        _, features = self._get_probs_and_features(x_static, x_temporal)

        min_vals = features.min(axis=0)
        max_vals = features.max(axis=0)
        denom = max_vals - min_vals
        denom[denom == 0] = 1e-6
        features_norm = (features - min_vals) / denom

        n_samples = features.shape[0]
        n_select = int(n_samples * budget_ratio)
        if n_select == 0:
            return np.array([], dtype=int)

        best_indices = None
        max_log_det = -np.inf

        for _ in range(iterations):
            current_indices = np.random.choice(n_samples, n_select, replace=False)
            subset = features_norm[current_indices]
            gram_matrix = np.matmul(subset, subset.T)
            gram_matrix += np.eye(n_select) * 1e-6
            sign, log_det = np.linalg.slogdet(gram_matrix)
            if sign > 0 and log_det > max_log_det:
                max_log_det = log_det
                best_indices = current_indices

        if best_indices is None:
            return np.random.choice(n_samples, n_select, replace=False)

        return best_indices

    def select_ensemble_p_value(self, x_static, x_temporal, budget_ratio):
        """Select samples that are statistically rare under train uncertainty stats.

        Criterion:
            Compute z-scores for Gini and entropy against fitted train means/std,
            convert them to one-sided p-values, then rank by lowest average p-value.

        Returns:
            indices: np.ndarray
        """
        if self.train_stats is None:
            print("[Warning] Train stats not found. Falling back to Rank Ensemble.")
            return self.select_ensemble_rank(x_static, x_temporal, budget_ratio)

        probs, _ = self._get_probs_and_features(x_static, x_temporal)

        gini = 1 - np.sum(probs ** 2, axis=1)
        z_gini = (gini - self.train_stats["gini_mean"]) / (self.train_stats["gini_std"] + 1e-9)
        p_gini = 1 - stats.norm.cdf(z_gini)

        p_safe = np.clip(probs, 1e-8, 1.0)
        entropy = -np.sum(p_safe * np.log(p_safe), axis=1)
        z_ent = (entropy - self.train_stats["entropy_mean"]) / (self.train_stats["entropy_std"] + 1e-9)
        p_ent = 1 - stats.norm.cdf(z_ent)

        avg_p = (p_gini + p_ent) / 2.0
        n_select = int(len(avg_p) * budget_ratio)
        return np.argsort(avg_p)[:n_select]

    def select_ensemble_hybrid(self, x_static, x_temporal, budget_ratio):
        """A two-stage ensemble selector combining p-value filtering and geometric diversity.

        Criterion:
            1. First select a larger pool by ensemble rank.
            2. Then apply geometric diversity selection within that pool to get final subset.

        Returns:
            indices: np.ndarray
        """
        n_samples = x_static.shape[0]
        n_final = int(n_samples * budget_ratio)
        n_pool = min(n_samples, n_final * 2)

        pool_indices = self.select_ensemble_rank(x_static, x_temporal, budget_ratio=(n_pool / n_samples))

        if len(pool_indices) <= n_final:
            return pool_indices

        x_s_pool = x_static[pool_indices]
        x_t_pool = x_temporal[pool_indices]
        gd_ratio = n_final / n_pool
        local_indices = self.select_geometric_diversity(x_s_pool, x_t_pool, gd_ratio, iterations=10)
        return pool_indices[local_indices]

    def select_ensemble_rank(self, x_static, x_temporal, budget_ratio):
        """Select by average rank of DeepGini and entropy uncertainty.

        Returns:
            indices: np.ndarray
        """
        probs, _ = self._get_probs_and_features(x_static, x_temporal)
        n_samples = probs.shape[0]

        gini_scores = 1 - np.sum(probs ** 2, axis=1)

        p_safe = np.clip(probs, 1e-8, 1.0)
        entropy_scores = -np.sum(p_safe * np.log(p_safe), axis=1)

        rank_gini = np.argsort(np.argsort(gini_scores))
        rank_entropy = np.argsort(np.argsort(entropy_scores))
        avg_rank = (rank_gini + rank_entropy) / 2.0

        n_select = int(n_samples * budget_ratio)
        return np.argsort(avg_rank)[::-1][:n_select]


def fit_uncertainty_stats_from_binary_probs(probs) -> dict[str, float]:
    """Fit uncertainty summary stats from binary probabilities.

    Returns:
        stats: dict[str, float]
    """
    p = np.asarray(probs, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
    p = np.clip(p, 0.0, 1.0)

    uncertainty = 1.0 - np.abs(p - 0.5) * 2.0
    p_safe = np.clip(p, 1e-8, 1 - 1e-8)
    entropy = -(p_safe * np.log(p_safe) + (1.0 - p_safe) * np.log(1.0 - p_safe))
    return {
        "uncertainty_mean": float(np.mean(uncertainty)),
        "uncertainty_std": float(np.std(uncertainty)),
        "entropy_mean": float(np.mean(entropy)),
        "entropy_std": float(np.std(entropy)),
    }


def select_indices_by_metric(metric, probs, budget_ratio, features: np.ndarray | None = None, train_stats: dict[str, float] | None = None) -> np.ndarray:
    """Select sample indices by metric from binary probabilities.

    Args:
        probs: Binary-class probability for class 1.
        features: Optional feature vectors for diversity terms.
        train_stats: Optional stats for ensemble_p_value metric.

    Returns:
        selected_idx: np.ndarray
    """
    p = np.asarray(probs, dtype=np.float64)
    p = np.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
    p = np.clip(p, 0.0, 1.0)

    n = p.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    k = max(1, int(round(n * float(budget_ratio))))
    k = min(k, n)

    if metric == "random":
        return np.random.choice(n, size=k, replace=False)

    uncertainty = 1.0 - np.abs(p - 0.5) * 2.0
    p_safe = np.clip(p, 1e-8, 1 - 1e-8)
    entropy = -(p_safe * np.log(p_safe) + (1.0 - p_safe) * np.log(1.0 - p_safe))

    if metric in ["uncertainty", "deep_gini"]:
        score = uncertainty
    elif metric == "entropy":
        score = entropy
    elif metric == "gd":
        if features is None:
            score = uncertainty
        else:
            center = np.mean(features, axis=0, keepdims=True)
            score = np.linalg.norm(features - center, axis=1)
    elif metric == "ensemble_rank":
        rank_u = np.argsort(np.argsort(uncertainty))
        rank_e = np.argsort(np.argsort(entropy))
        score = (rank_u + rank_e) / 2.0
    elif metric == "ensemble_p_value":
        if not train_stats:
            rank_u = np.argsort(np.argsort(uncertainty))
            rank_e = np.argsort(np.argsort(entropy))
            score = (rank_u + rank_e) / 2.0
        else:
            z_u = (uncertainty - float(train_stats.get("uncertainty_mean", 0.0))) / (float(train_stats.get("uncertainty_std", 0.0)) + 1e-9)
            z_e = (entropy - float(train_stats.get("entropy_mean", 0.0))) / (float(train_stats.get("entropy_std", 0.0)) + 1e-9)
            p_u = 1.0 - stats.norm.cdf(z_u)
            p_e = 1.0 - stats.norm.cdf(z_e)
            score = -(p_u + p_e) / 2.0
    elif metric == "ensemble_hybrid":
        if features is None:
            diversity = np.zeros_like(uncertainty)
        else:
            center = np.mean(features, axis=0, keepdims=True)
            diversity = np.linalg.norm(features - center, axis=1)
            diversity = diversity / (np.max(diversity) + 1e-8)
        entropy_norm = entropy / (np.max(entropy) + 1e-8)
        score = 0.45 * uncertainty + 0.45 * entropy_norm + 0.10 * diversity
    else:
        score = uncertainty

    return np.argsort(score)[::-1][:k]
