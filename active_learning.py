import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

class ActiveLearner:
    """
    Implements selection metrics derived from basic_DNN.py adapted for PyTorch models.
    Supports: Random, DeepGini, Entropy, and simplified Geometric Diversity.
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.train_stats = None

    def fit_train_stats(self, x_static, x_temporal):
        """
        Pre-compute training set statistics (Mean/Std of Gini and Entropy).
        Required for P-value based ensemble methods.
        """
        probs, _ = self._get_probs_and_features(x_static, x_temporal)
        
        # 1. Gini
        gini = 1 - np.sum(probs ** 2, axis=1)
        
        # 2. Entropy
        p_safe = np.clip(probs, 1e-8, 1.0)
        entropy = -np.sum(p_safe * np.log(p_safe), axis=1)
        
        self.train_stats = {
            'gini_mean': np.mean(gini),
            'gini_std': np.std(gini),
            'entropy_mean': np.mean(entropy),
            'entropy_std': np.std(entropy)
        }
        print(f"[ActiveLearner] Train stats fitted: {self.train_stats}")


    def _get_probs_and_features(self, x_static, x_temporal):
        """Helper to get probabilities and features for a batch efficiently."""
        self.model.eval()
        # Convert to tensor if numpy
        if isinstance(x_static, np.ndarray):
            x_static = torch.FloatTensor(x_static).to(self.device)
        if isinstance(x_temporal, np.ndarray):
            x_temporal = torch.FloatTensor(x_temporal).to(self.device)

        with torch.no_grad():
            logits = self.model(x_static, x_temporal)
            probs = F.softmax(logits, dim=1)
            
            # Try to get features if model supports it (for GD metric)
            if hasattr(self.model, 'extract_features'):
                features = self.model.extract_features(x_static, x_temporal)
            else:
                features = logits # Fallback to logits if features unavailable
                
        return probs.cpu().numpy(), features.cpu().numpy()

    def select_random(self, x_static, x_temporal, budget_ratio):
        """Random Selection"""
        n_samples = x_static.shape[0]
        n_select = int(n_samples * budget_ratio)
        indices = np.random.choice(n_samples, n_select, replace=False)
        return indices

    def select_deep_gini(self, x_static, x_temporal, budget_ratio):
        """
        DeepGini: 1 - sum(prob^2).
        Higher score means more uncertainty.
        """
        probs, _ = self._get_probs_and_features(x_static, x_temporal)
        # sum of squares for each sample
        gini_scores = 1 - np.sum(probs ** 2, axis=1)
        
        # Select top-k highest scores (most uncertain)
        n_select = int(len(gini_scores) * budget_ratio)
        indices = np.argsort(gini_scores)[::-1][:n_select]
        return indices

    def select_entropy(self, x_static, x_temporal, budget_ratio):
        """
        Entropy: -sum(p * log(p)).
        Higher score means more uncertainty.
        """
        probs, _ = self._get_probs_and_features(x_static, x_temporal)
        # Clip to avoid log(0)
        probs = np.clip(probs, 1e-8, 1.0)
        entropy_scores = -np.sum(probs * np.log(probs), axis=1)
        
        n_select = int(len(entropy_scores) * budget_ratio)
        indices = np.argsort(entropy_scores)[::-1][:n_select]
        return indices

    def select_geometric_diversity(self, x_static, x_temporal, budget_ratio, iterations=20):
        """
        Geometric Diversity (GD):
        1. Extract features.
        2. Randomly sample subsets 'iterations' times.
        3. Calculate determinant of (X * X.T) for each subset (volume).
        4. Pick subset with max determinant.
        """
        _, features = self._get_probs_and_features(x_static, x_temporal)
        
        # MinMax Scale features (as per basic_DNN.py)
        min_vals = features.min(axis=0)
        max_vals = features.max(axis=0)
        # Avoid div by zero
        denom = max_vals - min_vals
        denom[denom == 0] = 1e-6
        features_norm = (features - min_vals) / denom
        
        n_samples = features.shape[0]
        n_select = int(n_samples * budget_ratio)
        if n_select == 0: return np.array([], dtype=int)
        
        best_indices = None
        max_log_det = -np.inf
        
        # Monte Carlo approximation
        for _ in range(iterations):
            # Sample indices
            current_indices = np.random.choice(n_samples, n_select, replace=False)
            subset = features_norm[current_indices]
            
            # Calculate Determinant
            # det(A) -> can be very small/large, use slogdet.
            # We compute det(X @ X.T) which is Gram matrix.
            # For numerical stability with large matrices, we usually use logdet.
            gram_matrix = np.matmul(subset, subset.T) 
            # Add small jitter to diagonal for stability
            gram_matrix += np.eye(n_select) * 1e-6
            
            sign, log_det = np.linalg.slogdet(gram_matrix)
            
            if sign > 0 and log_det > max_log_det:
                max_log_det = log_det
                best_indices = current_indices
                
        if best_indices is None:
            # Fallback to random if all dets are 0 or fail
            return np.random.choice(n_samples, n_select, replace=False)
            
        return best_indices

    def select_ensemble_p_value(self, x_static, x_temporal, budget_ratio):
        """
        Ensemble (P-Value):
        Uses Training stats to normalize Gini and Entropy, then averages P-values.
        Requires fit_train_stats() to be called first.
        """
        if self.train_stats is None:
            print("[Warning] Train stats not found. Falling back to Rank Ensemble.")
            return self.select_ensemble_rank(x_static, x_temporal, budget_ratio)

        probs, _ = self._get_probs_and_features(x_static, x_temporal)
        
        # 1. Gini
        gini = 1 - np.sum(probs ** 2, axis=1)
        # P-value (assuming normal distribution, upper tail test for abnormality)
        # Note: Gini increases with uncertainty. 
        z_gini = (gini - self.train_stats['gini_mean']) / (self.train_stats['gini_std'] + 1e-9)
        p_gini = 1 - stats.norm.cdf(z_gini)
        
        # 2. Entropy
        p_safe = np.clip(probs, 1e-8, 1.0)
        entropy = -np.sum(p_safe * np.log(p_safe), axis=1)
        z_ent = (entropy - self.train_stats['entropy_mean']) / (self.train_stats['entropy_std'] + 1e-9)
        p_ent = 1 - stats.norm.cdf(z_ent)
        
        # Average P-value
        avg_p = (p_gini + p_ent) / 2.0
        
        # Smallest P-value -> Most "Abnormal" / Uncertain relative to training dist
        n_select = int(len(avg_p) * budget_ratio)
        indices = np.argsort(avg_p)[:n_select]
        
        return indices

    def select_ensemble_hybrid(self, x_static, x_temporal, budget_ratio):
        """
        Ensemble (Hybrid):
        1. Select 2*Candidate_Size using Uncertainty (Ensemble P-Value or Rank).
        2. From that subset, select final Budget using Geometric Diversity (GD).
        """
        n_samples = x_static.shape[0]
        n_final = int(n_samples * budget_ratio)
        n_pool = min(n_samples, n_final * 2) # Pool size = 2x Budget
        
        # Step 1: Uncertainty Filter (Using Rank Ensemble for stability)
        # We want top 'n_pool' uncertain samples
        pool_indices = self.select_ensemble_rank(x_static, x_temporal, budget_ratio=(n_pool/n_samples))
        
        if len(pool_indices) <= n_final:
            return pool_indices
            
        # Extract pool data
        x_s_pool = x_static[pool_indices]
        x_t_pool = x_temporal[pool_indices]
        
        # Step 2: Diversity Selection (GD) on the pool
        # GD expects a ratio relative to the input set (the pool)
        # We want to select n_final samples from n_pool
        gd_ratio = n_final / n_pool
        local_indices = self.select_geometric_diversity(x_s_pool, x_t_pool, gd_ratio, iterations=10)
        
        # Map back to original indices
        final_indices = pool_indices[local_indices]
        return final_indices

    def select_ensemble_rank(self, x_static, x_temporal, budget_ratio):
        """
        Ensemble (Rank Averaging):
        Combines DeepGini and Entropy by averaging their ranks.
        Robust to different scales of metrics.
        """
        probs, _ = self._get_probs_and_features(x_static, x_temporal)
        n_samples = probs.shape[0]
        
        # 1. Gini
        gini_scores = 1 - np.sum(probs ** 2, axis=1)
        
        # 2. Entropy
        p_safe = np.clip(probs, 1e-8, 1.0)
        entropy_scores = -np.sum(p_safe * np.log(p_safe), axis=1)
        
        # 3. Rank Averaging
        # Higher score = Higher rank
        rank_gini = np.argsort(np.argsort(gini_scores))
        rank_entropy = np.argsort(np.argsort(entropy_scores))
        
        avg_rank = (rank_gini + rank_entropy) / 2.0
        
        # Select top-k highest rank
        n_select = int(n_samples * budget_ratio)
        indices = np.argsort(avg_rank)[::-1][:n_select]
        return indices
