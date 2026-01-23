import numpy as np
import torch
import torch.nn.functional as F

class ActiveLearner:
    """
    Implements selection metrics derived from basic_DNN.py adapted for PyTorch models.
    Supports: Random, DeepGini, Entropy, and simplified Geometric Diversity.
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

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
