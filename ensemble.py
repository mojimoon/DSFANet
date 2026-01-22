import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score

class UnificationLayer:
    """
    Standardizes scores to [0, 1] range based on learned statistics (MinMax strategy).
    """
    def __init__(self):
        self.stats = {} 

    def register_stats(self, model_name, scores):
        self.stats[model_name] = {
            'min': np.min(scores),
            'max': np.max(scores)
        }
        # Avoid division by zero later
        if self.stats[model_name]['max'] == self.stats[model_name]['min']:
             self.stats[model_name]['max'] += 1e-6

    def unify(self, model_name, raw_scores):
        if model_name not in self.stats:
            return raw_scores
        stats = self.stats[model_name]
        # MinMax Scaling
        unified = (raw_scores - stats['min']) / (stats['max'] - stats['min'])
        return np.clip(unified, 0.0, 1.0)

class ModelWrapper:
    """
    Wraps individual models to provide a consistent predict interface 
    and handles input data selection (Static vs Temporal).
    """
    def __init__(self, name, model, model_type, input_req, unifier):
        self.name = name
        self.model = model
        self.model_type = model_type  # 'classifier' or 'anomaly'
        self.input_req = input_req    # 'static', 'temporal', 'both'
        self.unifier = unifier

    def get_raw_score(self, x_static, x_temporal):
        # 1. Prepare Inputs
        inputs = []
        if self.input_req in ['static', 'both']:
            inputs.append(torch.FloatTensor(x_static) if isinstance(self.model, torch.nn.Module) else x_static)
        if self.input_req in ['temporal', 'both']:
            inputs.append(torch.FloatTensor(x_temporal) if isinstance(self.model, torch.nn.Module) else x_temporal)
        
        # 2. Forward Pass
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            with torch.no_grad():
                # Handling different input signatures
                if len(inputs) == 2:
                    out = self.model(inputs[0], inputs[1])
                else:
                    out = self.model(inputs[0])
                
                # Output Processing
                if self.model_type == 'classifier':
                    # Softmax -> P(class=1)
                    probs = torch.softmax(out, dim=1).numpy()
                    raw = probs[:, 1]
                elif self.model_type == 'anomaly':
                    # MSE
                    x_in = inputs[0].numpy()
                    x_out = out.numpy()
                    raw = np.mean(np.power(x_in - x_out, 2), axis=1)
        else:
            # Sklearn
            if hasattr(self.model, 'predict_proba'):
                raw = self.model.predict_proba(inputs[0])[:, 1]
            else:
                raw = self.model.predict(inputs[0]) # Fallback
                
        return raw

    def get_unified_score(self, x_static, x_temporal):
        raw = self.get_raw_score(x_static, x_temporal)
        return self.unifier.unify(self.name, raw)

class BaseEnsemble:
    def __init__(self, unifier):
        self.unifier = unifier
        self.models = [] # List of ModelWrapper
        self.last_intermediate_results = {}

    def add_model(self, name, model, model_type='classifier', input_req='static'):
        wrapper = ModelWrapper(name, model, model_type, input_req, self.unifier)
        self.models.append(wrapper)

    def calibrate(self, x_static_val, x_temporal_val):
        print(f"[{self.__class__.__name__}] Calibrating base models...")
        for wrapper in self.models:
            raw = wrapper.get_raw_score(x_static_val, x_temporal_val)
            self.unifier.register_stats(wrapper.name, raw)

    def _collect_base_scores(self, x_static, x_temporal):
        """
        Collects unified scores from all base models.
        Returns: Matrix of shape (N_samples, N_models)
        """
        scores_list = []
        self.last_intermediate_results = {}
        
        for wrapper in self.models:
            score = wrapper.get_unified_score(x_static, x_temporal)
            scores_list.append(score)
            self.last_intermediate_results[wrapper.name] = score
            
        return np.column_stack(scores_list)

    def get_intermediate_results(self):
        """Returns the individual model scores from the last prediction."""
        return self.last_intermediate_results

    def predict(self, x_static, x_temporal):
        raise NotImplementedError

class VotingEnsemble(BaseEnsemble):
    """
    Implements Weighted Soft Voting.
    """
    def __init__(self, unifier, weights=None):
        super().__init__(unifier)
        self.weights = weights # Dict {model_name: weight}

    def set_weights(self, weights_dict):
        """Allows dynamic setting of weights"""
        self.weights = weights_dict

    def predict(self, x_static, x_temporal):
        # (N, M) matrix of scores
        base_scores = self._collect_base_scores(x_static, x_temporal)
        
        if self.weights:
            weight_vec = np.array([self.weights.get(m.name, 1.0) for m in self.models])
        else:
            weight_vec = np.ones(len(self.models))
        
        # Ensure total weight is not zero
        total_weight = np.sum(weight_vec)
        if total_weight == 0: total_weight = 1.0

        # Weighted Average
        final_scores = np.average(base_scores, axis=1, weights=weight_vec)
        return final_scores

class StackingEnsemble(BaseEnsemble):
    """
    Implements Stacking with Logistic Regression Meta-Learner.
    """
    def __init__(self, unifier):
        super().__init__(unifier)
        self.meta_learner = LogisticRegression()
        self.is_fitted = False

    def fit_meta(self, x_static_val, x_temporal_val, y_val):
        print("[StackingEnsemble] Training Meta-Learner...")
        base_preds = self._collect_base_scores(x_static_val, x_temporal_val)
        self.meta_learner.fit(base_preds, y_val)
        self.is_fitted = True
        print(f"[StackingEnsemble] Meta-Learner Coeffs: {self.meta_learner.coef_}")

    def predict(self, x_static, x_temporal):
        if not self.is_fitted:
            print("Warning: Meta-learner not fitted. Returning Voting average instead.")
            base_scores = self._collect_base_scores(x_static, x_temporal)
            return np.mean(base_scores, axis=1)

        base_preds = self._collect_base_scores(x_static, x_temporal)
        final_probs = self.meta_learner.predict_proba(base_preds)[:, 1]
        return final_probs

class RankAveragingEnsemble(BaseEnsemble):
    """
    Combines models by averaging their ranks.
    Rank averaging is less sensitive to outliers and calibration errors (e.g., when mixing MSE and Probabilities).
    """
    def __init__(self, unifier):
        super().__init__(unifier)

    def predict(self, x_static, x_temporal):
        # (N, M) matrix of unified scores
        base_scores = self._collect_base_scores(x_static, x_temporal)
        n_samples = base_scores.shape[0]
        n_models = base_scores.shape[1]

        if n_samples == 0:
            return np.array([])

        # Calculate ranks for each model (column-wise)
        # rankdata returns 1..N
        ranks = np.zeros_like(base_scores)
        for i in range(n_models):
            ranks[:, i] = rankdata(base_scores[:, i], method='min')
        
        # Average Rank
        avg_rank = np.mean(ranks, axis=1)
        
        # Normalize to 0-1 range for consistency
        # Max rank average is N (if all models rank it last), Min is 1.
        final_scores = (avg_rank - 1) / (n_samples - 1 + 1e-8)
        return final_scores

class PerformanceWeightedEnsemble(VotingEnsemble):
    """
    Automatically assigns weights based on performance (Accuracy^2) on a validation set.
    """
    def __init__(self, unifier):
        super().__init__(unifier, weights=None)

    def fit_weights(self, x_static_val, x_temporal_val, y_val):
        print(f"[{self.__class__.__name__}] Calculating weights based on validation accuracy...")
        new_weights = {}
        
        for wrapper in self.models:
            # Get unified score
            unified = wrapper.get_unified_score(x_static_val, x_temporal_val)
            # Threshold at 0.5 for simple accuracy check
            preds = (unified > 0.5).astype(int)
            acc = accuracy_score(y_val, preds)
            
            # Simple heuristic: weight = accuracy^2 (to punish bad models more strictly)
            # You can change this to match F1-score or Precision if needed.
            weight = acc ** 2
            new_weights[wrapper.name] = weight
            print(f"  Model {wrapper.name}: Acc={acc:.4f}, Weight={weight:.4f}")
            
        self.set_weights(new_weights)