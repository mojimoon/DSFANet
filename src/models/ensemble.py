import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


class UnificationLayer:
    """Map model-specific scores to [0, 1] using learned min-max stats."""

    def __init__(self):
        self.stats = {}

    def register_stats(self, model_name, scores):
        self.stats[model_name] = {"min": np.min(scores), "max": np.max(scores)}
        if self.stats[model_name]["max"] == self.stats[model_name]["min"]:
            self.stats[model_name]["max"] += 1e-6

    def unify(self, model_name, raw_scores):
        if model_name not in self.stats:
            return raw_scores
        stats = self.stats[model_name]
        unified = (raw_scores - stats["min"]) / (stats["max"] - stats["min"])
        return np.clip(unified, 0.0, 1.0)


class ModelWrapper:
    """Provide a unified score interface for torch and sklearn models."""

    def __init__(self, name, model, model_type, input_req, unifier):
        self.name = name
        self.model = model
        self.model_type = model_type
        self.input_req = input_req
        self.unifier = unifier

    def get_raw_score(self, x_static, x_temporal):
        inputs = []
        is_torch_model = isinstance(self.model, torch.nn.Module)

        if self.input_req in ["static", "both"]:
            inputs.append(torch.FloatTensor(x_static) if is_torch_model else x_static)
        if self.input_req in ["temporal", "both"]:
            inputs.append(torch.FloatTensor(x_temporal) if is_torch_model else x_temporal)

        if is_torch_model:
            self.model.eval()
            with torch.no_grad():
                out = self.model(inputs[0], inputs[1]) if len(inputs) == 2 else self.model(inputs[0])
                if self.model_type == "classifier":
                    probs = torch.softmax(out, dim=1).numpy()
                    raw = probs[:, 1]
                else:
                    x_in = inputs[0].numpy()
                    x_out = out.numpy()
                    raw = np.mean(np.power(x_in - x_out, 2), axis=1)
        else:
            if hasattr(self.model, "predict_proba"):
                raw = self.model.predict_proba(inputs[0])[:, 1]
            else:
                raw = self.model.predict(inputs[0])

        return raw

    def get_unified_score(self, x_static, x_temporal):
        raw = self.get_raw_score(x_static, x_temporal)
        return self.unifier.unify(self.name, raw)


class BaseEnsemble:
    def __init__(self, unifier):
        self.unifier = unifier
        self.models = []
        self.last_intermediate_results = {}

    def add_model(self, name, model, model_type="classifier", input_req="static"):
        self.models.append(ModelWrapper(name, model, model_type, input_req, self.unifier))

    def calibrate(self, x_static_val, x_temporal_val):
        print(f"[{self.__class__.__name__}] Calibrating base models...")
        for wrapper in self.models:
            raw = wrapper.get_raw_score(x_static_val, x_temporal_val)
            self.unifier.register_stats(wrapper.name, raw)

    def _collect_base_scores(self, x_static, x_temporal):
        scores_list = []
        self.last_intermediate_results = {}
        for wrapper in self.models:
            score = wrapper.get_unified_score(x_static, x_temporal)
            scores_list.append(score)
            self.last_intermediate_results[wrapper.name] = score
        return np.column_stack(scores_list)

    def get_intermediate_results(self):
        return self.last_intermediate_results

    def predict(self, x_static, x_temporal):
        raise NotImplementedError


class VotingEnsemble(BaseEnsemble):
    """Weighted soft-voting on unified scores."""

    def __init__(self, unifier, weights=None):
        super().__init__(unifier)
        self.weights = weights

    def set_weights(self, weights_dict):
        self.weights = weights_dict

    def predict(self, x_static, x_temporal):
        base_scores = self._collect_base_scores(x_static, x_temporal)
        if self.weights:
            weight_vec = np.array([self.weights.get(model.name, 1.0) for model in self.models])
        else:
            weight_vec = np.ones(len(self.models))

        if np.sum(weight_vec) == 0:
            weight_vec = np.ones(len(self.models))
        return np.average(base_scores, axis=1, weights=weight_vec)


class StackingEnsemble(BaseEnsemble):
    """Stacking with logistic-regression meta learner."""

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
            print("Warning: Meta-learner not fitted. Returning voting average instead.")
            base_scores = self._collect_base_scores(x_static, x_temporal)
            return np.mean(base_scores, axis=1)

        base_preds = self._collect_base_scores(x_static, x_temporal)
        return self.meta_learner.predict_proba(base_preds)[:, 1]


class XGBoostStackingEnsemble(BaseEnsemble):
    """Stacking with XGBoost meta learner."""

    def __init__(self, unifier, xgb_params=None):
        super().__init__(unifier)
        self.params = xgb_params if xgb_params else {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "eval_metric": "logloss",
            "use_label_encoder": False,
        }
        self.meta_learner = XGBClassifier(**self.params)
        self.is_fitted = False

    def fit_meta(self, x_static_val, x_temporal_val, y_val):
        print(f"[{self.__class__.__name__}] Training XGBoost Meta-Learner...")
        base_preds = self._collect_base_scores(x_static_val, x_temporal_val)
        self.meta_learner.fit(base_preds, y_val)
        self.is_fitted = True

    def predict(self, x_static, x_temporal):
        if not self.is_fitted:
            print("Warning: Meta-learner not fitted. Returning voting average instead.")
            base_scores = self._collect_base_scores(x_static, x_temporal)
            return np.mean(base_scores, axis=1)
        base_preds = self._collect_base_scores(x_static, x_temporal)
        return self.meta_learner.predict_proba(base_preds)[:, 1]


class SimpleDNNMetaLearner(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class DNNStackingEnsemble(BaseEnsemble):
    """Stacking with a shallow DNN meta learner."""

    def __init__(self, unifier, epochs=50, lr=0.01):
        super().__init__(unifier)
        self.epochs = epochs
        self.lr = lr
        self.meta_learner = None
        self.is_fitted = False

    def fit_meta(self, x_static_val, x_temporal_val, y_val):
        print(f"[{self.__class__.__name__}] Training DNN Meta-Learner...")
        base_preds = self._collect_base_scores(x_static_val, x_temporal_val)

        x_torch = torch.FloatTensor(base_preds)
        y_torch = torch.FloatTensor(y_val).unsqueeze(1)
        self.meta_learner = SimpleDNNMetaLearner(base_preds.shape[1])

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.meta_learner.parameters(), lr=self.lr)

        self.meta_learner.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            out = self.meta_learner(x_torch)
            loss = criterion(out, y_torch)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item():.4f}")

        self.is_fitted = True

    def predict(self, x_static, x_temporal):
        if not self.is_fitted or self.meta_learner is None:
            print("Warning: Meta-learner not fitted. Returning voting average instead.")
            base_scores = self._collect_base_scores(x_static, x_temporal)
            return np.mean(base_scores, axis=1)

        base_preds = self._collect_base_scores(x_static, x_temporal)
        x_torch = torch.FloatTensor(base_preds)
        self.meta_learner.eval()
        with torch.no_grad():
            preds = self.meta_learner(x_torch)
        return preds.numpy().flatten()


class RankAveragingEnsemble(BaseEnsemble):
    """Average score ranks across base models."""

    def predict(self, x_static, x_temporal):
        base_scores = self._collect_base_scores(x_static, x_temporal)
        n_samples, n_models = base_scores.shape
        if n_samples == 0:
            return np.array([])

        ranks = np.zeros_like(base_scores)
        for idx in range(n_models):
            ranks[:, idx] = rankdata(base_scores[:, idx], method="min")

        avg_rank = np.mean(ranks, axis=1)
        return (avg_rank - 1) / (n_samples - 1 + 1e-8)


class PerformanceWeightedEnsemble(VotingEnsemble):
    """Automatically set weights from validation performance."""

    def __init__(self, unifier):
        super().__init__(unifier, weights=None)

    def fit_weights(self, x_static_val, x_temporal_val, y_val):
        print(f"[{self.__class__.__name__}] Calculating weights based on validation accuracy...")
        new_weights = {}
        for wrapper in self.models:
            unified = wrapper.get_unified_score(x_static_val, x_temporal_val)
            preds = (unified > 0.5).astype(int)
            acc = accuracy_score(y_val, preds)
            new_weights[wrapper.name] = acc**2
            print(f"  Model {wrapper.name}: Acc={acc:.4f}, Weight={new_weights[wrapper.name]:.4f}")

        self.set_weights(new_weights)
