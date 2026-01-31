import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from copy import deepcopy

# Local imports
import config
from drift_tester import DriftGenerator
from active_learning import ActiveLearner

class MetricsCalculator:
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_prob=None):
        """
        Calculates Accuracy, Precision, Recall, F1.
        Returns a dictionary.
        """
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

    @staticmethod
    def print_metrics(title, metrics):
        print(f"\n--- {title} ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

class ExperimentSuite:
    def __init__(self, data_pack, device='cpu'):
        """
        data_pack: ((X_s_train, X_t_train, y_train), (X_s_test, X_t_test, y_test), (X_s_val, X_t_val, y_val))
        """
        self.train_data = data_pack[0]
        self.test_data = data_pack[1]
        self.val_data = data_pack[2]
        self.device = device
        
        self.results = {}

    def run_baseline_comparison(self, models_dict):
        """
        Experiment 1: Compare base models and ensembles on the clean test set.
        """
        print("\n[Exp 1] Baseline Performance Comparison")
        X_s_test, X_t_test, y_test = self.test_data
        
        records = []
        
        for name, model in models_dict.items():
            # Handle different model types (sklearn vs PyTorch models/wrappers)
            if hasattr(model, 'predict'): # Sklearn or EnsembleWrapper
                preds = model.predict(X_s_test, X_t_test)
                # Ensure binary hard predictions if probas returned
                if hasattr(model, 'predict_proba') or (isinstance(preds, np.ndarray) and preds.dtype == float):
                     # Assume probs/scores > 0.5 is class 1
                     preds = (preds > 0.5).astype(int)
            elif isinstance(model, nn.Module):
                # PyTorch raw model
                model.eval()
                model.to(self.device)
                with torch.no_grad():
                    xs_t = torch.FloatTensor(X_s_test).to(self.device)
                    xt_t = torch.FloatTensor(X_t_test).to(self.device)
                    if hasattr(model, 'input_req') and model.input_req == 'static':
                        logits = model(xs_t)
                    elif hasattr(model, 'input_req') and model.input_req == 'temporal':
                        logits = model(xt_t)
                    else:
                        logits = model(xs_t, xt_t)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
            else:
                continue

            metrics = MetricsCalculator.calculate_metrics(y_test, preds)
            records.append({
                "Model": name,
                **metrics
            })
            
        df = pd.DataFrame(records)
        print(df)
        self.results['baseline'] = df
        return df

    def run_drift_robustness(self, target_model, drift_type='adolescence'):
        """
        Experiment 2: Evaluate model degradation under drift (Adversarial/Simulation).
        Uses a subset of test data to generate drift.
        """
        print("\n[Exp 2] Drift Robustness Analysis")
        X_s_test, X_t_test, y_test = self.test_data
        
        drifter = DriftGenerator()
        
        # 1. Baseline (Clean)
        indices = np.random.choice(len(y_test), 1000, replace=False) # Sample 1000
        clean_subset = (X_s_test[indices], X_t_test[indices], y_test[indices])
        
        # Helper to eval
        def eval_quick(m, data):
            m.eval()
            m.to(self.device)
            with torch.no_grad():
                xs = torch.FloatTensor(data[0]).to(self.device)
                xt = torch.FloatTensor(data[1]).to(self.device)
                logits = m(xs, xt)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
            return accuracy_score(data[2], preds)

        acc_clean = eval_quick(target_model, clean_subset)
        
        # 2. Add Drift
        # Using Adversarial Shift as proxy for severe concept drift
        drift_x_s, drift_x_t, drift_y = drifter.simulate_adversarial(
            target_model.cpu(), 
            clean_subset[0], clean_subset[1], clean_subset[2], 
            method='fgsm', epsilon=0.1
        )
        target_model.to(self.device)
        
        acc_drift = eval_quick(target_model, (drift_x_s, drift_x_t, drift_y))
        
        res = pd.DataFrame([
            {'Condition': 'Clean Data', 'Accuracy': acc_clean},
            {'Condition': 'Drifted (Adv) Data', 'Accuracy': acc_drift},
            {'Condition': 'Degradation', 'Accuracy': acc_clean - acc_drift}
        ])
        print(res)
        self.results['drift'] = (res, (drift_x_s, drift_x_t, drift_y))
        return res

    def run_retraining_efficiency(self, target_model, strategies=['random', 'deep_gini']):
        """
        Experiment 3: Compare retraining strategies.
        """
        print("\n[Exp 3] Adaptive Retraining Efficiency")
        
        if 'drift' not in self.results:
            print("Run drift experiment first to get candidate data.")
            return
            
        _, (drift_x_s, drift_x_t, drift_y) = self.results['drift']
        X_s_train, X_t_train, y_train = self.train_data

        learner = ActiveLearner(target_model, self.device)
        initial_state = deepcopy(target_model.state_dict())
        optimizer = torch.optim.Adam(target_model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()
        
        budget = 0.3 # 30% Labeling Budget
        records = []
        
        for strategy in strategies:
            # Reset
            target_model.load_state_dict(initial_state)
            target_model.to(self.device)
            target_model.train()
            
            # Select
            if strategy == 'random':
                idx = learner.select_random(drift_x_s, drift_x_t, budget)
            elif strategy == 'deep_gini':
                idx = learner.select_deep_gini(drift_x_s, drift_x_t, budget)
            elif strategy == 'entropy':
                idx = learner.select_entropy(drift_x_s, drift_x_t, budget)
            else:
                idx = []
                
            # Retrain (Mix with some old data)
            # Replay 100 old samples to prevent catastrophic forgetting
            replay_idx = np.random.choice(len(y_train), 100, replace=False)
            
            rx_s = np.concatenate([drift_x_s[idx], X_s_train[replay_idx]])
            rx_t = np.concatenate([drift_x_t[idx], X_t_train[replay_idx]])
            ry = np.concatenate([drift_y[idx], y_train[replay_idx]])
            
            # Simple loop
            ds = torch.utils.data.TensorDataset(
                torch.FloatTensor(rx_s), torch.FloatTensor(rx_t), torch.LongTensor(ry)
            )
            dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
            
            for epoch in range(3): # 3 Epochs fine-tuning
                for bx_s, bx_t, by in dl:
                    bx_s, bx_t, by = bx_s.to(self.device), bx_t.to(self.device), by.to(self.device)
                    optimizer.zero_grad()
                    out = target_model(bx_s, bx_t)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
            
            # Eval on full drift set (how well did we adapt?)
            target_model.eval()
            with torch.no_grad():
                ds_full = torch.FloatTensor(drift_x_s).to(self.device)
                dt_full = torch.FloatTensor(drift_x_t).to(self.device)
                logits = target_model(ds_full, dt_full)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                acc = accuracy_score(drift_y, preds)
                
            records.append({
                'Strategy': strategy,
                'Refined Accuracy': acc
            })
            
        df = pd.DataFrame(records)
        print(df)
        self.results['retraining'] = df
        return df

if __name__ == "__main__":
    pass
