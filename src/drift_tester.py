from __future__ import annotations

import numpy as np
import torch

from .attacker import FGSMAttack, PGDAttack, MimicryAttack, GDKDEAttack
from .data_loader import DataPreprocessor


class DriftGenerator:
    def load_natural_shift_data(self, filepath):
        print(f"[Natural Shift] Loading external dataset: {filepath}")
        preprocessor = DataPreprocessor(filepath)
        train_data, test_data = preprocessor.prepare_data()

        x_s_all = np.concatenate([train_data[0], test_data[0]], axis=0)
        x_t_all = np.concatenate([train_data[1], test_data[1]], axis=0)
        y_all = np.concatenate([train_data[2], test_data[2]], axis=0)
        return x_s_all, x_t_all, y_all

    def simulate_label_shift(self, x_s, x_t, y, target_malicious_ratio=0.9):
        print(f"[Label Shift] Resampling to target malicious ratio: {target_malicious_ratio}")

        indices_normal = np.where(y == 0)[0]
        indices_malicious = np.where(y == 1)[0]

        n_malicious = len(indices_malicious)
        n_normal = len(indices_normal)

        if n_malicious == 0:
            print("Warning: No malicious samples found.")
            return x_s, x_t, y

        n_new_normal = int(n_malicious * (1 - target_malicious_ratio) / target_malicious_ratio)
        replace = n_new_normal > n_normal
        selected_normal_indices = np.random.choice(indices_normal, size=n_new_normal, replace=replace)

        new_indices = np.concatenate([indices_malicious, selected_normal_indices])
        np.random.shuffle(new_indices)
        return x_s[new_indices], x_t[new_indices], y[new_indices]

    def simulate_corruption(self, x, noise_type="gaussian", severity=0.1, target_cols_indices=None):
        x_corrupted = x.copy()
        _, cols = x.shape

        if target_cols_indices is None:
            target_cols_indices = range(cols)

        print(f"[Corrupted Shift] Applying {noise_type} noise (severity={severity}) to {len(target_cols_indices)} columns.")

        mask = np.zeros_like(x)
        mask[:, target_cols_indices] = 1.0

        if noise_type == "gaussian":
            noise = np.random.normal(loc=0.0, scale=severity, size=x.shape)
            x_corrupted += noise * mask
        elif noise_type == "zero":
            dropout_mask = np.random.rand(*x.shape) > severity
            final_mask = np.ones_like(x)
            final_mask[:, target_cols_indices] = dropout_mask[:, target_cols_indices]
            x_corrupted *= final_mask

        return x_corrupted

    def simulate_adversarial(
        self,
        model,
        x_s,
        x_t,
        y,
        method="fgsm",
        epsilon=0.05,
        steps=10,
        alpha=0.01,
        device="cpu",
        benign_x_s=None,
        benign_x_t=None,
    ):
        print(f"[Adversarial Shift] Generating {method} examples. Epsilon={epsilon}")

        x_s_tensor = torch.FloatTensor(x_s)
        x_t_tensor = torch.FloatTensor(x_t)
        y_tensor = torch.LongTensor(y)

        attacker = None
        if method == "fgsm":
            attacker = FGSMAttack(model, device=device, epsilon=epsilon)
        elif method == "pgd":
            attacker = PGDAttack(model, device=device, epsilon=epsilon, steps=steps, alpha=alpha)
        elif method == "mimicry":
            if benign_x_s is None or benign_x_t is None:
                benign_idx = np.where(y == 0)[0]
                if len(benign_idx) == 0:
                    print("Warning: No benign candidates found for mimicry; returning original samples.")
                    return x_s, x_t, y
                benign_x_s = x_s[benign_idx]
                benign_x_t = x_t[benign_idx]
            attacker = MimicryAttack(model, device=device, benign_X_s=benign_x_s, benign_X_t=benign_x_t)
        elif method == "gdkde":
            if benign_x_s is None or benign_x_t is None:
                benign_idx = np.where(y == 0)[0]
                if len(benign_idx) == 0:
                    print("Warning: No benign candidates found for gdkde; returning original samples.")
                    return x_s, x_t, y
                benign_x_s = x_s[benign_idx]
                benign_x_t = x_t[benign_idx]
            attacker = GDKDEAttack(model, device=device, benign_X_s=benign_x_s, benign_X_t=benign_x_t, epsilon=epsilon)

        if attacker:
            adv_s, adv_t = attacker.generate(x_s_tensor, x_t_tensor, y_tensor)
            return adv_s.cpu().numpy(), adv_t.cpu().numpy(), y

        return x_s, x_t, y
