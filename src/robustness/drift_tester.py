import numpy as np
import torch

from src.data.data_loader import DataPreprocessor
from src.robustness.attacks import FGSMAttack, GDKDEAttack, MimicryAttack, PGDAttack


class DriftGenerator:
    def __init__(self):
        pass

    def load_natural_shift_data(self, filepath):
        """Load an external dataset to simulate natural distribution shift."""
        print(f"[Natural Shift] Loading external dataset: {filepath}")
        preprocessor = DataPreprocessor(filepath)
        train_split, test_split = preprocessor.prepare_data()

        x_s_train, x_t_train, y_train = train_split
        x_s_test, x_t_test, y_test = test_split

        x_s_all = np.concatenate([x_s_train, x_s_test], axis=0)
        x_t_all = np.concatenate([x_t_train, x_t_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)
        return x_s_all, x_t_all, y_all

    def simulate_label_shift(self, x_s, x_t, y, target_malicious_ratio=0.9):
        """Resample to reach a target malicious class ratio."""
        print(f"[Label Shift] Resampling to target malicious ratio: {target_malicious_ratio}")

        indices_normal = np.where(y == 0)[0]
        indices_malicious = np.where(y == 1)[0]

        n_malicious = len(indices_malicious)
        n_normal = len(indices_normal)

        if n_malicious == 0:
            print("Warning: no malicious samples found.")
            return x_s, x_t, y

        n_new_normal = int(n_malicious * (1 - target_malicious_ratio) / target_malicious_ratio)
        replace = n_new_normal > n_normal
        selected_normal_indices = np.random.choice(indices_normal, size=n_new_normal, replace=replace)

        new_indices = np.concatenate([indices_malicious, selected_normal_indices])
        np.random.shuffle(new_indices)
        return x_s[new_indices], x_t[new_indices], y[new_indices]

    def simulate_corruption(self, x, noise_type="gaussian", severity=0.1, target_cols_indices=None):
        """Apply corruption to selected feature columns."""
        x_corrupted = x.copy()
        _, cols = x.shape

        if target_cols_indices is None:
            target_cols_indices = range(cols)

        print(
            f"[Corrupted Shift] Applying {noise_type} noise (severity={severity}) to {len(target_cols_indices)} columns."
        )

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
        target_stream="temporal",
    ):
        """Generate adversarial examples for static/temporal dual inputs."""
        print(f"[Adversarial Shift] Generating {method} examples. Epsilon={epsilon}, Target={target_stream}")

        x_s_tensor = torch.FloatTensor(x_s)
        x_t_tensor = torch.FloatTensor(x_t)
        y_tensor = torch.LongTensor(y)

        attacker = None
        if method == "fgsm":
            attacker = FGSMAttack(model, epsilon=epsilon)
        elif method == "pgd":
            attacker = PGDAttack(model, epsilon=epsilon, steps=steps, alpha=alpha)
        elif method == "mimicry":
            print("Warning: Mimicry needs benign references; using input batch as placeholder.")
            attacker = MimicryAttack(model, benign_x_static=x_s_tensor, benign_x_temporal=x_t_tensor)
        elif method == "gdkde":
            print("Warning: GD-KDE needs benign references; using input batch as placeholder.")
            attacker = GDKDEAttack(model, benign_x_static=x_s_tensor, benign_x_temporal=x_t_tensor, epsilon=epsilon)

        if attacker is None:
            return x_s, x_t, y

        adv_s, adv_t = attacker.generate(x_s_tensor, x_t_tensor, y_tensor)
        return adv_s.cpu().numpy(), adv_t.cpu().numpy(), y
