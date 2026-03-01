from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .base_attack import BaseAttack


class GDKDEAttack(BaseAttack):
    def __init__(
        self,
        model,
        device: str = "cpu",
        benign_X_s=None,
        benign_X_t=None,
        epsilon: float = 0.05,
        steps: int = 20,
        alpha: float = 0.01,
        lambda_kde: float = 0.5,
        bandwidth: float = 1.0,
    ):
        super().__init__(model, device)
        self.benign_s = torch.FloatTensor(benign_X_s).to(self.device) if benign_X_s is not None else None
        self.benign_t = torch.FloatTensor(benign_X_t).to(self.device) if benign_X_t is not None else None
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha
        self.lambda_kde = lambda_kde
        self.bandwidth = bandwidth
        self.criterion = nn.CrossEntropyLoss()

    def _compute_kde(self, x_s, x_t):
        if self.benign_s is None:
            return torch.tensor(0.0, device=self.device)

        n_kde = min(100, len(self.benign_s))
        idx = np.random.choice(len(self.benign_s), n_kde, replace=False)
        b_s = self.benign_s[idx]
        b_t = self.benign_t[idx]

        diff_s = torch.abs(x_s.unsqueeze(1) - b_s.unsqueeze(0))
        sum_s = torch.sum(diff_s, dim=2)

        diff_t = torch.abs(x_t.unsqueeze(1) - b_t.unsqueeze(0))
        sum_t = torch.sum(diff_t, dim=2)

        total_dist = sum_s + sum_t
        kernel_val = torch.exp(-total_dist / self.bandwidth)
        return torch.mean(kernel_val, dim=1)

    def generate(self, x_static, x_temporal, y):
        x_s_adv = x_static.clone().detach().to(self.device)
        x_t_adv = x_temporal.clone().detach().to(self.device)
        y = y.to(self.device)

        x_s_orig = x_s_adv.clone()
        x_t_orig = x_t_adv.clone()

        self.model.eval()
        self.model.to(self.device)

        for _ in range(self.steps):
            x_s_adv.requires_grad = True
            x_t_adv.requires_grad = True
            self.model.zero_grad()

            outputs = self.model(x_s_adv, x_t_adv)
            loss_cls = self.criterion(outputs, y)
            kde_val = self._compute_kde(x_s_adv, x_t_adv)
            total_obj = loss_cls + self.lambda_kde * torch.mean(kde_val)
            total_obj.backward()

            with torch.no_grad():
                if x_s_adv.grad is not None:
                    x_s_adv += self.alpha * x_s_adv.grad.sign()
                if x_t_adv.grad is not None:
                    x_t_adv += self.alpha * x_t_adv.grad.sign()

                d_s = torch.clamp(x_s_adv - x_s_orig, min=-self.epsilon, max=self.epsilon)
                d_t = torch.clamp(x_t_adv - x_t_orig, min=-self.epsilon, max=self.epsilon)
                x_s_adv.copy_(x_s_orig + d_s)
                x_t_adv.copy_(x_t_orig + d_t)

                if x_s_adv.grad is not None:
                    x_s_adv.grad.zero_()
                if x_t_adv.grad is not None:
                    x_t_adv.grad.zero_()

        return x_s_adv.detach(), x_t_adv.detach()
