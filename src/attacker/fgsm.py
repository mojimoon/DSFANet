from __future__ import annotations

import torch
import torch.nn as nn

from .base_attack import BaseAttack


class FGSMAttack(BaseAttack):
    def __init__(self, model, device: str = "cpu", epsilon: float = 0.05):
        super().__init__(model, device)
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()

    def generate(self, x_static, x_temporal, y):
        x_s_adv = x_static.clone().detach().to(self.device).requires_grad_(True)
        x_t_adv = x_temporal.clone().detach().to(self.device).requires_grad_(True)
        y = y.to(self.device)

        self.model.eval()
        self.model.to(self.device)
        self.model.zero_grad()

        outputs = self.model(x_s_adv, x_t_adv)
        loss = self.criterion(outputs, y)
        loss.backward()

        with torch.no_grad():
            if x_s_adv.grad is not None:
                x_s_adv = x_s_adv + self.epsilon * x_s_adv.grad.sign()
            if x_t_adv.grad is not None:
                x_t_adv = x_t_adv + self.epsilon * x_t_adv.grad.sign()

        return x_s_adv.detach(), x_t_adv.detach()
