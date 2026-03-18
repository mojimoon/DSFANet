

import torch
import torch.nn as nn

from .base_attack import BaseAttack


class FGSMAttack(BaseAttack):
    """Fast Gradient Sign Method (single-step) attack."""

    def __init__(self, model, device="cpu", epsilon=0.05):
        """Configure FGSM step size epsilon."""
        super().__init__(model, device)
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()

    def generate(self, x_static, x_temporal, y):
        """Generate one-step gradient-sign perturbations.

        Criterion:
            Maximize cross-entropy loss with sign(grad) step.

        Returns:
            adv_x_static: torch.Tensor
            adv_x_temporal: torch.Tensor
        """
        x_s_adv = x_static.clone().detach().to(self.device).requires_grad_(True)
        x_t_adv = x_temporal.clone().detach().to(self.device).requires_grad_(True)
        y = y.to(self.device)

        self.model.eval()
        self.model.to(self.device)
        self.model.zero_grad()

        with torch.backends.cudnn.flags(enabled=False):
            outputs = self.model(x_s_adv, x_t_adv)
            loss = self.criterion(outputs, y)
            loss.backward()

        with torch.no_grad():
            if x_s_adv.grad is not None:
                x_s_adv = x_s_adv + self.epsilon * x_s_adv.grad.sign()
            if x_t_adv.grad is not None:
                x_t_adv = x_t_adv + self.epsilon * x_t_adv.grad.sign()

        return x_s_adv.detach(), x_t_adv.detach()
