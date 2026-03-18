

import torch
import torch.nn as nn

from .base_attack import BaseAttack


class PGDAttack(BaseAttack):
    """Projected Gradient Descent attack."""

    def __init__(self, model, device="cpu", epsilon=0.05, steps=10, alpha=0.01):
        """Configure epsilon-ball radius, step count, and step size."""
        super().__init__(model, device)
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss()

    def generate(self, x_static, x_temporal, y):
        """Generate iterative projected perturbations.

        Criterion:
            Iteratively ascend loss gradient and project perturbation into
            epsilon-bounded box around original samples.

        Returns:
            adv_x_static: torch.Tensor
            adv_x_temporal: torch.Tensor
        """
        x_s_adv = x_static.clone().detach().to(self.device)
        x_t_adv = x_temporal.clone().detach().to(self.device)
        y = y.to(self.device)

        x_s_orig = x_static.clone().detach().to(self.device)
        x_t_orig = x_temporal.clone().detach().to(self.device)

        self.model.eval()
        self.model.to(self.device)

        for _ in range(self.steps):
            x_s_adv.requires_grad = True
            x_t_adv.requires_grad = True

            self.model.zero_grad()
            with torch.backends.cudnn.flags(enabled=False):
                outputs = self.model(x_s_adv, x_t_adv)
                loss = self.criterion(outputs, y)
                loss.backward()

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
