

import torch

from .base_attack import BaseAttack


class MimicryAttack(BaseAttack):
    """Mimicry attack using benign-sample substitution trials."""

    def __init__(self, model, device="cpu", benign_X_s=None, benign_X_t=None, trials=20):
        """Store benign reference pool and trial budget."""
        super().__init__(model, device)
        self.benign_s = torch.as_tensor(benign_X_s, dtype=torch.float32, device=self.device) if benign_X_s is not None else None
        self.benign_t = torch.as_tensor(benign_X_t, dtype=torch.float32, device=self.device) if benign_X_t is not None else None
        self.trials = trials

    def generate(self, x_static, x_temporal, y):
        """Generate adversarial samples by replacing malicious rows with benign candidates.

        Criterion:
            Keep candidate replacements that flip target predictions to benign.

        Returns:
            adv_x_static: torch.Tensor
            adv_x_temporal: torch.Tensor
        """
        x_static = x_static.to(self.device)
        x_temporal = x_temporal.to(self.device)
        y = y.to(self.device)

        if self.benign_s is None or self.benign_t is None or len(self.benign_s) == 0:
            print("[Mimicry] Warning: No benign data provided.")
            return x_static, x_temporal

        x_s_best = x_static.clone()
        x_t_best = x_temporal.clone()
        target_indices = (y == 1).nonzero(as_tuple=True)[0]

        if len(target_indices) == 0:
            return x_static, x_temporal

        self.model.eval()
        self.model.to(self.device)

        for _ in range(self.trials):
            idx = torch.randint(0, len(self.benign_s), (len(target_indices),), device=self.device)
            cand_s = self.benign_s[idx]
            cand_t = self.benign_t[idx]

            curr_s = x_s_best.clone()
            curr_t = x_t_best.clone()
            curr_s[target_indices] = cand_s
            curr_t[target_indices] = cand_t

            with torch.no_grad():
                logits = self.model(curr_s, curr_t)
                preds = torch.argmax(logits, dim=1)
                success = preds[target_indices] == 0

                if success.sum() > 0:
                    succ_idx = target_indices[success]
                    x_s_best[succ_idx] = cand_s[success]
                    x_t_best[succ_idx] = cand_t[success]

        return x_s_best, x_t_best
