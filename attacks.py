import torch
import torch.nn as nn
import numpy as np

class BaseAttack:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

    def generate(self, x_static, x_temporal, y):
        raise NotImplementedError

class FGSMAttack(BaseAttack):
    def __init__(self, model, device='cpu', epsilon=0.05):
        super().__init__(model, device)
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss()

    def generate(self, x_static, x_temporal, y):
        x_s_adv = x_static.clone().detach().requires_grad_(True)
        x_t_adv = x_temporal.clone().detach().requires_grad_(True)
        
        self.model.eval()
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

class PGDAttack(BaseAttack):
    def __init__(self, model, device='cpu', epsilon=0.05, steps=10, alpha=0.01):
        super().__init__(model, device)
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss()

    def generate(self, x_static, x_temporal, y):
        x_s_adv = x_static.clone().detach()
        x_t_adv = x_temporal.clone().detach()
        
        # Originals for projection
        x_s_orig = x_static.clone().detach()
        x_t_orig = x_temporal.clone().detach()
        
        self.model.eval()
        
        for _ in range(self.steps):
            x_s_adv.requires_grad = True
            x_t_adv.requires_grad = True
            
            self.model.zero_grad()
            outputs = self.model(x_s_adv, x_t_adv)
            loss = self.criterion(outputs, y)
            loss.backward()
            
            with torch.no_grad():
                # Apply gradient
                if x_s_adv.grad is not None:
                    x_s_adv += self.alpha * x_s_adv.grad.sign()
                if x_t_adv.grad is not None:
                    x_t_adv += self.alpha * x_t_adv.grad.sign()
                
                # Projection (Clip to epsilon ball)
                d_s = torch.clamp(x_s_adv - x_s_orig, min=-self.epsilon, max=self.epsilon)
                d_t = torch.clamp(x_t_adv - x_t_orig, min=-self.epsilon, max=self.epsilon)
                
                x_s_adv.copy_(x_s_orig + d_s)
                x_t_adv.copy_(x_t_orig + d_t)
                
                # Manual zero grad (safe)
                if x_s_adv.grad is not None: x_s_adv.grad.zero_()
                if x_t_adv.grad is not None: x_t_adv.grad.zero_()

        return x_s_adv.detach(), x_t_adv.detach()

class MimicryAttack(BaseAttack):
    """
    Mimicry Attack: Replaces features of malicious samples with those of random benign samples.
    """
    def __init__(self, model, device='cpu', benign_X_s=None, benign_X_t=None, trials=20):
        super().__init__(model, device)
        self.benign_s = torch.FloatTensor(benign_X_s).to(device) if benign_X_s is not None else None
        self.benign_t = torch.FloatTensor(benign_X_t).to(device) if benign_X_t is not None else None
        self.trials = trials

    def generate(self, x_static, x_temporal, y):
        if self.benign_s is None:
            print("[Mimicry] Warning: No benign data provided.")
            return x_static, x_temporal

        batch_size = x_static.shape[0]
        # Initialize with original (failed attack)
        x_s_best = x_static.clone()
        x_t_best = x_temporal.clone()
        
        # We assume we only target Malicious samples (class 1)
        # If input y=0 (benign), we don't need to mimic.
        target_indices = (y == 1).nonzero(as_tuple=True)[0]
        
        if len(target_indices) == 0:
            return x_static, x_temporal
            
        for _ in range(self.trials):
            # Randomly select benign samples
            idx = torch.randint(0, len(self.benign_s), (len(target_indices),)).to(self.device)
            cand_s = self.benign_s[idx]
            cand_t = self.benign_t[idx]
            
            # Temporary Batch Construction:
            # We construct a full batch where only target_indices are replaced by candidates
            curr_s = x_s_best.clone()
            curr_t = x_t_best.clone()
            
            curr_s[target_indices] = cand_s
            curr_t[target_indices] = cand_t
            
            with torch.no_grad():
                logits = self.model(curr_s, curr_t)
                preds = torch.argmax(logits, dim=1)
                
                # Success criteria: Malicious(1) predicted as Benign(0)
                # We check only for target_indices
                success = (preds[target_indices] == 0)
                
                # If success, update the 'best'
                if success.sum() > 0:
                    succ_idx = target_indices[success]
                    # Select the candidates that caused success
                    # (Note: cand_s aligns with target_indices)
                    # We need indices of cand_s corresponding to success
                    # success is a boolean mask of length len(target_indices)
                    x_s_best[succ_idx] = cand_s[success]
                    x_t_best[succ_idx] = cand_t[success]

        return x_s_best, x_t_best

class GDKDEAttack(BaseAttack):
    """
    GD-KDE: Maximizes (CrossEntropy + lambda * KDE)
    """
    def __init__(self, model, device='cpu', benign_X_s=None, benign_X_t=None, 
                 epsilon=0.05, steps=20, alpha=0.01, lambda_kde=0.5, bandwidth=1.0):
        super().__init__(model, device)
        self.benign_s = torch.FloatTensor(benign_X_s).to(device) if benign_X_s is not None else None
        self.benign_t = torch.FloatTensor(benign_X_t).to(device) if benign_X_t is not None else None
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha
        self.lambda_kde = lambda_kde
        self.bandwidth = bandwidth
        self.criterion = nn.CrossEntropyLoss()

    def _compute_kde(self, x_s, x_t):
        if self.benign_s is None: return torch.tensor(0.0).to(self.device)
        
        # Sample for speed
        n_kde = min(100, len(self.benign_s))
        idx = np.random.choice(len(self.benign_s), n_kde, replace=False)
        b_s = self.benign_s[idx] 
        b_t = self.benign_t[idx] 
        
        # Distance
        diff_s = torch.abs(x_s.unsqueeze(1) - b_s.unsqueeze(0)) 
        sum_s = torch.sum(diff_s, dim=2) 
        
        diff_t = torch.abs(x_t.unsqueeze(1) - b_t.unsqueeze(0))
        sum_t = torch.sum(diff_t, dim=2) 
        
        total_dist = sum_s + sum_t
        kernel_val = torch.exp(-total_dist / self.bandwidth)
        kde = torch.mean(kernel_val, dim=1)
        return kde

    def generate(self, x_static, x_temporal, y):
        x_s_adv = x_static.clone().detach()
        x_t_adv = x_temporal.clone().detach()
        x_s_orig = x_s_adv.clone()
        x_t_orig = x_t_adv.clone()
        
        self.model.eval()
        
        for _ in range(self.steps):
            x_s_adv.requires_grad = True
            x_t_adv.requires_grad = True
            self.model.zero_grad()
            
            outputs = self.model(x_s_adv, x_t_adv)
            
            # 1. Classification Loss (Maximize CE -> Misclassify)
            loss_cls = self.criterion(outputs, y)
             
            # 2. KDE (Maximize KDE -> Look Benign)
            # We add lambda * KDE to the objective we want to MAXIMIZE
            kde_val = self._compute_kde(x_s_adv, x_t_adv)
            
            # Total Objective to MAXIMIZE: J = Loss_CLS + lambda * KDE
            # Gradient Ascent on J
            total_obj = loss_cls + self.lambda_kde * torch.mean(kde_val)
            
            total_obj.backward()
            
            with torch.no_grad():
                if x_s_adv.grad is not None:
                    x_s_adv += self.alpha * x_s_adv.grad.sign()
                if x_t_adv.grad is not None:
                     x_t_adv += self.alpha * x_t_adv.grad.sign()
                
                # Projection
                d_s = torch.clamp(x_s_adv - x_s_orig, min=-self.epsilon, max=self.epsilon)
                d_t = torch.clamp(x_t_adv - x_t_orig, min=-self.epsilon, max=self.epsilon)
                x_s_adv.copy_(x_s_orig + d_s)
                x_t_adv.copy_(x_t_orig + d_t)
                
                if x_s_adv.grad is not None: x_s_adv.grad.zero_()
                if x_t_adv.grad is not None: x_t_adv.grad.zero_()

        return x_s_adv.detach(), x_t_adv.detach()
