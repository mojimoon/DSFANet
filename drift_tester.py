import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from copy import deepcopy
import config
from preprocessing import DataPreprocessor

class DriftGenerator:
    def __init__(self):
        pass

    def load_natural_shift_data(self, filepath):
        """
        Natural Shift: 读取一个未使用的数据集 (如 NF-ToN-IoT-v3.csv)
        复用 DataPreprocessor 的逻辑以确保特征对齐和归一化一致性。
        """
        print(f"[Natural Shift] Loading external dataset: {filepath}")
        preprocessor = DataPreprocessor(filepath)
        # 注意：这里我们假设直接使用 prepare_data 的逻辑
        # 在实际部署中，应该使用训练集的 Scaler 来 transform 这个新数据集
        # 为了简单演示，这里重新 fit_transform，或者若有保存的 scaler 应该 load 进来
        (X_s, X_t, y), _ = preprocessor.prepare_data()
        
        # 只返回"训练集"部分作为全部数据，因为 prepare_data 做了分割
        # 这里为了获取全部数据可能需要调整 preprocessor，但暂时复用现有的
        X_s_all = np.concatenate([X_s, _[0]], axis=0)
        X_t_all = np.concatenate([X_t, _[1]], axis=0)
        y_all = np.concatenate([y, _[2]], axis=0)
        
        return X_s_all, X_t_all, y_all

    def simulate_label_shift(self, x_s, x_t, y, target_malicious_ratio=0.9):
        """
        Label Shift: 重新采样 Testing Set，使 Malicious (label=1) 的占比达到指定值。
        策略：保持 Malicious 样本数量不变 (模拟攻击爆发)，下采样 Normal 样本；
        或者如果 Normal 不够，上采样 Malicious。
        这里采用：固定 Malicious，调整 Normal 的数量。
        """
        print(f"[Label Shift] Resampling to target malicious ratio: {target_malicious_ratio}")
        
        indices_normal = np.where(y == 0)[0]
        indices_malicious = np.where(y == 1)[0]
        
        n_malicious = len(indices_malicious)
        n_normal = len(indices_normal)
        
        if n_malicious == 0:
            print("Warning: No malicious samples found.")
            return x_s, x_t, y
            
        # 目标: n_malicious / (n_malicious + n_new_normal) = ratio
        # n_malicious = ratio * (n_malicious + n_new_normal)
        # n_malicious * (1 - ratio) = ratio * n_new_normal
        # n_new_normal = n_malicious * (1 - ratio) / ratio
        
        n_new_normal = int(n_malicious * (1 - target_malicious_ratio) / target_malicious_ratio)
        
        # 如果需要的 normal 样本比现有的多，则需要重复采样 (replace=True)
        replace = n_new_normal > n_normal
        selected_normal_indices = np.random.choice(indices_normal, size=n_new_normal, replace=replace)
        
        new_indices = np.concatenate([indices_malicious, selected_normal_indices])
        np.random.shuffle(new_indices) # Shuffle to mix classes
        
        return x_s[new_indices], x_t[new_indices], y[new_indices]

    def simulate_corruption(self, x, noise_type='gaussian', severity=0.1, target_cols_indices=None):
        """
        Corrupted Shift: 给指定数值列加噪音
        x: numpy array (features)
        target_cols_indices: list of integers (indices of columns to corrupt). If None, apply to all.
        """
        x_corrupted = x.copy()
        rows, cols = x.shape
        
        if target_cols_indices is None:
            target_cols_indices = range(cols)
            
        print(f"[Corrupted Shift] Applying {noise_type} noise (severity={severity}) to {len(target_cols_indices)} columns.")
        
        mask = np.zeros_like(x)
        mask[:, target_cols_indices] = 1.0
        
        if noise_type == 'gaussian':
            # Add N(0, severity) noise
            noise = np.random.normal(loc=0.0, scale=severity, size=x.shape)
            x_corrupted += noise * mask
            
        elif noise_type == 'zero':
            # Dropout mimic: zero out values
            # severity here acts as probability to zero out
            dropout_mask = np.random.rand(*x.shape) > severity
            # Only apply dropout logic to target columns (inverted logic for mask multiplication)
            # We want to KEEP data where mask=0 (untargeted) OR where dropout_mask=1
            final_mask = np.ones_like(x)
            final_mask[:, target_cols_indices] = dropout_mask[:, target_cols_indices]
            x_corrupted *= final_mask

        return x_corrupted

    def simulate_adversarial(self, model, x_s, x_t, y, method='fgsm', epsilon=0.05, steps=10, alpha=0.01, target_stream='temporal'):
        """
        Adversarial Shift: 针对数值列生成对抗样本 (White-box attack)
        需要 PyTorch 模型来计算梯度。
        
        Args:
            model: PyTorch model (e.g., DSFANet)
            x_s, x_t: numpy arrays
            y: numpy array
            target_stream: 'static' or 'temporal' (which input to perturb)
        Returns:
            x_s_adv, x_t_adv (numpy arrays)
        """
        print(f"[Adversarial Shift] Generating {method} examples. Epsilon={epsilon}, Target={target_stream}")
        
        # Convert to Tensor and enable gradient
        x_s_tensor = torch.FloatTensor(x_s)
        x_t_tensor = torch.FloatTensor(x_t)
        y_tensor = torch.LongTensor(y)
        
        criterion = nn.CrossEntropyLoss()
        
        model.eval() # Keep model in eval mode (e.g. fix dropout), but we need grad for input
        
        # Clone inputs to detach and allow gradient tracking
        xs_adv = x_s_tensor.clone().detach()
        xt_adv = x_t_tensor.clone().detach()
        
        if target_stream == 'static':
            xs_adv.requires_grad = True
        elif target_stream == 'temporal':
            xt_adv.requires_grad = True
        else: # both
            xs_adv.requires_grad = True
            xt_adv.requires_grad = True

        def get_loss():
            # Zero model gradients (though we optimize input, nice to safeguard)
            model.zero_grad()
            outputs = model(xs_adv, xt_adv)
            loss = criterion(outputs, y_tensor)
            return loss

        # --- FGSM ---
        if method == 'fgsm':
            loss = get_loss()
            loss.backward()
            
            if target_stream in ['static', 'both'] and xs_adv.grad is not None:
                # Perturb inputs: x + eps * sign(grad)
                # Note: valid adversarial usually maximizes loss
                xs_adv.data = xs_adv.data + epsilon * xs_adv.grad.data.sign()
                
            if target_stream in ['temporal', 'both'] and xt_adv.grad is not None:
                xt_adv.data = xt_adv.data + epsilon * xt_adv.grad.data.sign()

        # --- PGD (Projected Gradient Descent) ---
        elif method == 'pgd':
            # Start with random perturbation within epsilon ball? (Optional, here strict PGD)
            orig_xs = xs_adv.clone().detach()
            orig_xt = xt_adv.clone().detach()
            
            for i in range(steps):
                if xs_adv.grad is not None: xs_adv.grad.zero_()
                if xt_adv.grad is not None: xt_adv.grad.zero_()
                
                loss = get_loss()
                loss.backward()
                
                with torch.no_grad():
                    if target_stream in ['static', 'both']:
                        pert = alpha * xs_adv.grad.sign()
                        xs_adv += pert
                        # Projection: Clip perturbation to [-eps, eps]
                        perturbation = torch.clamp(xs_adv - orig_xs, min=-epsilon, max=epsilon)
                        xs_adv.copy_(orig_xs + perturbation)
                        
                    if target_stream in ['temporal', 'both']:
                        pert = alpha * xt_adv.grad.sign()
                        xt_adv += pert
                        perturbation = torch.clamp(xt_adv - orig_xt, min=-epsilon, max=epsilon)
                        xt_adv.copy_(orig_xt + perturbation)

                # Need to reset requires_grad for next iter loop logic if detached
                if target_stream == 'static': xs_adv.requires_grad = True
                if target_stream == 'temporal': xt_adv.requires_grad = True

        return xs_adv.detach().numpy(), xt_adv.detach().numpy(), y
