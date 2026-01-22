import numpy as np
import torch

class UnificationLayer:
    """
    负责将不同模型（分类器/异常检测器）的输出统一为 z-score 标准化分数。
    """
    def __init__(self):
        self.stats = {} # 存储每个模型的 (mu, sigma)

    def register_stats(self, model_name, scores):
        """
        在训练/验证阶段注册模型的统计分布
        """
        mu = np.mean(scores)
        sigma = np.std(scores) + 1e-8
        self.stats[model_name] = {'mu': mu, 'sigma': sigma}
        print(f"[{model_name}] Unification Stats: Mu={mu:.4f}, Sigma={sigma:.4f}")

    def unify_score(self, model_name, raw_scores):
        """
        将原始分数转换为标准化分数
        """
        if model_name not in self.stats:
            # 如果没有注册，默认不做缩放 (或抛出警告)
            return raw_scores
        
        mu = self.stats[model_name]['mu']
        sigma = self.stats[model_name]['sigma']
        return (raw_scores - mu) / sigma

class EnsembleManager:
    def __init__(self, unification_layer):
        self.unifier = unification_layer
        self.models = [] # List of dicts: {name, model, weight, type, input_req}

    def add_model(self, name, model, weight=1.0, model_type='classifier', input_req='static'):
        """
        name: 模型名称
        model: 模型实例 (PyTorch nn.Module 或 Sklearn estimator)
        weight: 集成权重
        model_type: 'classifier' (输出概率) 或 'anomaly' (输出重构误差)
        input_req: 'static', 'temporal', 或 'both' (决定喂给模型什么数据)
        """
        self.models.append({
            'name': name,
            'model': model,
            'weight': weight,
            'type': model_type,
            'input_req': input_req
        })

    def _get_raw_scores(self, model_info, x_static, x_temporal):
        model = model_info['model']
        req = model_info['input_req']
        m_type = model_info['type']
        
        # 1. 准备输入数据
        if req == 'both':
            # 只有 PyTorch 模型 (DSFANet) 会走这里，假设它接受两个张量
            # 需要转为 Tensor
            inputs = (torch.FloatTensor(x_static), torch.FloatTensor(x_temporal))
        elif req == 'static':
            data = x_static
            if not isinstance(model, torch.nn.Module): # Sklearn
                inputs = (data,) 
            else: # PyTorch
                inputs = (torch.FloatTensor(data),)
        elif req == 'temporal':
            data = x_temporal
            if not isinstance(model, torch.nn.Module):
                inputs = (data,)
            else:
                inputs = (torch.FloatTensor(data),)
        
        # 2. 获取模型输出
        # PyTorch 模型
        if isinstance(model, torch.nn.Module):
            model.eval()
            with torch.no_grad():
                if req == 'both':
                    output = model(*inputs)
                else:
                    output = model(inputs[0])
                
                # 处理输出类型
                if m_type == 'classifier':
                    # Logits -> Softmax -> Prob of class 1
                    probs = torch.softmax(output, dim=1).numpy()
                    raw_scores = probs[:, 1]
                elif m_type == 'anomaly':
                    # Reconstruction -> MSE
                    x_in = inputs[0].numpy()
                    x_out = output.numpy()
                    raw_scores = np.mean(np.power(x_in - x_out, 2), axis=1)

        # Sklearn 模型 (RF, SVM)
        else:
            if hasattr(model, 'predict_proba'):
                raw_scores = model.predict_proba(inputs[0])[:, 1]
            else:
                # Fallback for models without probability (e.g. OneClassSVM if used)
                raw_scores = model.predict(inputs[0])
        
        return raw_scores

    def calibrate_unifier(self, x_static_val, x_temporal_val):
        """
        使用验证集计算每个模型的 mu 和 sigma
        """
        print("\n--- Calibrating Unification Layer ---")
        for info in self.models:
            raw_scores = self._get_raw_scores(info, x_static_val, x_temporal_val)
            self.unifier.register_stats(info['name'], raw_scores)

    def predict(self, x_static, x_temporal):
        """
        集成预测
        """
        final_score = np.zeros(x_static.shape[0])
        total_weight = 0.0
        
        for info in self.models:
            raw_scores = self._get_raw_scores(info, x_static, x_temporal)
            unified_scores = self.unifier.unify_score(info['name'], raw_scores)
            
            final_score += unified_scores * info['weight']
            total_weight += info['weight']
        
        return final_score / total_weight