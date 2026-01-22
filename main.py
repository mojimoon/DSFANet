import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import config
from data_loader import DataPreprocessor, get_dataloaders
from models import DSFANet, Autoencoder, LSTMClassifier
from ensemble import UnificationLayer, EnsembleManager

# 通用 PyTorch 训练函数
def train_torch_model(model, train_loader, model_type='classifier', epochs=5, input_req='static'):
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    if model_type == 'classifier':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    
    print(f"Training {model.__class__.__name__} ({input_req})...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_s, x_t, y in train_loader:
            optimizer.zero_grad()
            
            # 准备输入
            if input_req == 'both':
                out = model(x_s, x_t)
                target = y
            elif input_req == 'static':
                out = model(x_s)
                target = y if model_type == 'classifier' else x_s # AE 目标是自身
            elif input_req == 'temporal':
                out = model(x_t)
                target = y
                
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # print(f"  Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
    return model

def main():
    # 1. 加载数据
    csv_path = 'NetFlow_v3_Features.csv'
    try:
        preprocessor = DataPreprocessor(csv_path)
        (X_s_train, X_t_train, y_train), (X_s_test, X_t_test, y_test) = preprocessor.prepare_data()
        
        # 划分一个小的验证集用于 Unification 参数校准 (简单起见，这里直接用测试集的一部分或训练集)
        # 实际操作中应使用单独的 val set
        X_s_val, X_t_val = X_s_train[:1000], X_t_train[:1000] 
        
        train_loader, test_loader = get_dataloaders(
            (X_s_train, X_t_train, y_train), 
            (X_s_test, X_t_test, y_test), 
            config.BATCH_SIZE
        )
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    # 维度信息
    static_dim = X_s_train.shape[1]
    temporal_dim = X_t_train.shape[1]
    num_classes = config.NUM_CLASSES

    # ----------------------------------------
    # 2. 实例化并训练模型
    # ----------------------------------------

    # A. DSFANet (PyTorch, Input: Both)
    dsfanet = DSFANet(static_dim, temporal_dim, num_classes)
    dsfanet = train_torch_model(dsfanet, train_loader, model_type='classifier', input_req='both', epochs=3)

    # B. Autoencoder (PyTorch, Input: Static)
    ae = Autoencoder(static_dim)
    ae = train_torch_model(ae, train_loader, model_type='anomaly', input_req='static', epochs=3)

    # C. LSTM (PyTorch, Input: Temporal)
    lstm = LSTMClassifier(temporal_dim, num_classes)
    lstm = train_torch_model(lstm, train_loader, model_type='classifier', input_req='temporal', epochs=3)

    # D. Random Forest (Sklearn, Input: Static)
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    rf.fit(X_s_train, y_train)

    # E. SVM (Sklearn, Input: Static)
    # 注意: probability=True 是必须的，以便获取置信度分数
    print("Training SVM...")
    svm = SVC(probability=True, kernel='rbf', max_iter=1000) # max_iter限制用于加速演示
    svm.fit(X_s_train, y_train)

    # ----------------------------------------
    # 3. 构建集成系统
    # ----------------------------------------
    unifier = UnificationLayer()
    ensemble = EnsembleManager(unifier)

    # 添加模型，指定不同的 input_req 和 model_type
    ensemble.add_model('DSFANet', dsfanet, weight=0.3, model_type='classifier', input_req='both')
    ensemble.add_model('Autoencoder', ae, weight=0.1, model_type='anomaly', input_req='static')
    ensemble.add_model('LSTM', lstm, weight=0.2, model_type='classifier', input_req='temporal')
    ensemble.add_model('RandomForest', rf, weight=0.2, model_type='classifier', input_req='static')
    ensemble.add_model('SVM', svm, weight=0.2, model_type='classifier', input_req='static')

    # 4. 校准 Unification Layer (计算 mu, sigma)
    ensemble.calibrate_unifier(X_s_val, X_t_val)

    # ----------------------------------------
    # 4. 最终评估
    # ----------------------------------------
    print("\n--- Evaluating Ensemble on Test Set ---")
    final_scores = ensemble.predict(X_s_test, X_t_test)
    
    # 简单的阈值判定 (Z-Score 空间)
    # 经过 Unification，分数大概是 N(0, 1) 分布，正态样本在 0 附近
    # 如果是攻击概率转换来的 z-score，可能会偏离
    # 这里我们简化：取 Top-k 或 设定阈值。
    # 由于 Unification 混合了 MSE 和 概率，如果攻击类产生高 MSE 和 高概率，则 score 越高越可能是攻击。
    
    # 将 Score 映射回 Label 0/1 需要确定阈值。
    # 这里仅打印分数的统计信息，或假设 > 0 (均值) 为异常
    predictions = (final_scores > 0.5).astype(int) 
    # 注意：Z-score 0.5 意味着高于平均值 0.5 个标准差，这只是一个示例阈值
    
    acc = np.mean(predictions == y_test)
    print(f"Ensemble Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()