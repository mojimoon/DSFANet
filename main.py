import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import config
from data_loader import DataPreprocessor, get_dataloaders
from models import DSFANet, Autoencoder, LSTMClassifier
from ensemble import UnificationLayer, VotingEnsemble, StackingEnsemble

# 通用 PyTorch 训练函数
def train_torch_model(model, train_loader, model_type='classifier', epochs=5, input_req='static'):
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss() if model_type == 'classifier' else nn.MSELoss()
    
    print(f"Training {model.__class__.__name__} ({input_req})...")
    model.train()
    for epoch in range(epochs):
        for x_s, x_t, y in train_loader:
            optimizer.zero_grad()
            if input_req == 'both':
                out = model(x_s, x_t)
                target = y
            elif input_req == 'static':
                out = model(x_s)
                target = y if model_type == 'classifier' else x_s
            elif input_req == 'temporal':
                out = model(x_t)
                target = y
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
    return model

def main():
    # 1. Preparation
    csv_path = 'NF-UNSW-NB15-v3.csv'
    try:
        preprocessor = DataPreprocessor(csv_path)
        (X_s_train, X_t_train, y_train), (X_s_test, X_t_test, y_test) = preprocessor.prepare_data()
        
        # Split a Validation set for Stacking Meta-Training & Calibration
        val_size = int(len(X_s_train) * 0.2)
        X_s_val, X_t_val, y_val = X_s_train[:val_size], X_t_train[:val_size], y_train[:val_size]
        X_s_train_sub, X_t_train_sub, y_train_sub = X_s_train[val_size:], X_t_train[val_size:], y_train[val_size:]

        train_loader, _ = get_dataloaders(
            (X_s_train_sub, X_t_train_sub, y_train_sub), 
            (X_s_test, X_t_test, y_test), 
            config.BATCH_SIZE
        )
    except Exception as e:
        print(f"Data Init Failed: {e}")
        return

    # 2. Train Base Models (Tier-1)
    static_dim = X_s_train.shape[1]
    temporal_dim = X_t_train.shape[1]
    
    dsfanet = DSFANet(static_dim, temporal_dim, config.NUM_CLASSES)
    train_torch_model(dsfanet, train_loader, 'classifier', 3, 'both')
    
    ae = Autoencoder(static_dim)
    train_torch_model(ae, train_loader, 'anomaly', 3, 'static')
    
    lstm = LSTMClassifier(temporal_dim, config.NUM_CLASSES)
    train_torch_model(lstm, train_loader, 'classifier', 3, 'temporal')
    
    print("Training RF & SVM...")
    rf = RandomForestClassifier(n_estimators=50, max_depth=10).fit(X_s_train_sub, y_train_sub)
    svm = SVC(probability=True, kernel='rbf', max_iter=1000).fit(X_s_train_sub, y_train_sub)

    # print accuracy
    accs = {
        'RF': rf.score(X_s_test, y_test),
        'SVM': svm.score(X_s_test, y_test)
    }
    # how do you test torch models?
    with torch.no_grad():
        dsfanet.eval()
        dsfanet_preds = dsfanet(torch.tensor(X_s_test, dtype=torch.float32), torch.tensor(X_t_test, dtype=torch.float32))
        dsfanet_acc = (dsfanet_preds.argmax(dim=1).numpy() == y_test).mean()
        accs['DSFANet'] = dsfanet_acc

        lstm.eval()
        lstm_preds = lstm(torch.tensor(X_t_test, dtype=torch.float32))
        lstm_acc = (lstm_preds.argmax(dim=1).numpy() == y_test).mean()
        accs['LSTM'] = lstm_acc

        ae.eval()
        ae_recon = ae(torch.tensor(X_s_test, dtype=torch.float32))
        ae_errors = np.mean((ae_recon.numpy() - X_s_test) ** 2, axis=1)
        # Simple thresholding for anomaly detection
        threshold = np.percentile(ae_errors, 95)
        ae_preds = (ae_errors > threshold).astype(int)
        ae_acc = (ae_preds == y_test).mean()
        accs['Autoencoder'] = ae_acc
    print("Base Model Accuracies:", accs)

    # 3. Initialize Ensembles
    unifier = UnificationLayer()
    
    # We can share base models between different ensemble strategies
    # Define models dict for easy adding
    models_config = [
        ('DSFANet', dsfanet, 'classifier', 'both'),
        ('Autoencoder', ae, 'anomaly', 'static'),
        ('LSTM', lstm, 'classifier', 'temporal'),
        ('RF', rf, 'classifier', 'static'),
        ('SVM', svm, 'classifier', 'static')
    ]

    # --- Strategy A: Voting ---
    voting_ens = VotingEnsemble(unifier, weights={'DSFANet': 2.0, 'RF': 1.5, 'Autoencoder': 1.0})
    for m in models_config: voting_ens.add_model(*m)
    
    # --- Strategy B: Stacking ---
    stacking_ens = StackingEnsemble(unifier)
    for m in models_config: stacking_ens.add_model(*m)

    # 4. Calibration & Meta-Training (Using Validation Set)
    # Both ensembles need calibrated unifiers
    voting_ens.calibrate(X_s_val, X_t_val) 
    
    # Stacking specifically needs to "fit" the meta-learner on val data
    # (Note: stacking_ens shares the unifier, so it's already calibrated by the line above)
    stacking_ens.fit_meta(X_s_val, X_t_val, y_val)

    # 5. Evaluation
    print("\n--- Evaluation on Test Set ---")
    
    # Test Voting
    voting_scores = voting_ens.predict(X_s_test, X_t_test)
    voting_acc = np.mean((voting_scores > 0.5) == y_test)
    print(f"Voting Ensemble Accuracy: {voting_acc:.4f}")
    
    # Observe Intermediate Results (Why did it decide that?)
    intermediates = voting_ens.get_intermediate_results()
    print("  Sample Intermediate Scores (first 2 samples):")
    for name, scores in intermediates.items():
        print(f"    {name}: {scores[:2]}")

    # Test Stacking
    stack_scores = stacking_ens.predict(X_s_test, X_t_test)
    stack_acc = np.mean((stack_scores > 0.5) == y_test)
    print(f"Stacking Ensemble Accuracy: {stack_acc:.4f}")

if __name__ == "__main__":
    main()