import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from copy import deepcopy

import config
from preprocessing import DataPreprocessor, get_dataloaders, IDSDataset
from models import DSFANet
from drift_tester import DriftGenerator
from active_learning import ActiveLearner

def evaluate_model(model, dataloader, device='cpu'):
    """Helper to calculate accuracy on a loader"""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_s, x_t, y in dataloader:
            if isinstance(x_s, torch.Tensor): x_s = x_s.to(device)
            if isinstance(x_t, torch.Tensor): x_t = x_t.to(device)
            if isinstance(y, torch.Tensor): y = y.to('cpu')
            
            logits = model(x_s, x_t)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    return accuracy_score(all_labels, all_preds)

def train_one_epoch(model, loader, optimizer, criterion, device='cpu'):
    """Helper for a single training epoch"""
    model.train()
    total_loss = 0
    for x_s, x_t, y in loader:
        x_s, x_t, y = x_s.to(device), x_t.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x_s, x_t)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    csv_path = 'NetFlow_v3_Features.csv'
    try:
        preprocessor = DataPreprocessor(csv_path)
        (X_s_train, X_t_train, y_train), (X_s_test, X_t_test, y_test) = preprocessor.prepare_data()
    except Exception as e:
        print(f"Data Init Failed: {e}")
        return

    # 2. Initial Training
    static_dim = X_s_train.shape[1]
    temporal_dim = X_t_train.shape[1]
    
    print("\n[Phase 1] Initial Training of DSFANet...")
    model = DSFANet(static_dim, temporal_dim, config.NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    train_loader, test_loader = get_dataloaders(
        (X_s_train, X_t_train, y_train), 
        (X_s_test, X_t_test, y_test), 
        batch_size=64
    )
    
    # Train for a few epochs
    for epoch in range(3):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"  Epoch {epoch+1}: Loss {loss:.4f}")
        
    initial_acc = evaluate_model(model, test_loader, device)
    print(f"  Initial Accuracy: {initial_acc:.4f}")

    # 3. Simulate Concept Drift (Adversarial Shift)
    print("\n[Phase 2] Simulating Concept Drift (Adversarial FGSM)...")
    drifter = DriftGenerator()
    
    # Select specific target subset to attack (e.g. 1000 samples) to create "Candidate Set"
    subset_idx = np.random.choice(len(y_test), 1000, replace=False)
    
    # Generate adversarial examples (using CPU for generation simplicity in loop)
    # This simulates "Drifted" or "Hard" traffic appearing
    drift_x_s, drift_x_t, drift_y = drifter.simulate_adversarial(
        model.cpu(), 
        X_s_test[subset_idx], 
        X_t_test[subset_idx], 
        y_test[subset_idx], 
        method='fgsm', epsilon=0.1
    )
    model.to(device) # Move model back
    
    # Evaluate model on this new "Candidate/Drift" dataset
    drift_dataset = IDSDataset(drift_x_s, drift_x_t, drift_y)
    drift_loader = torch.utils.data.DataLoader(drift_dataset, batch_size=64)
    drift_acc = evaluate_model(model, drift_loader, device)
    print(f"  Accuracy on Drifted/Candidate Data: {drift_acc:.4f} (Baseline before retrain)")

    # 4. Selective Retraining
    print("\n[Phase 3] Selective Retraining Comparison")
    budget_ratio = 0.3 # We can only label/retrain on 30% of the drift data
    learner = ActiveLearner(model, device)
    
    metrics = ['random', 'deep_gini', 'entropy']
    results = []
    
    # Save state to reset between metrics
    initial_state = deepcopy(model.state_dict())

    for metric in metrics:
        print(f"\n--- Retraining Strategy: {metric.upper()} ---")
        
        # Reset model
        model.load_state_dict(initial_state)
        # Use lower LR for fine-tuning
        optimizer = optim.Adam(model.parameters(), lr=0.0001) 
        
        # Selection Step
        if metric == 'random':
            indices = learner.select_random(drift_x_s, drift_x_t, budget_ratio)
        elif metric == 'deep_gini':
            indices = learner.select_deep_gini(drift_x_s, drift_x_t, budget_ratio)
        elif metric == 'entropy':
            indices = learner.select_entropy(drift_x_s, drift_x_t, budget_ratio)
            
        print(f"  Selected {len(indices)} samples out of {len(drift_y)} candidates.")
        
        # Prepare Retrain Data
        # Strategy: Mix selected drift data w/ small subset of original data (Replay)
        mix_idx = np.random.choice(len(y_train), len(indices), replace=False)
        
        retrain_x_s = np.concatenate([drift_x_s[indices], X_s_train[mix_idx]])
        retrain_x_t = np.concatenate([drift_x_t[indices], X_t_train[mix_idx]])
        retrain_y = np.concatenate([drift_y[indices], y_train[mix_idx]])
        
        retrain_loader, _ = get_dataloaders((retrain_x_s, retrain_x_t, retrain_y), (retrain_x_s, retrain_x_t, retrain_y), batch_size=32)
        
        # Retrain Loop
        for epoch in range(3):
            train_one_epoch(model, retrain_loader, optimizer, criterion, device)
        
        # Final Evaluation
        new_drift_acc = evaluate_model(model, drift_loader, device)
        print(f"  Result -> accuracy on candidate set: {new_drift_acc:.4f}")
        
        results.append({
            'Strategy': metric,
            'Accuracy': new_drift_acc
        })

    print("\n--- Final Comparison ---")
    df_res = pd.DataFrame(results)
    print(df_res)

if __name__ == "__main__":
    main()
