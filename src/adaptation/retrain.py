from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from src import config
from src.adaptation.active_learning import ActiveLearner
from src.data.data_loader import DataPreprocessor, IDSDataset, get_dataloaders
from src.models.networks import DSFANet
from src.robustness.drift_tester import DriftGenerator


def evaluate_model(model, dataloader, device="cpu"):
    """Compute classification accuracy for a dataloader."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_s, x_t, y in dataloader:
            x_s = x_s.to(device)
            x_t = x_t.to(device)

            logits = model(x_s, x_t)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    return accuracy_score(all_labels, all_preds)


def train_one_epoch(model, loader, optimizer, criterion, device="cpu"):
    model.train()
    total_loss = 0.0

    for x_s, x_t, y in loader:
        x_s = x_s.to(device)
        x_t = x_t.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x_s, x_t)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    csv_path = "NetFlow_v3_Features.csv"
    try:
        preprocessor = DataPreprocessor(csv_path)
        (x_s_train, x_t_train, y_train), (x_s_test, x_t_test, y_test) = preprocessor.prepare_data()
    except Exception as exc:
        print(f"Data init failed: {exc}")
        return

    static_dim = x_s_train.shape[1]
    temporal_dim = x_t_train.shape[1]

    print("\n[Phase 1] Initial training of DSFANet...")
    model = DSFANet(static_dim, temporal_dim, config.NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = get_dataloaders(
        (x_s_train, x_t_train, y_train),
        (x_s_test, x_t_test, y_test),
        batch_size=64,
    )

    for epoch in range(3):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"  Epoch {epoch + 1}: Loss {loss:.4f}")

    initial_acc = evaluate_model(model, test_loader, device)
    print(f"  Initial Accuracy: {initial_acc:.4f}")

    print("\n[Phase 2] Simulating concept drift (adversarial FGSM)...")
    drifter = DriftGenerator()

    subset_idx = np.random.choice(len(y_test), 1000, replace=False)
    drift_x_s, drift_x_t, drift_y = drifter.simulate_adversarial(
        model.cpu(),
        x_s_test[subset_idx],
        x_t_test[subset_idx],
        y_test[subset_idx],
        method="fgsm",
        epsilon=0.1,
    )
    model.to(device)

    drift_dataset = IDSDataset(drift_x_s, drift_x_t, drift_y)
    drift_loader = torch.utils.data.DataLoader(drift_dataset, batch_size=64)
    drift_acc = evaluate_model(model, drift_loader, device)
    print(f"  Accuracy on drifted candidate data: {drift_acc:.4f} (before retraining)")

    print("\n[Phase 3] Selective retraining comparison")
    budget_ratio = 0.3
    learner = ActiveLearner(model, device)

    metrics = ["random", "deep_gini", "entropy"]
    results = []
    initial_state = deepcopy(model.state_dict())

    for metric in metrics:
        print(f"\n--- Retraining strategy: {metric.upper()} ---")

        model.load_state_dict(initial_state)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        if metric == "random":
            indices = learner.select_random(drift_x_s, drift_x_t, budget_ratio)
        elif metric == "deep_gini":
            indices = learner.select_deep_gini(drift_x_s, drift_x_t, budget_ratio)
        else:
            indices = learner.select_entropy(drift_x_s, drift_x_t, budget_ratio)

        print(f"  Selected {len(indices)} samples out of {len(drift_y)} candidates.")

        mix_idx = np.random.choice(len(y_train), len(indices), replace=False)
        retrain_x_s = np.concatenate([drift_x_s[indices], x_s_train[mix_idx]])
        retrain_x_t = np.concatenate([drift_x_t[indices], x_t_train[mix_idx]])
        retrain_y = np.concatenate([drift_y[indices], y_train[mix_idx]])

        retrain_loader, _ = get_dataloaders(
            (retrain_x_s, retrain_x_t, retrain_y),
            (retrain_x_s, retrain_x_t, retrain_y),
            batch_size=32,
        )

        for _ in range(3):
            train_one_epoch(model, retrain_loader, optimizer, criterion, device)

        new_drift_acc = evaluate_model(model, drift_loader, device)
        print(f"  Result -> accuracy on candidate set: {new_drift_acc:.4f}")
        results.append({"Strategy": metric, "Accuracy": new_drift_acc})

    print("\n--- Final Comparison ---")
    print(pd.DataFrame(results))


if __name__ == "__main__":
    main()
