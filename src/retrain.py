

import argparse
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from . import config
from .active_learning import ActiveLearner
from .data_loader import IDSDataset, DataPreprocessor, get_dataloaders
from .drift_tester import DriftGenerator
from .models import DSFANet
from .runtime import resolve_device


def evaluate_model(model, dataloader, device="cpu"):
    """Evaluate classification accuracy on a dataloader.

    Returns:
        acc: float
    """
    device = resolve_device(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_s, x_t, y in dataloader:
            x_s, x_t = x_s.to(device), x_t.to(device)
            logits = model(x_s, x_t)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    return accuracy_score(all_labels, all_preds)


def train_one_epoch(model, loader, optimizer, criterion, device="cpu"):
    """Train one epoch and return mean batch loss.

    Returns:
        mean_loss: float
    """
    device = resolve_device(device)
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


def run_retraining_comparison(
    model,
    drift_x_s,
    drift_x_t,
    drift_y,
    x_s_train,
    x_t_train,
    y_train,
    budget_ratio=0.3,
    id_ratio=1.0,
    strategies=None,
    retrain_epochs=3,
    device="cpu",
):
    """Run strategy-wise selective retraining and report post-retrain accuracy.

    Returns:
        df: pd.DataFrame
    """
    if strategies is None:
        strategies = ["random", "deep_gini", "entropy"]

    learner = ActiveLearner(model, str(device))
    initial_state = deepcopy(model.state_dict())
    criterion = nn.CrossEntropyLoss()
    results = []

    for metric in strategies:
        print(f"\n--- Retraining Strategy: {metric.upper()} ---")
        model.load_state_dict(initial_state)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        if metric == "random":
            indices = learner.select_random(drift_x_s, drift_x_t, budget_ratio)
        elif metric == "deep_gini":
            indices = learner.select_deep_gini(drift_x_s, drift_x_t, budget_ratio)
        elif metric == "entropy":
            indices = learner.select_entropy(drift_x_s, drift_x_t, budget_ratio)
        elif metric == "gd":
            indices = learner.select_geometric_diversity(drift_x_s, drift_x_t, budget_ratio)
        elif metric == "ensemble_rank":
            indices = learner.select_ensemble_rank(drift_x_s, drift_x_t, budget_ratio)
        elif metric == "ensemble_hybrid":
            indices = learner.select_ensemble_hybrid(drift_x_s, drift_x_t, budget_ratio)
        else:
            indices = np.array([], dtype=int)

        n_drift = len(indices)
        n_id = int(max(1, round(n_drift * id_ratio))) if n_drift > 0 else 0
        n_id = min(n_id, len(y_train))

        print(f"  Selected drift={n_drift}, ID replay={n_id}.")

        if n_drift == 0:
            results.append({"Strategy": metric, "Accuracy": np.nan, "selected": 0, "id_replay": 0})
            continue

        mix_idx = np.random.choice(len(y_train), n_id, replace=False)
        retrain_x_s = np.concatenate([drift_x_s[indices], x_s_train[mix_idx]])
        retrain_x_t = np.concatenate([drift_x_t[indices], x_t_train[mix_idx]])
        retrain_y = np.concatenate([drift_y[indices], y_train[mix_idx]])

        retrain_loader, _ = get_dataloaders(
            (retrain_x_s, retrain_x_t, retrain_y),
            (retrain_x_s, retrain_x_t, retrain_y),
            batch_size=32,
        )

        for _ in range(retrain_epochs):
            train_one_epoch(model, retrain_loader, optimizer, criterion, device)

        drift_dataset = IDSDataset(drift_x_s, drift_x_t, drift_y)
        drift_loader = torch.utils.data.DataLoader(drift_dataset, batch_size=64)
        new_drift_acc = evaluate_model(model, drift_loader, device)
        print(f"  Result -> accuracy on candidate set: {new_drift_acc:.4f}")

        results.append(
            {
                "Strategy": metric,
                "Accuracy": new_drift_acc,
                "selected": int(n_drift),
                "id_replay": int(n_id),
            }
        )

    return pd.DataFrame(results)


def main(device="cpu", budget_ratio=0.3, id_ratio=1.0, metrics=None):
    device = resolve_device(device)
    print(f"Using device: {device}")

    csv_path = "NetFlow_v3_Features.csv"
    try:
        preprocessor = DataPreprocessor(csv_path)
        (x_s_train, x_t_train, y_train), (x_s_test, x_t_test, y_test) = preprocessor.prepare_data()
    except Exception as e:
        print(f"Data Init Failed: {e}")
        return

    static_dim = x_s_train.shape[1]
    temporal_dim = x_t_train.shape[1]

    print("\n[Phase 1] Initial Training of DSFANet...")
    model = DSFANet(static_dim, temporal_dim, config.NUM_CLASSES, device=str(device)).to(device)
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

    print("\n[Phase 2] Simulating Concept Drift (Adversarial FGSM)...")
    drifter = DriftGenerator()
    subset_idx = np.random.choice(len(y_test), 1000, replace=False)

    drift_x_s, drift_x_t, drift_y = drifter.simulate_adversarial(
        model,
        x_s_test[subset_idx],
        x_t_test[subset_idx],
        y_test[subset_idx],
        method="fgsm",
        epsilon=0.1,
        device=str(device),
    )

    drift_dataset = IDSDataset(drift_x_s, drift_x_t, drift_y)
    drift_loader = torch.utils.data.DataLoader(drift_dataset, batch_size=64)
    drift_acc = evaluate_model(model, drift_loader, device)
    print(f"  Accuracy on Drifted/Candidate Data: {drift_acc:.4f} (Baseline before retrain)")

    print("\n[Phase 3] Selective Retraining Comparison")
    if metrics is None:
        metrics = ["random", "deep_gini", "entropy"]

    df_res = run_retraining_comparison(
        model=model,
        drift_x_s=drift_x_s,
        drift_x_t=drift_x_t,
        drift_y=drift_y,
        x_s_train=x_s_train,
        x_t_train=x_t_train,
        y_train=y_train,
        budget_ratio=budget_ratio,
        id_ratio=id_ratio,
        strategies=metrics,
        retrain_epochs=3,
        device=device,
    )

    print("\n--- Final Comparison ---")
    print(df_res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retraining experiment for DSFANet")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--budget-ratio", type=float, default=0.3)
    parser.add_argument("--id-ratio", type=float, default=1.0)
    parser.add_argument("--metrics", default="random,deep_gini,entropy")
    args = parser.parse_args()

    metric_list = [m.strip() for m in args.metrics.split(",") if m.strip()]
    main(device=args.device, budget_ratio=args.budget_ratio, id_ratio=args.id_ratio, metrics=metric_list)
