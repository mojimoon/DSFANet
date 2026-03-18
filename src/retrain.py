

import argparse
from copy import deepcopy
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from . import config
from .active_learning import ActiveLearner, fit_uncertainty_stats_from_binary_probs, select_indices_by_metric
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


def retrain_model_generic(
    model_name,
    model,
    x_s_train,
    x_t_train,
    y_train,
    drift_s,
    drift_t,
    drift_y,
    metric,
    budget_ratio,
    id_ratio,
    get_probs_and_features: Callable,
    get_model_input: Callable,
    device="cpu",
    t_stream_dim: int | None = None,
) -> tuple[object, np.ndarray]:
    """Generic selective retraining for AE/LSTM/DSFANet.

    Args:
        get_probs_and_features: model -> (probs, features)
        get_model_input: (input_req, x_s, x_t, t_stream_dim) -> model_input
        t_stream_dim: Optional temporal width for combined_no_ts input mode.

    Returns:
        retrained_model: object
        probs_after: np.ndarray
    """
    model = deepcopy(model)

    probs, features = get_probs_and_features(
        model_name=model_name,
        model=model,
        x_s=drift_s,
        x_t=drift_t,
        device=device,
        t_stream_dim=t_stream_dim,
        need_features=True,
    )

    # Train stats for ensemble_p_value do not require full-train/full-feature inference.
    max_stats_samples = 20000 if model_name == "AE" else 12000
    if len(y_train) > max_stats_samples:
        stats_idx = np.random.choice(len(y_train), size=max_stats_samples, replace=False)
        x_s_stats = x_s_train[stats_idx]
        x_t_stats = x_t_train[stats_idx]
    else:
        x_s_stats = x_s_train
        x_t_stats = x_t_train

    try:
        train_probs, _ = get_probs_and_features(
            model_name=model_name,
            model=model,
            x_s=x_s_stats,
            x_t=x_t_stats,
            device=device,
            t_stream_dim=t_stream_dim,
            need_features=False,
        )
    except TypeError:
        # Backward compatibility for old callback signatures.
        train_probs, _ = get_probs_and_features(
            model_name=model_name,
            model=model,
            x_s=x_s_stats,
            x_t=x_t_stats,
            device=device,
            t_stream_dim=t_stream_dim,
        )
    train_stats = fit_uncertainty_stats_from_binary_probs(train_probs)
    selected = select_indices_by_metric(metric, probs, budget_ratio, features=features, train_stats=train_stats)
    if selected.size == 0:
        selected = np.array([int(np.argmax(np.asarray(probs)))], dtype=int)

    id_count = int(max(1, round(len(selected) * id_ratio)))
    id_count = min(id_count, len(y_train))
    replay_idx = np.random.choice(len(y_train), size=id_count, replace=False)

    if model_name == "AE":
        ae_t_stream_dim = t_stream_dim
        if ae_t_stream_dim is None:
            try:
                expected_dim = int(model.encoder[0].in_features)
                inferred = expected_dim - int(drift_s.shape[1])
                if 0 <= inferred <= int(drift_t.shape[1]):
                    ae_t_stream_dim = inferred
            except Exception:
                ae_t_stream_dim = t_stream_dim

        drift_input = get_model_input("combined_no_ts", drift_s, drift_t, t_stream_dim=ae_t_stream_dim)
        train_input = get_model_input("combined_no_ts", x_s_train, x_t_train, t_stream_dim=ae_t_stream_dim)
        retrain_s = np.concatenate([drift_input[selected], train_input[replay_idx]])
        x = torch.tensor(retrain_s, dtype=torch.float32, device=device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        model.train()
        for _ in range(3):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, x)
            loss.backward()
            optimizer.step()

        p_after, _ = get_probs_and_features(
            model_name,
            model,
            drift_s,
            drift_t,
            device,
            t_stream_dim=t_stream_dim,
        )
        return model, p_after

    retrain_s = np.concatenate([drift_s[selected], x_s_train[replay_idx]])
    if model_name == "LSTM":
        drift_t_retrain = get_model_input("combined_all", drift_s, drift_t)
        train_t_retrain = get_model_input("combined_all", x_s_train, x_t_train)
        retrain_t = np.concatenate([drift_t_retrain[selected], train_t_retrain[replay_idx]])
    else:
        retrain_t = np.concatenate([drift_t[selected], x_t_train[replay_idx]])
    retrain_y = np.concatenate([drift_y[selected], y_train[replay_idx]])

    if model_name == "LSTM":
        retrain_epochs = 8
        retrain_lr = 3e-4
    elif model_name == "DSFANet":
        retrain_epochs = 6
        retrain_lr = 2e-4
    else:
        retrain_epochs = 3
        retrain_lr = 1e-4

    class_counts = np.bincount(retrain_y.astype(np.int64), minlength=config.NUM_CLASSES).astype(np.float32)
    class_counts[class_counts == 0] = 1.0
    class_weights = class_counts.sum() / (config.NUM_CLASSES * class_counts)

    optimizer = optim.Adam(model.parameters(), lr=retrain_lr)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    model.train()

    bs = 32 if model_name == "LSTM" else 64
    for _ in range(retrain_epochs):
        perm = np.random.permutation(len(retrain_y))
        for i in range(0, len(retrain_y), bs):
            idx = perm[i : i + bs]
            xs = torch.tensor(retrain_s[idx], dtype=torch.float32, device=device)
            xt = torch.tensor(retrain_t[idx], dtype=torch.float32, device=device)
            yy = torch.tensor(retrain_y[idx], dtype=torch.long, device=device)
            optimizer.zero_grad()
            if model_name == "LSTM":
                logits = model(xt)
            else:
                logits = model(xs, xt)
            loss = criterion(logits, yy)
            loss.backward()
            if model_name in ["LSTM", "DSFANet"]:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    try:
        p, _ = get_probs_and_features(
            model_name,
            model,
            drift_s,
            drift_t,
            device,
            t_stream_dim=t_stream_dim,
            need_features=False,
        )
    except TypeError:
        p, _ = get_probs_and_features(
            model_name,
            model,
            drift_s,
            drift_t,
            device,
            t_stream_dim=t_stream_dim,
        )
    return model, p


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
