import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src import config
from src.data_loader import DataPreprocessor, extract_benign_samples, get_dataloaders
from src.drift_tester import DriftGenerator
from src.models import Autoencoder, DSFANet, LSTMClassifier
from src.models.ensemble import StackingEnsemble, UnificationLayer, VotingEnsemble, XGBoostStackingEnsemble
from src.retrain import retrain_model_generic
from src.runtime import resolve_device
from src.shap_analysis import analyze_ae_shap, analyze_dsfanet_shap, analyze_lstm_shap, train_autoencoder_model, train_lstm_model

DEFAULT_DATASETS = [
    "NF-UNSW-NB15-v3.csv",
    "NF-ToN-IoT-v3.csv",
    "NF-CICIDS2018-v3.csv",
    "NF-BoT-IoT-v3.csv",
]


def slug(text: str) -> str:
    """Convert dataset or model name into a filesystem-safe slug.

    Returns:
        slug_text: str
    """
    return text.replace(".csv", "").replace(".", "_").replace("-", "_").replace(" ", "_")


def ensure_dir(path: Path) -> Path:
    """Create a directory recursively if it does not exist.

    Returns:
        path: Path
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_float_list(value) -> list[float]:
    """Parse a comma-separated string into float values.

    Returns:
        values: list[float]
    """
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_str_list(value) -> list[str]:
    """Parse a comma-separated string into trimmed strings.

    Returns:
        values: list[str]
    """
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_int_list(value) -> list[int]:
    """Parse a comma-separated string into int values.

    Returns:
        values: list[int]
    """
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def stratified_sample_indices(y, max_samples, seed=42) -> np.ndarray:
    """Return sampled indices with class-ratio preservation when possible.

    Returns:
        sampled_idx: np.ndarray
    """
    total = len(y)
    if max_samples <= 0 or total <= max_samples:
        return np.arange(total)

    all_idx = np.arange(total)
    try:
        sampled_idx, _ = train_test_split(
            all_idx,
            train_size=max_samples,
            random_state=seed,
            stratify=y,
        )
        return np.sort(sampled_idx)
    except Exception:
        return np.sort(np.random.RandomState(seed).choice(total, size=max_samples, replace=False))


def metric_row(y_true, y_prob, y_pred: np.ndarray | None = None) -> dict[str, float]:
    """Compute binary classification metrics from labels and probabilities.

    Args:
        y_pred: Optional predicted labels. If omitted, threshold 0.5 is used.

    Returns:
        metrics: dict[str, float]
    """
    if y_pred is None:
        y_pred = (y_prob >= 0.5).astype(int)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "ap": float(average_precision_score(y_true, y_prob)),
    }


def combine_static_temporal(x_s, x_t, t_stream_dim: int | None = None) -> np.ndarray:
    """Concatenate static and temporal features.

    Args:
        t_stream_dim: Optional temporal width used to trim timestamp columns.

    Returns:
        x_combined: np.ndarray
    """
    if t_stream_dim is None:
        x_t_use = x_t
    else:
        x_t_use = x_t[:, :t_stream_dim]
    return np.concatenate([x_s, x_t_use], axis=1)


def get_model_input(input_req, x_s, x_t, t_stream_dim: int | None = None) -> np.ndarray:
    """Build model-specific input arrays for sklearn/torch models.

    Args:
        t_stream_dim: Optional temporal width for no-timestamp mode.

    Returns:
        x_in: np.ndarray
    """
    if input_req == "combined_all":
        return combine_static_temporal(x_s, x_t, t_stream_dim=None)
    if input_req == "combined_no_ts":
        return combine_static_temporal(x_s, x_t, t_stream_dim=t_stream_dim)
    if input_req == "temporal":
        return x_t
    return x_s


def get_model_input_batch(input_req, x_s_batch, x_t_batch, t_stream_dim: int | None = None) -> np.ndarray:
    """Batch variant of model input builder with float32 cast.

    Args:
        t_stream_dim: Optional temporal width for no-timestamp mode.

    Returns:
        x_in_batch: np.ndarray
    """
    if input_req == "combined_all":
        return combine_static_temporal(x_s_batch, x_t_batch, t_stream_dim=None).astype(np.float32, copy=False)
    if input_req == "combined_no_ts":
        return combine_static_temporal(x_s_batch, x_t_batch, t_stream_dim=t_stream_dim).astype(np.float32, copy=False)
    if input_req == "temporal":
        return x_t_batch.astype(np.float32, copy=False)
    return x_s_batch.astype(np.float32, copy=False)


def _torch_eval_batch_size(device="cpu", preferred=2048) -> int:
    """Get a conservative evaluation batch size for current device.

    Returns:
        batch_size: int
    """
    device_type = device.type if hasattr(device, "type") else str(device)
    if str(device_type).lower().startswith("cuda"):
        return min(preferred, 1024)
    return max(preferred, 2048)


def _iter_numpy_batches(*arrays: np.ndarray, batch_size: int):
    """Yield aligned numpy mini-batches from one or more arrays."""
    total = arrays[0].shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        yield tuple(arr[start:end] for arr in arrays)


def save_predictions(out_dir: Path, dataset_key, model_name, y_true, y_prob) -> Path:
    """Write per-sample prediction results to CSV.

    Returns:
        pred_path: Path
    """
    y_pred = (y_prob >= 0.5).astype(int)
    df = pd.DataFrame(
        {
            "sample_id": np.arange(len(y_true), dtype=int),
            "y_true": y_true.astype(int),
            "y_pred": y_pred.astype(int),
            "y_prob": y_prob.astype(float),
        }
    )
    pred_path = out_dir / f"pred_{dataset_key}_{slug(model_name)}.csv"
    df.to_csv(pred_path, index=False)
    return pred_path


def train_dsfanet(
    x_s_train,
    x_t_train,
    y_train,
    device="cpu",
    epochs=2,
) -> DSFANet:
    """Train DSFANet with class-balanced cross-entropy.

    Returns:
        model: DSFANet
    """
    model = DSFANet(
        static_dim=x_s_train.shape[1],
        temporal_dim=x_t_train.shape[1],
        n_classes=config.NUM_CLASSES,
        device=str(device),
    )
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    class_counts = np.bincount(y_train.astype(np.int64), minlength=config.NUM_CLASSES).astype(np.float32)
    class_counts[class_counts == 0] = 1.0
    class_weights = class_counts.sum() / (config.NUM_CLASSES * class_counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    batch_size = 128
    train_loader, _ = get_dataloaders(
        (x_s_train, x_t_train, y_train),
        (x_s_train[:1], x_t_train[:1], y_train[:1]),
        batch_size=batch_size,
    )

    model.train()
    for _ in range(epochs):
        for bx_s, bx_t, by in train_loader:
            bx_s = bx_s.to(device, non_blocking=True)
            bx_t = bx_t.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(bx_s, bx_t)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def torch_probs(
    model,
    x_s,
    x_t,
    input_req,
    device="cpu",
    t_stream_dim: int | None = None,
) -> np.ndarray:
    """Run batched torch inference and return class-1 probabilities.

    Args:
        t_stream_dim: Optional temporal width for no-timestamp mode.

    Returns:
        probs: np.ndarray
    """
    model.eval()
    probs_batches: list[np.ndarray] = []
    batch_size = _torch_eval_batch_size(device)
    with torch.no_grad():
        if input_req == "both":
            for x_s_batch, x_t_batch in _iter_numpy_batches(x_s, x_t, batch_size=batch_size):
                logits = model(
                    torch.tensor(x_s_batch, dtype=torch.float32, device=device),
                    torch.tensor(x_t_batch, dtype=torch.float32, device=device),
                )
                probs_batches.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
        else:
            for x_s_batch, x_t_batch in _iter_numpy_batches(x_s, x_t, batch_size=batch_size):
                x_in_batch = get_model_input_batch(input_req, x_s_batch, x_t_batch, t_stream_dim=t_stream_dim)
                logits = model(torch.tensor(x_in_batch, dtype=torch.float32, device=device))
                probs_batches.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())

    if not probs_batches:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(probs_batches, axis=0)


def ae_probs(model, x_input, ae_min, ae_max, device="cpu") -> np.ndarray:
    """Convert AE reconstruction errors into normalized anomaly probabilities.

    Returns:
        probs: np.ndarray
    """
    batch_size = _torch_eval_batch_size(device)
    recon_batches: list[np.ndarray] = []
    with torch.no_grad():
        for (x_batch,) in _iter_numpy_batches(x_input.astype(np.float32, copy=False), batch_size=batch_size):
            recon_batch = model(torch.tensor(x_batch, dtype=torch.float32, device=device)).detach().cpu().numpy()
            recon_batches.append(recon_batch)
    recon = np.concatenate(recon_batches, axis=0) if recon_batches else np.empty_like(x_input)
    err = np.mean((recon - x_input) ** 2, axis=1)
    denom = max(ae_max - ae_min, 1e-8)
    return np.clip((err - ae_min) / denom, 0.0, 1.0)


def get_model_probs_and_features(
    model_name,
    model,
    x_s,
    x_t,
    device="cpu",
    ae_min: float | None = None,
    ae_max: float | None = None,
    t_stream_dim: int | None = None,
    need_features: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Get probabilities and retraining features for AE, LSTM, and DSFANet.

    Args:
        ae_min: Minimum AE calibration value. If None, infer from current input.
        ae_max: Maximum AE calibration value. If None, infer from current input.
        t_stream_dim: Optional temporal width used when composing AE inputs.
        need_features: For LSTM and DSFANet, control whether to return penultimate layer features.

    Returns:
        probs: np.ndarray
        features: np.ndarray
    """
    if model_name == "AE":
        ae_t_stream_dim = t_stream_dim
        if ae_t_stream_dim is None:
            try:
                expected_dim = int(model.encoder[0].in_features)
                inferred = expected_dim - int(x_s.shape[1])
                if 0 <= inferred <= int(x_t.shape[1]):
                    ae_t_stream_dim = inferred
            except Exception:
                ae_t_stream_dim = t_stream_dim
        ae_input = get_model_input("combined_no_ts", x_s, x_t, t_stream_dim=t_stream_dim)
        if ae_t_stream_dim is not None:
            ae_input = get_model_input("combined_no_ts", x_s, x_t, t_stream_dim=ae_t_stream_dim)
        if ae_min is None or ae_max is None:
            batch_size = _torch_eval_batch_size(device)
            recon_batches: list[np.ndarray] = []
            with torch.no_grad():
                for (x_batch,) in _iter_numpy_batches(ae_input.astype(np.float32, copy=False), batch_size=batch_size):
                    recon_batch = model(torch.tensor(x_batch, dtype=torch.float32, device=device)).detach().cpu().numpy()
                    recon_batches.append(recon_batch)
            recon = np.concatenate(recon_batches, axis=0) if recon_batches else np.empty_like(ae_input)
            raw_err = np.mean((recon - ae_input) ** 2, axis=1)
            ae_min = float(np.min(raw_err))
            ae_max = float(np.max(raw_err))
        probs = ae_probs(model, ae_input, ae_min, ae_max, device)
        if not need_features:
            return probs, np.empty((0, 0), dtype=np.float32)
        features = ae_input
        return probs, features

    if model_name == "LSTM":
        probs = torch_probs(model, x_s, x_t, "combined_all", device, t_stream_dim=None)
        if not need_features:
            return probs, np.empty((0, 0), dtype=np.float32)
        batch_size = _torch_eval_batch_size(device)
        features_batches: list[np.ndarray] = []
        with torch.no_grad():
            for x_s_batch, x_t_batch in _iter_numpy_batches(x_s, x_t, batch_size=batch_size):
                x_batch = get_model_input_batch("combined_all", x_s_batch, x_t_batch, t_stream_dim=None)
                xt = torch.tensor(x_batch, dtype=torch.float32, device=device).unsqueeze(1)
                h_seq, _ = model.lstm(xt)
                features_batches.append(h_seq[:, -1, :].detach().cpu().numpy())
        features = np.concatenate(features_batches, axis=0) if features_batches else np.empty((0, model.hidden_size), dtype=np.float32)
        return probs, features

    probs = torch_probs(model, x_s, x_t, "both", device)
    if not need_features:
        return probs, np.empty((0, 0), dtype=np.float32)
    if hasattr(model, "extract_features"):
        with torch.no_grad():
            features = model.extract_features(
                torch.tensor(x_s, dtype=torch.float32, device=device),
                torch.tensor(x_t, dtype=torch.float32, device=device),
            ).detach().cpu().numpy()
    else:
        features = np.concatenate([x_s, x_t], axis=1)
    return probs, features


def get_raw_score(
    model,
    model_type,
    input_req,
    x_s,
    x_t,
    device="cpu",
    t_stream_dim: int | None = None,
) -> np.ndarray:
    """Get raw model score before optional calibration/unification.

    Args:
        t_stream_dim: Optional temporal width for combined_no_ts paths.

    Returns:
        raw_score: np.ndarray
    """
    if isinstance(model, nn.Module):
        if model_type == "classifier":
            return torch_probs(
                model,
                x_s,
                x_t,
                input_req=input_req,
                device=device,
                t_stream_dim=t_stream_dim,
            )
        ae_t_stream_dim = t_stream_dim
        if ae_t_stream_dim is None and isinstance(model, Autoencoder):
            try:
                expected_dim = int(model.encoder[0].in_features)
                inferred = expected_dim - int(x_s.shape[1])
                if 0 <= inferred <= int(x_t.shape[1]):
                    ae_t_stream_dim = inferred
            except Exception:
                ae_t_stream_dim = t_stream_dim
        ae_input = get_model_input(input_req, x_s, x_t, t_stream_dim=ae_t_stream_dim)
        batch_size = _torch_eval_batch_size(device)
        err_batches: list[np.ndarray] = []
        with torch.no_grad():
            for (x_batch,) in _iter_numpy_batches(ae_input.astype(np.float32, copy=False), batch_size=batch_size):
                recon = model(torch.tensor(x_batch, dtype=torch.float32, device=device)).detach().cpu().numpy()
                err_batches.append(np.mean((recon - x_batch) ** 2, axis=1))
        return np.concatenate(err_batches, axis=0) if err_batches else np.empty((0,), dtype=np.float32)

    x_in = get_model_input(input_req, x_s, x_t, t_stream_dim=t_stream_dim)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_in)[:, 1]
    return model.predict(x_in)


def unify_scores(raw_scores, stats: dict[str, float]) -> np.ndarray:
    """Apply min-max unification for ensemble meta features.

    Returns:
        unified: np.ndarray
    """
    low = stats["min"]
    high = stats["max"]
    if high == low:
        high += 1e-6
    unified = (raw_scores - low) / (high - low)
    return np.clip(unified, 0.0, 1.0)


def train_and_eval_dataset(
    dataset,
    run_dir: Path,
    device="cpu",
    max_train_samples=0,
    ensemble_types: list[str] | None = None,
    epochs: list[int] | None = None,
):
    """Train base models and ensembles for one dataset, then evaluate on clean test set.

    Returns:
        rows: list[dict]
        registry: dict[str, object]
        data_pack: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    """
    dataset_key = slug(dataset)
    if ensemble_types is None:
        ensemble_types = ["voting", "stacking", "xgboost"]
    if epochs is None:
        epochs = [20, 20, 20]

    ds_dir = ensure_dir(run_dir / dataset_key)
    model_dir = ensure_dir(ds_dir / "models")
    pred_dir = ensure_dir(ds_dir / "predictions")

    prep = DataPreprocessor(dataset)
    (x_s_train, x_t_train, y_train), (x_s_test, x_t_test, y_test) = prep.prepare_data()
    t_stream_dim = len(prep.used_t_stream_cols)

    if max_train_samples > 0 and len(y_train) > max_train_samples:
        idx = np.random.RandomState(42).choice(len(y_train), size=max_train_samples, replace=False)
        x_s_train, x_t_train, y_train = x_s_train[idx], x_t_train[idx], y_train[idx]

    val_n = min(max(200, int(0.2 * len(y_train))), len(y_train) - 1)
    x_s_val, x_t_val, y_val = x_s_train[:val_n], x_t_train[:val_n], y_train[:val_n]
    x_s_sub, x_t_sub, y_sub = x_s_train[val_n:], x_t_train[val_n:], y_train[val_n:]

    x_comb_no_ts_sub = combine_static_temporal(x_s_sub, x_t_sub, t_stream_dim=t_stream_dim)
    x_comb_no_ts_test = combine_static_temporal(x_s_test, x_t_test, t_stream_dim=t_stream_dim)

    models = {}
    model_meta = {}

    rf_path = model_dir / f"{dataset_key}_rf.joblib"
    sgd_path = model_dir / f"{dataset_key}_sgd.joblib"
    ae_path = model_dir / f"{dataset_key}_ae.pt"
    lstm_path = model_dir / f"{dataset_key}_lstm.pt"
    dsfa_path = model_dir / f"{dataset_key}_dsfanet.pt"

    if rf_path.exists():
        rf = joblib.load(rf_path)
        print(f"[Learner] Loading existing RandomForest from {rf_path}")
    else:
        time_start = time.time()
        rf = RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42)
        rf.fit(x_comb_no_ts_sub, y_sub)
        joblib.dump(rf, rf_path)
        print(f"[Learner] Trained RandomForest in {time.time() - time_start:.2f} seconds and saved to {rf_path}")

    models["RandomForest"] = rf
    model_meta["RandomForest"] = {
        "path": str(rf_path),
        "model_type": "classifier",
        "input_req": "combined_no_ts",
        "kind": "sklearn",
        "t_stream_dim": t_stream_dim,
    }

    if sgd_path.exists():
        sgd = joblib.load(sgd_path)
        print(f"[Learner] Loading existing SGD from {sgd_path}")
    else:
        time_start = time.time()
        sgd = SGDClassifier(
            loss="hinge",  # Similar to a linear SVM margin objective.
            alpha=1e-4,
            max_iter=50,
            tol=1e-3,
            random_state=42,
            class_weight="balanced",
        )
        sgd.fit(x_comb_no_ts_sub, y_sub)
        joblib.dump(sgd, sgd_path)
        print(f"[Learner] Trained SGD in {time.time() - time_start:.2f} seconds and saved to {sgd_path}")

    models["SGD"] = sgd
    model_meta["SGD"] = {
        "path": str(sgd_path),
        "model_type": "classifier",
        "input_req": "combined_no_ts",
        "kind": "sklearn",
        "t_stream_dim": t_stream_dim,
    }

    if ae_path.exists():
        ae = Autoencoder.load_checkpoint(str(ae_path), device=str(device))
        print(f"[Learner] Loading existing AE from {ae_path}")
    else:
        time_start = time.time()
        ae = train_autoencoder_model(
            x_comb_no_ts_sub,
            x_t_sub,
            y_sub,
            x_comb_no_ts_test,
            x_t_test,
            y_test,
            device=device,
            epochs=epochs[0],
        )
        ae.save_checkpoint(filename=ae_path.name, checkpoint_dir=model_dir)
        print(f"[Learner] Trained AE in {time.time() - time_start:.2f} seconds and saved to {ae_path}")

    benign_mask = (y_sub == 0)
    ae_calib_input = x_comb_no_ts_sub[benign_mask] if np.any(benign_mask) else x_comb_no_ts_sub
    with torch.no_grad():
        ae_train_recon = ae(torch.tensor(ae_calib_input, dtype=torch.float32, device=device)).detach().cpu().numpy()
    ae_train_err = np.mean((ae_train_recon - ae_calib_input) ** 2, axis=1)
    models["AE"] = ae
    model_meta["AE"] = {
        "path": str(ae_path),
        "model_type": "anomaly",
        "input_req": "combined_no_ts",
        "kind": "torch",
        "ae_min": float(np.percentile(ae_train_err, 1)),
        "ae_max": float(np.percentile(ae_train_err, 99)),
        "t_stream_dim": t_stream_dim,
    }

    if lstm_path.exists():
        lstm = LSTMClassifier.load_checkpoint(str(lstm_path), device=str(device))
        print(f"[Learner] Loading existing LSTM from {lstm_path}")
    else:
        time_start = time.time()
        lstm = train_lstm_model(
            x_s_sub,
            x_t_sub,
            y_sub,
            x_s_test,
            x_t_test,
            y_test,
            device=device,
            epochs=epochs[1],
            combined_input=True,
        )
        lstm.save_checkpoint(filename=lstm_path.name, checkpoint_dir=model_dir)
        print(f"[Learner] Trained LSTM in {time.time() - time_start:.2f} seconds and saved to {lstm_path}")

    models["LSTM"] = lstm
    model_meta["LSTM"] = {
        "path": str(lstm_path),
        "model_type": "classifier",
        "input_req": "combined_all",
        "kind": "torch",
    }

    if dsfa_path.exists():
        dsfa = DSFANet.load_checkpoint(str(dsfa_path), device=str(device))
        print(f"[Learner] Loading existing DSFANet from {dsfa_path}")
    else:
        time_start = time.time()
        dsfa = train_dsfanet(x_s_sub, x_t_sub, y_sub, device=device, epochs=epochs[2])
        dsfa.save_checkpoint(filename=dsfa_path.name, checkpoint_dir=model_dir)
        print(f"[Learner] Trained DSFANet in {time.time() - time_start:.2f} seconds and saved to {dsfa_path}")

    models["DSFANet"] = dsfa
    model_meta["DSFANet"] = {"path": str(dsfa_path), "model_type": "classifier", "input_req": "both", "kind": "torch"}

    rows = []
    prob_bank = {}

    for name, model in models.items():
        meta = model_meta[name]
        if name == "AE":
            probs = ae_probs(model, x_comb_no_ts_test, meta["ae_min"], meta["ae_max"], device=device)
        else:
            probs = get_raw_score(
                model,
                meta["model_type"],
                meta["input_req"],
                x_s_test,
                x_t_test,
                device,
                t_stream_dim=meta.get("t_stream_dim"),
            )

        prob_bank[name] = probs
        save_predictions(pred_dir, dataset_key, name, y_test, probs)
        m = metric_row(y_test, probs)
        rows.append({"step": "baseline", "dataset": dataset, "model": name, **m})

    unifier = UnificationLayer()
    voting = VotingEnsemble(unifier=unifier, device=str(device)) if "voting" in ensemble_types else None
    stacking = StackingEnsemble(unifier=unifier, device=str(device)) if "stacking" in ensemble_types else None

    base_configs = [
        {
            "name": "RandomForest",
            "model": models["RandomForest"],
            "model_type": "classifier",
            "input_req": "combined_no_ts",
            "t_stream_dim": t_stream_dim,
        },
        {
            "name": "SGD",
            "model": models["SGD"],
            "model_type": "classifier",
            "input_req": "combined_no_ts",
            "t_stream_dim": t_stream_dim,
        },
        {
            "name": "AE",
            "model": models["AE"],
            "model_type": "anomaly",
            "input_req": "combined_no_ts",
            "t_stream_dim": t_stream_dim,
        },
        {
            "name": "LSTM",
            "model": models["LSTM"],
            "model_type": "classifier",
            "input_req": "combined_all",
        },
        {
            "name": "DSFANet",
            "model": models["DSFANet"],
            "model_type": "classifier",
            "input_req": "both",
        },
    ]
    for cfg in base_configs:
        if voting is not None:
            voting.add_model(
                cfg["name"],
                cfg["model"],
                cfg["model_type"],
                cfg["input_req"],
                t_stream_dim=cfg.get("t_stream_dim"),
            )
        if stacking is not None:
            stacking.add_model(
                cfg["name"],
                cfg["model"],
                cfg["model_type"],
                cfg["input_req"],
                t_stream_dim=cfg.get("t_stream_dim"),
            )

    if voting is not None:
        voting.calibrate(x_s_val, x_t_val)
        voting_probs = voting.predict(x_s_test, x_t_test)
        save_predictions(pred_dir, dataset_key, "Voting", y_test, voting_probs)
        rows.append({"step": "ensemble", "dataset": dataset, "model": "Voting", **metric_row(y_test, voting_probs)})
    
    if stacking is not None:
        stacking.calibrate(x_s_val, x_t_val)
        stacking.fit_meta(x_s_val, x_t_val, y_val)
        stacking_probs = stacking.predict(x_s_test, x_t_test)
        save_predictions(pred_dir, dataset_key, "Stacking", y_test, stacking_probs)
        rows.append({"step": "ensemble", "dataset": dataset, "model": "Stacking", **metric_row(y_test, stacking_probs)})

    ensemble_packages = []

    if stacking is not None:
        stack_pack = {
            "name": "Stacking",
            "model_order": [cfg["name"] for cfg in base_configs],
            "model_info": {k: model_meta[k] for k in [cfg["name"] for cfg in base_configs]},
            "unifier_stats": unifier.stats,
            "meta_learner": stacking.meta_learner,
        }
        stack_path = model_dir / f"{dataset_key}_stacking_pack.joblib"
        joblib.dump(stack_pack, stack_path)
        ensemble_packages.append({"name": "Stacking", "path": str(stack_path)})

    if "xgboost" in ensemble_types:
        try:
            xgb_ens = XGBoostStackingEnsemble(unifier=unifier, device=str(device))
            for cfg in base_configs:
                xgb_ens.add_model(
                    cfg["name"],
                    cfg["model"],
                    cfg["model_type"],
                    cfg["input_req"],
                    t_stream_dim=cfg.get("t_stream_dim"),
                )
            xgb_ens.fit_meta(x_s_val, x_t_val, y_val)
            xgb_probs = xgb_ens.predict(x_s_test, x_t_test)
            save_predictions(pred_dir, dataset_key, "XGBoostStacking", y_test, xgb_probs)
            rows.append({"step": "ensemble", "dataset": dataset, "model": "XGBoostStacking", **metric_row(y_test, xgb_probs)})

            xgb_pack = {
                "name": "XGBoostStacking",
                "model_order": [cfg["name"] for cfg in base_configs],
                "model_info": {k: model_meta[k] for k in [cfg["name"] for cfg in base_configs]},
                "unifier_stats": unifier.stats,
                "meta_learner": xgb_ens.meta_learner,
            }
            xgb_path = model_dir / f"{dataset_key}_xgb_stacking_pack.joblib"
            joblib.dump(xgb_pack, xgb_path)
            ensemble_packages.append({"name": "XGBoostStacking", "path": str(xgb_path)})
        except Exception as ex:
            rows.append({"step": "ensemble", "dataset": dataset, "model": "XGBoostStacking", "acc": np.nan, "f1": np.nan, "precision": np.nan, "recall": np.nan, "ap": np.nan, "error": str(ex)})

    registry = {
        "dataset": dataset,
        "dataset_key": dataset_key,
        "static_features": prep.used_static_cols,
        "temporal_features": prep.used_temporal_all_cols,
        "t_stream_features": prep.used_t_stream_cols,
        "timestamp_features": prep.used_timestamp_cols,
        "combined_features_no_ts": prep.used_static_cols + prep.used_t_stream_cols,
        "combined_features_all": prep.used_static_cols + prep.used_temporal_all_cols,
        "log_scaled_features": prep.log_scale_cols,
        "models": model_meta,
        "ensembles": ensemble_packages,
    }
    registry_path = ds_dir / f"registry_{dataset_key}.json"
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    return rows, registry, (x_s_train, x_t_train, y_train, x_s_test, x_t_test, y_test)


def load_model_from_meta(model_name, meta: dict, device="cpu"):
    """Load a model from registry metadata.

    Returns:
        model: object
    """
    if meta["kind"] == "torch":
        if model_name == "AE":
            return Autoencoder.load_checkpoint(meta["path"], device=str(device))
        if model_name == "LSTM":
            return LSTMClassifier.load_checkpoint(meta["path"], device=str(device))
        if model_name == "DSFANet":
            return DSFANet.load_checkpoint(meta["path"], device=str(device))
        raise ValueError(f"Unsupported torch model: {model_name}")

    return joblib.load(meta["path"])


def predict_from_package(pack: dict, loaded_models: dict, x_s, x_t, device="cpu") -> np.ndarray:
    """Predict with a serialized stacking package.

    Returns:
        probs: np.ndarray
    """
    feats = []
    for name in pack["model_order"]:
        m = loaded_models[name]
        meta = pack["model_info"][name]
        raw = get_raw_score(
            m,
            meta["model_type"],
            meta["input_req"],
            x_s,
            x_t,
            device,
            t_stream_dim=meta.get("t_stream_dim"),
        )
        unified = unify_scores(raw, pack["unifier_stats"][name])
        feats.append(unified)

    x_meta = np.column_stack(feats)
    return pack["meta_learner"].predict_proba(x_meta)[:, 1]


def step1_benchmarks(args, run_dir: Path, device="cpu"):
    """Run step 1 benchmark for selected datasets.

    Returns:
        df: pd.DataFrame
        registries: dict[str, dict]
        dataset_packs: dict[str, tuple]
    """
    all_rows = []
    registries = {}
    dataset_packs = {}

    for ds in args.datasets:
        rows, registry, data_pack = train_and_eval_dataset(
            dataset=ds,
            run_dir=run_dir,
            device=device,
            max_train_samples=args.max_train_samples,
            ensemble_types=args.ensembles,
            epochs=args.epochs,
        )
        all_rows.extend(rows)
        registries[registry["dataset_key"]] = registry
        if ds == args.base_dataset:
            dataset_packs[registry["dataset_key"]] = data_pack

    df = pd.DataFrame(all_rows)
    out_csv = run_dir / f"summary_step1_benchmark_{args.run_id}.csv"
    df.to_csv(out_csv, index=False)

    plt.figure(figsize=(10, 5))
    chart_df = df.dropna(subset=["ap"])
    if not chart_df.empty:
        pivot = chart_df.pivot_table(index="model", columns="dataset", values="ap", aggfunc="mean")
        pivot.plot(kind="bar")
        plt.ylabel("Average Precision")
        plt.title("Step 1 Benchmark AP by Dataset")
        plt.tight_layout()
        # Keep chart compact for single-dataset runs.
        if len(chart_df["dataset"].unique()) == 1:
            plt.legend().set_visible(False)
        plt.savefig(run_dir / f"chart_step1_ap_{args.run_id}.png")
    plt.close("all")

    return df, registries, dataset_packs


def step2_drift(args, run_dir: Path, device="cpu", registry: dict | None = None, base_pack=None):
    """Run drift robustness evaluation on base and ensemble models.

    Returns:
        df: pd.DataFrame
    """
    x_s_train, x_t_train, y_train, x_s_test, x_t_test, y_test = base_pack
    drifter = DriftGenerator()

    def _subset_triplet(x_s, x_t, y, max_samples, seed):
        """Subsample aligned arrays with a fixed random seed."""
        if max_samples <= 0 or len(y) <= max_samples:
            return x_s, x_t, y
        idx = np.random.RandomState(seed).choice(len(y), size=max_samples, replace=False)
        return x_s[idx], x_t[idx], y[idx]

    drift_limit = int(args.drift_subset_size) if int(args.drift_subset_size) > 0 else 0
    natural_limit = int(args.natural_shift_size) if int(args.natural_shift_size) > 0 else drift_limit
    t_stream_dim = len(registry.get("t_stream_features", []))
    if t_stream_dim <= 0:
        t_stream_dim = max(0, x_t_test.shape[1] - 2)

    loaded_models = {}
    for name, meta in registry["models"].items():
        loaded_models[name] = load_model_from_meta(name, meta, device)

    stack_pack_path = next((x["path"] for x in registry["ensembles"] if x["name"] == "Stacking"), None)
    xgb_pack_path = next((x["path"] for x in registry["ensembles"] if x["name"] == "XGBoostStacking"), None)
    stack_pack = joblib.load(stack_pack_path) if stack_pack_path else None
    xgb_pack = joblib.load(xgb_pack_path) if xgb_pack_path else None

    val_n = min(max(200, int(0.2 * len(y_train))), len(y_train) - 1)
    x_s_val, x_t_val = x_s_train[:val_n], x_t_train[:val_n]

    voting = VotingEnsemble(unifier=UnificationLayer(), device=str(device))
    voting_base_order = ["RandomForest", "SGD", "AE", "LSTM", "DSFANet"]
    for model_name in voting_base_order:
        meta = registry["models"][model_name]
        voting.add_model(
            model_name,
            loaded_models[model_name],
            meta["model_type"],
            meta["input_req"],
            t_stream_dim=meta.get("t_stream_dim"),
        )
    voting.calibrate(x_s_val, x_t_val)

    benign_s, benign_t = extract_benign_samples(args.base_dataset, max_samples=args.max_benign_for_attacks)

    clean_case = _subset_triplet(x_s_test, x_t_test, y_test, drift_limit, seed=42)
    label_case = drifter.simulate_label_shift(x_s_test, x_t_test, y_test, target_malicious_ratio=0.8)
    label_case = _subset_triplet(label_case[0], label_case[1], label_case[2], drift_limit, seed=42042)
    x_t_corrupt = x_t_test.copy()
    x_t_corrupt_t = drifter.simulate_corruption(x_t_test[:, :t_stream_dim], noise_type="gaussian", severity=0.1)
    x_t_corrupt[:, :t_stream_dim] = x_t_corrupt_t
    corruption_case = (
        drifter.simulate_corruption(x_s_test, noise_type="gaussian", severity=0.1),
        x_t_corrupt,
        y_test,
    )
    corruption_case = _subset_triplet(corruption_case[0], corruption_case[1], corruption_case[2], drift_limit, seed=103)

    drift_cases = {
        "clean": clean_case,
        "label_shift": label_case,
        "corruption": corruption_case,
    }

    for natural_ds in args.natural_datasets:
        try:
            n_s, n_t, n_y = drifter.load_natural_shift_data(natural_ds, max_samples=natural_limit)
            if n_s.shape[1] == x_s_test.shape[1] and n_t.shape[1] == x_t_test.shape[1]:
                drift_cases[f"natural_{slug(natural_ds)}"] = (n_s, n_t, n_y)
        except Exception:
            continue

    for adv in ["fgsm", "pgd", "mimicry", "gdkde"]:
        sub_n = len(y_test) if drift_limit <= 0 else min(drift_limit, len(y_test))
        idx = np.random.RandomState(42).choice(len(y_test), size=sub_n, replace=False)
        adv_s, adv_t, adv_y = drifter.simulate_adversarial(
            loaded_models["DSFANet"],
            x_s_test[idx],
            x_t_test[idx],
            y_test[idx],
            method=adv,
            epsilon=0.08,
            steps=6,
            alpha=0.02,
            device=str(device),
            benign_x_s=benign_s,
            benign_x_t=benign_t,
        )
        drift_cases[f"adv_{adv}"] = (adv_s, adv_t, adv_y)

    rows = []
    out_pred_dir = ensure_dir(run_dir / "base_drift_predictions")

    for drift_name, (dxs, dxt, dy) in drift_cases.items():
        rf_meta = registry["models"]["RandomForest"]
        sgd_meta = registry["models"]["SGD"]
        ae_meta = registry["models"]["AE"]
        lstm_meta = registry["models"]["LSTM"]
        dsfa_meta = registry["models"]["DSFANet"]

        rf_prob = get_raw_score(
            loaded_models["RandomForest"],
            rf_meta["model_type"],
            rf_meta["input_req"],
            dxs,
            dxt,
            device,
            t_stream_dim=rf_meta.get("t_stream_dim"),
        )
        sgd_prob = get_raw_score(
            loaded_models["SGD"],
            sgd_meta["model_type"],
            sgd_meta["input_req"],
            dxs,
            dxt,
            device,
            t_stream_dim=sgd_meta.get("t_stream_dim"),
        )
        ae_raw = get_raw_score(
            loaded_models["AE"],
            ae_meta["model_type"],
            ae_meta["input_req"],
            dxs,
            dxt,
            device,
            t_stream_dim=ae_meta.get("t_stream_dim"),
        )
        ae_denom = max(ae_meta["ae_max"] - ae_meta["ae_min"], 1e-8)
        ae_prob = np.clip((ae_raw - ae_meta["ae_min"]) / ae_denom, 0.0, 1.0)
        lstm_prob = get_raw_score(
            loaded_models["LSTM"],
            lstm_meta["model_type"],
            lstm_meta["input_req"],
            dxs,
            dxt,
            device,
            t_stream_dim=lstm_meta.get("t_stream_dim"),
        )
        dsfa_prob = get_raw_score(
            loaded_models["DSFANet"],
            dsfa_meta["model_type"],
            dsfa_meta["input_req"],
            dxs,
            dxt,
            device,
            t_stream_dim=dsfa_meta.get("t_stream_dim"),
        )

        voting_prob = voting.predict(dxs, dxt)

        for mname, probs in [
            ("RandomForest", rf_prob),
            ("SGD", sgd_prob),
            ("AE", ae_prob),
            ("LSTM", lstm_prob),
            ("DSFANet", dsfa_prob),
            ("Voting", voting_prob),
        ]:
            save_predictions(out_pred_dir, slug(args.base_dataset), f"{mname}_{drift_name}", dy, probs)
            rows.append({"step": "drift", "dataset": args.base_dataset, "drift": drift_name, "model": mname, **metric_row(dy, probs)})

        if stack_pack is not None:
            stack_prob = predict_from_package(stack_pack, loaded_models, dxs, dxt, device)
            rows.append({"step": "drift", "dataset": args.base_dataset, "drift": drift_name, "model": "Stacking", **metric_row(dy, stack_prob)})
            save_predictions(out_pred_dir, slug(args.base_dataset), f"Stacking_{drift_name}", dy, stack_prob)

        if xgb_pack is not None:
            xgb_prob = predict_from_package(xgb_pack, loaded_models, dxs, dxt, device)
            rows.append({"step": "drift", "dataset": args.base_dataset, "drift": drift_name, "model": "XGBoostStacking", **metric_row(dy, xgb_prob)})
            save_predictions(out_pred_dir, slug(args.base_dataset), f"XGBoostStacking_{drift_name}", dy, xgb_prob)

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / f"summary_step2_drift_{args.run_id}.csv", index=False)
    return df


def _build_ensemble_bank(
    model_bank: dict,
    registry_models: dict,
    x_s_val,
    x_t_val,
    y_val,
    device="cpu",
    include_xgb=False,
):
    """Construct calibrated ensemble objects from a model bank.

    Returns:
        voting: VotingEnsemble
        stacking: StackingEnsemble
        xgb: XGBoostStackingEnsemble | None
    """
    unifier = UnificationLayer()
    voting = VotingEnsemble(unifier=unifier, device=str(device))
    stacking = StackingEnsemble(unifier=unifier, device=str(device))

    base_order = ["RandomForest", "SGD", "AE", "LSTM", "DSFANet"]
    for model_name in base_order:
        meta = registry_models[model_name]
        voting.add_model(
            model_name,
            model_bank[model_name],
            meta["model_type"],
            meta["input_req"],
            t_stream_dim=meta.get("t_stream_dim"),
        )
        stacking.add_model(
            model_name,
            model_bank[model_name],
            meta["model_type"],
            meta["input_req"],
            t_stream_dim=meta.get("t_stream_dim"),
        )

    voting.calibrate(x_s_val, x_t_val)
    stacking.calibrate(x_s_val, x_t_val)
    stacking.fit_meta(x_s_val, x_t_val, y_val)

    xgb = None
    if include_xgb:
        try:
            xgb = XGBoostStackingEnsemble(unifier=unifier, device=str(device))
            for model_name in base_order:
                meta = registry_models[model_name]
                xgb.add_model(
                    model_name,
                    model_bank[model_name],
                    meta["model_type"],
                    meta["input_req"],
                    t_stream_dim=meta.get("t_stream_dim"),
                )
            xgb.calibrate(x_s_val, x_t_val)
            xgb.fit_meta(x_s_val, x_t_val, y_val)
        except Exception:
            xgb = None

    return voting, stacking, xgb


def _load_ood_case(dataset: str, max_samples: int, seed: int = 8080):
    """Load one OOD evaluation case from a dataset test split."""
    prep = DataPreprocessor(dataset)
    (_, _, _), (x_s_test, x_t_test, y_test) = prep.prepare_data()
    if max_samples > 0 and len(y_test) > max_samples:
        idx = np.random.RandomState(seed).choice(len(y_test), size=max_samples, replace=False)
        return x_s_test[idx], x_t_test[idx], y_test[idx]
    return x_s_test, x_t_test, y_test


def _best_key_by_model(best_models: dict, model_name: str) -> str | None:
    """Choose best saved checkpoint for a model across all drift cases."""
    candidates = [k for k in best_models if k.startswith(f"{model_name}_")]
    if not candidates:
        return None
    return max(candidates, key=lambda k: float(best_models[k].get("acc_gain", -1e9)))


def step3_retrain(args, run_dir: Path, device="cpu", registry: dict | None = None, base_pack=None, advs=None):
    """Run adaptive retraining on 4 base attacks plus 1 OOD dataset case.
    
    Args:
        advs: List of adversarial attack types to run. If None, defaults to ["fgsm", "pgd", "mimicry", "gdkde"].

    Returns:
        df: pd.DataFrame
        best_models: dict[str, dict[str, object]]
        transfer_best_models: dict[str, dict[str, object]]
    """
    x_s_train, x_t_train, y_train, x_s_test, x_t_test, y_test = base_pack

    advs = advs or ["fgsm", "pgd", "mimicry", "gdkde"]

    models = {
        "AE": load_model_from_meta("AE", registry["models"]["AE"], device),
        "LSTM": load_model_from_meta("LSTM", registry["models"]["LSTM"], device),
        "DSFANet": load_model_from_meta("DSFANet", registry["models"]["DSFANet"], device),
    }

    drifter = DriftGenerator()
    benign_s, benign_t = extract_benign_samples(args.base_dataset, max_samples=args.max_benign_for_attacks)

    subset_n = min(args.drift_subset_size, len(y_test))
    idx = np.random.RandomState(123).choice(len(y_test), size=subset_n, replace=False)
    base_s, base_t, base_y = x_s_test[idx], x_t_test[idx], y_test[idx]

    drift_cases: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for adv in advs:
        adv_s, adv_t, adv_y = drifter.simulate_adversarial(
            models["DSFANet"],
            base_s,
            base_t,
            base_y,
            method=adv,
            epsilon=0.08,
            steps=8,
            alpha=0.02,
            device=str(device),
            benign_x_s=benign_s,
            benign_x_t=benign_t,
        )
        drift_cases[f"adv_{adv}"] = (adv_s, adv_t, adv_y)

    ood_case = _load_ood_case(args.ood_dataset, args.drift_subset_size, seed=4242)
    drift_cases[f"ood_{slug(args.ood_dataset)}"] = ood_case

    retrain_dir = ensure_dir(run_dir / "retrain_models")
    transfer_retrain_dir = ensure_dir(run_dir / "transfer_retrain_models")
    rows = []
    best_models = {}
    transfer_best_models = {}

    metrics_list = parse_str_list(args.retrain_metrics)
    budgets = parse_float_list(args.retrain_budgets)
    id_ratios = parse_float_list(args.retrain_id_ratios)

    for model_name, model in models.items():
        for case_name, (dxs, dxt, dy) in drift_cases.items():
            meta = registry["models"][model_name]
            before_prob = get_raw_score(
                model,
                meta["model_type"],
                meta["input_req"],
                dxs,
                dxt,
                device,
                t_stream_dim=meta.get("t_stream_dim"),
            )
            if model_name == "AE":
                denom = max(meta["ae_max"] - meta["ae_min"], 1e-8)
                before_prob = np.clip((before_prob - meta["ae_min"]) / denom, 0.0, 1.0)
            before_metrics = metric_row(dy, before_prob)

            best_gain = -1e9
            best_state = None
            best_tag = ""

            for metric in metrics_list:
                for budget in budgets:
                    for id_ratio in id_ratios:
                        # print(f"[Retrain] Model: {model_name}, Case: {case_name}, Metric: {metric}, Budget: {budget}, ID Ratio: {id_ratio}")
                        retrained, after_prob = retrain_model_generic(
                            model_name,
                            model,
                            x_s_train,
                            x_t_train,
                            y_train,
                            dxs,
                            dxt,
                            dy,
                            metric,
                            budget,
                            id_ratio,
                            get_probs_and_features=get_model_probs_and_features,
                            get_model_input=get_model_input,
                            device=device,
                            t_stream_dim=meta.get("t_stream_dim"),
                        )
                        after_metrics = metric_row(dy, after_prob)
                        gain = float(after_metrics["acc"] - before_metrics["acc"])
                        rows.append(
                            {
                                "step": "retrain",
                                "dataset": args.base_dataset,
                                "drift_case": case_name,
                                "ood_dataset": args.ood_dataset,
                                "model": model_name,
                                "selection_metric": metric,
                                "budget_ratio": budget,
                                "id_ratio": id_ratio,
                                "before_acc": float(before_metrics["acc"]),
                                "after_acc": float(after_metrics["acc"]),
                                "acc_gain": gain,
                                "before_ap": float(before_metrics["ap"]),
                                "after_ap": float(after_metrics["ap"]),
                                "ap_gain": float(after_metrics["ap"] - before_metrics["ap"]),
                            }
                        )

                        if gain > best_gain: # Acc gain
                            best_gain = gain
                            best_tag = f"{model_name}_{case_name}_{metric}_b{budget:.2f}_id{id_ratio:.2f}"
                            if isinstance(retrained, nn.Module):
                                best_state = deepcopy(retrained.state_dict())
                            else:
                                best_state = deepcopy(retrained)

            if best_state is not None:
                is_transfer_case = str(case_name).startswith("ood_")
                save_dir = transfer_retrain_dir if is_transfer_case else retrain_dir
                if isinstance(model, nn.Module):
                    model_copy = deepcopy(model)
                    model_copy.load_state_dict(best_state)
                    filename = f"best_{slug(best_tag)}_{args.run_id}.pt"
                    saved = model_copy.save_checkpoint(filename=filename, checkpoint_dir=save_dir)
                else:
                    filename = save_dir / f"best_{slug(best_tag)}_{args.run_id}.joblib"
                    joblib.dump(best_state, filename)
                    saved = str(filename)
                record = {
                    "path": saved,
                    "acc_gain": best_gain,
                    "drift_case": case_name,
                    "model": model_name,
                }
                if is_transfer_case:
                    transfer_best_models[f"{model_name}_{case_name}"] = record
                else:
                    best_models[f"{model_name}_{case_name}"] = record

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / f"summary_step3_retrain_{args.run_id}.csv", index=False)
    (run_dir / f"best_models_step3_{args.run_id}.json").write_text(json.dumps(best_models, indent=2), encoding="utf-8")
    (run_dir / f"transfer_best_models_step3_{args.run_id}.json").write_text(json.dumps(transfer_best_models, indent=2), encoding="utf-8")
    return df, best_models, transfer_best_models


def step4_best_ensemble_shap(args, run_dir: Path, device="cpu", base_pack=None, best_models: dict | None = None, registry: dict | None = None):
    """Re-evaluate ensemble on base dataset and export SHAP artifacts.

    Returns:
        df: pd.DataFrame
    """
    if best_models is None:
        best_models = {}
    if registry is None:
        registry = {}
    x_s_train, x_t_train, y_train, x_s_test, x_t_test, y_test = base_pack
    val_n = min(max(200, int(0.2 * len(y_train))), len(y_train) - 1)
    x_s_val, x_t_val, y_val = x_s_train[:val_n], x_t_train[:val_n], y_train[:val_n]

    best_ae_key = _best_key_by_model(best_models, "AE")
    best_lstm_key = _best_key_by_model(best_models, "LSTM")
    best_dsfa_key = _best_key_by_model(best_models, "DSFANet")

    ae_model = Autoencoder.load_checkpoint(best_models[best_ae_key]["path"], device=str(device)) if best_ae_key else load_model_from_meta("AE", registry["models"]["AE"], device)
    lstm_model = LSTMClassifier.load_checkpoint(best_models[best_lstm_key]["path"], device=str(device)) if best_lstm_key else load_model_from_meta("LSTM", registry["models"]["LSTM"], device)
    dsfa_model = DSFANet.load_checkpoint(best_models[best_dsfa_key]["path"], device=str(device)) if best_dsfa_key else load_model_from_meta("DSFANet", registry["models"]["DSFANet"], device)

    before_bank = {
        "RandomForest": load_model_from_meta("RandomForest", registry["models"]["RandomForest"], device),
        "SGD": load_model_from_meta("SGD", registry["models"]["SGD"], device),
        "AE": load_model_from_meta("AE", registry["models"]["AE"], device),
        "LSTM": load_model_from_meta("LSTM", registry["models"]["LSTM"], device),
        "DSFANet": load_model_from_meta("DSFANet", registry["models"]["DSFANet"], device),
    }
    after_bank = deepcopy(before_bank)
    after_bank["AE"] = ae_model
    after_bank["LSTM"] = lstm_model
    after_bank["DSFANet"] = dsfa_model

    include_xgb = "xgboost" in args.ensembles
    before_voting, before_stacking, before_xgb = _build_ensemble_bank(
        before_bank,
        registry["models"],
        x_s_val,
        x_t_val,
        y_val,
        device,
        include_xgb=include_xgb,
    )
    after_voting, after_stacking, after_xgb = _build_ensemble_bank(
        after_bank,
        registry["models"],
        x_s_val,
        x_t_val,
        y_val,
        device,
        include_xgb=include_xgb,
    )

    eval_cases = {
        f"base_{slug(args.base_dataset)}": (x_s_test, x_t_test, y_test),
    }

    rows = []
    for case_name, (dxs, dxt, dy) in eval_cases.items():
        before_rows = [
            ("Voting", before_voting.predict(dxs, dxt)),
            ("Stacking", before_stacking.predict(dxs, dxt)),
        ]
        if before_xgb is not None:
            before_rows.append(("XGBoostStacking", before_xgb.predict(dxs, dxt)))

        for ens_name, probs in before_rows:
            rows.append(
                {
                    "step": "best_ensemble",
                    "dataset": args.base_dataset,
                    "eval_case": case_name,
                    "model": ens_name,
                    "phase": "before_retrain",
                    **metric_row(dy, probs),
                }
            )

        after_rows = [
            ("Voting", after_voting.predict(dxs, dxt)),
            ("Stacking", after_stacking.predict(dxs, dxt)),
        ]
        if after_xgb is not None:
            after_rows.append(("XGBoostStacking", after_xgb.predict(dxs, dxt)))

        for ens_name, probs in after_rows:
            rows.append(
                {
                    "step": "best_ensemble",
                    "dataset": args.base_dataset,
                    "eval_case": case_name,
                    "model": ens_name,
                    "phase": "after_retrain",
                    **metric_row(dy, probs),
                }
            )

    shap_dir = ensure_dir(run_dir / "shap_best_models")
    prep = DataPreprocessor(args.base_dataset)
    prep.prepare_data()
    s_features = prep.used_static_cols
    t_features = prep.used_t_stream_cols
    ts_features = prep.used_timestamp_cols
    temporal_all_features = t_features + ts_features
    t_stream_dim = len(t_features)
    combined_all_features = s_features + temporal_all_features
    combined_no_ts_features = s_features + t_features

    x_comb_all_test = combine_static_temporal(x_s_test, x_t_test)
    x_comb_no_ts_test = combine_static_temporal(x_s_test, x_t_test, t_stream_dim=t_stream_dim)

    shap_lstm = analyze_lstm_shap(lstm_model, x_comb_all_test, combined_all_features, out_dir=shap_dir)
    shap_ae = analyze_ae_shap(ae_model, x_comb_no_ts_test, combined_no_ts_features, out_dir=shap_dir)
    shap_ds = analyze_dsfanet_shap(dsfa_model, x_s_test, x_t_test, s_features, temporal_all_features, out_dir=shap_dir)

    shap_visuals = {
        "LSTM": generate_shap_visuals(shap_lstm, shap_dir, "lstm"),
        "AE": generate_shap_visuals(shap_ae, shap_dir, "ae"),
        "DSFANet": generate_shap_visuals(shap_ds, shap_dir, "dsfanet"),
    }

    (shap_dir / f"shap_best_summary_{args.run_id}.json").write_text(
        json.dumps({"LSTM": shap_lstm, "AE": shap_ae, "DSFANet": shap_ds, "visualizations": shap_visuals}, indent=2),
        encoding="utf-8",
    )

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / f"summary_step4_best_ensemble_shap_{args.run_id}.csv", index=False)
    return df


def step7_transfer_ensemble_compare(
    args,
    run_dir: Path,
    device="cpu",
    registry: dict | None = None,
    base_pack=None,
    best_models: dict | None = None,
    transfer_best_models: dict | None = None,
):
    """Evaluate new ensemble on OOD dataset and compare retrain vs transfer-retrain models.

    Returns:
        df: pd.DataFrame
    """
    if best_models is None:
        best_models = {}
    if transfer_best_models is None:
        transfer_best_models = {}
    x_s_train, x_t_train, y_train, _x_s_test, _x_t_test, _y_test = base_pack
    val_n = min(max(200, int(0.2 * len(y_train))), len(y_train) - 1)
    x_s_val, x_t_val, y_val = x_s_train[:val_n], x_t_train[:val_n], y_train[:val_n]

    ood_s, ood_t, ood_y = _load_ood_case(args.ood_dataset, args.drift_subset_size, seed=7070)

    base_bank = {
        "RandomForest": load_model_from_meta("RandomForest", registry["models"]["RandomForest"], device),
        "SGD": load_model_from_meta("SGD", registry["models"]["SGD"], device),
        "AE": load_model_from_meta("AE", registry["models"]["AE"], device),
        "LSTM": load_model_from_meta("LSTM", registry["models"]["LSTM"], device),
        "DSFANet": load_model_from_meta("DSFANet", registry["models"]["DSFANet"], device),
    }

    retrain_ae_key = _best_key_by_model(best_models, "AE")
    retrain_lstm_key = _best_key_by_model(best_models, "LSTM")
    retrain_dsfa_key = _best_key_by_model(best_models, "DSFANet")

    transfer_ae_key = _best_key_by_model(transfer_best_models, "AE")
    transfer_lstm_key = _best_key_by_model(transfer_best_models, "LSTM")
    transfer_dsfa_key = _best_key_by_model(transfer_best_models, "DSFANet")

    retrain_bank = deepcopy(base_bank)
    if retrain_ae_key:
        retrain_bank["AE"] = Autoencoder.load_checkpoint(best_models[retrain_ae_key]["path"], device=str(device))
    if retrain_lstm_key:
        retrain_bank["LSTM"] = LSTMClassifier.load_checkpoint(best_models[retrain_lstm_key]["path"], device=str(device))
    if retrain_dsfa_key:
        retrain_bank["DSFANet"] = DSFANet.load_checkpoint(best_models[retrain_dsfa_key]["path"], device=str(device))

    transfer_bank = deepcopy(base_bank)
    if transfer_ae_key:
        transfer_bank["AE"] = Autoencoder.load_checkpoint(transfer_best_models[transfer_ae_key]["path"], device=str(device))
    if transfer_lstm_key:
        transfer_bank["LSTM"] = LSTMClassifier.load_checkpoint(transfer_best_models[transfer_lstm_key]["path"], device=str(device))
    if transfer_dsfa_key:
        transfer_bank["DSFANet"] = DSFANet.load_checkpoint(transfer_best_models[transfer_dsfa_key]["path"], device=str(device))

    include_xgb = "xgboost" in args.ensembles
    retrain_voting, retrain_stacking, retrain_xgb = _build_ensemble_bank(
        retrain_bank,
        registry["models"],
        x_s_val,
        x_t_val,
        y_val,
        device,
        include_xgb=include_xgb,
    )
    transfer_voting, transfer_stacking, transfer_xgb = _build_ensemble_bank(
        transfer_bank,
        registry["models"],
        x_s_val,
        x_t_val,
        y_val,
        device,
        include_xgb=include_xgb,
    )

    rows = []
    retrain_rows = [
        ("Voting", retrain_voting.predict(ood_s, ood_t)),
        ("Stacking", retrain_stacking.predict(ood_s, ood_t)),
    ]
    if retrain_xgb is not None:
        retrain_rows.append(("XGBoostStacking", retrain_xgb.predict(ood_s, ood_t)))

    for ens_name, probs in retrain_rows:
        rows.append(
            {
                "step": "transfer_ensemble_compare",
                "dataset": args.base_dataset,
                "ood_dataset": args.ood_dataset,
                "model": ens_name,
                "phase": "retrain_models",
                **metric_row(ood_y, probs),
            }
        )

    transfer_rows = [
        ("Voting", transfer_voting.predict(ood_s, ood_t)),
        ("Stacking", transfer_stacking.predict(ood_s, ood_t)),
    ]
    if transfer_xgb is not None:
        transfer_rows.append(("XGBoostStacking", transfer_xgb.predict(ood_s, ood_t)))

    for ens_name, probs in transfer_rows:
        rows.append(
            {
                "step": "transfer_ensemble_compare",
                "dataset": args.base_dataset,
                "ood_dataset": args.ood_dataset,
                "model": ens_name,
                "phase": "transfer_retrain_models",
                **metric_row(ood_y, probs),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / f"summary_step7_transfer_ensemble_compare_{args.run_id}.csv", index=False)
    return df


def generate_shap_visuals(shap_report: dict, out_dir: Path, tag: str) -> dict[str, str]:
    """Generate static SHAP plots from exported report files.

    Returns:
        out: dict[str, str]
    """
    out = {}
    importance_csv = shap_report.get("importance_csv")
    samples_json = shap_report.get("samples_json")

    if importance_csv and Path(importance_csv).exists():
        imp_df = pd.read_csv(importance_csv).sort_values("mean_abs_shap", ascending=False).head(20)
        if not imp_df.empty:
            plt.figure(figsize=(9, 6))
            plot_df = imp_df.iloc[::-1]
            plt.barh(plot_df["feature"], plot_df["mean_abs_shap"])
            plt.xlabel("Mean |SHAP|")
            plt.ylabel("Feature")
            plt.title(f"{tag.upper()} Feature Importance")
            plt.tight_layout()
            bar_path = out_dir / f"{tag}_feature_importance.png"
            plt.savefig(bar_path)
            plt.close("all")
            out["feature_importance"] = str(bar_path)

    if samples_json and Path(samples_json).exists():
        records = json.loads(Path(samples_json).read_text(encoding="utf-8"))
        if isinstance(records, list) and records:
            sdf = pd.DataFrame(records)
            shap_cols = [c for c in sdf.columns if c.startswith("shap::")]
            if shap_cols:
                ranked_cols = sorted(shap_cols, key=lambda c: float(np.nanmean(np.abs(sdf[c].values))), reverse=True)[:8]
                plot_values = [sdf[c].values.astype(float) for c in ranked_cols]
                labels = [c.replace("shap::", "") for c in ranked_cols]
                plt.figure(figsize=(9, 6))
                plt.boxplot(plot_values, vert=False, tick_labels=labels, showfliers=False)
                plt.xlabel("SHAP value")
                plt.ylabel("Feature")
                plt.title(f"{tag.upper()} SHAP Distribution")
                plt.tight_layout()
                box_path = out_dir / f"{tag}_shap_distribution.png"
                plt.savefig(box_path)
                plt.close("all")
                out["shap_distribution"] = str(box_path)

    return out


class DSFANetAblation(nn.Module):
    def __init__(self, static_dim: int, temporal_dim: int, n_classes: int, mode: str = "full"):
        """Create a compact DSFANet ablation variant for step 5."""
        super().__init__()
        self.mode = mode
        self.static_fc = nn.Sequential(nn.Linear(static_dim, 64), nn.ReLU())
        self.temporal_conv = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.temporal_lstm = nn.LSTM(16, 32, batch_first=True, bidirectional=True)
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        if mode == "s_only":
            self.head = nn.Linear(64, n_classes)
        elif mode == "t_only":
            self.head = nn.Linear(64, n_classes)
        elif mode == "no_attn":
            self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_classes))
        else:
            self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_classes))

    def forward(self, x_s, x_t):
        """Forward pass for the selected ablation mode."""
        hs = self.static_fc(x_s)
        ht = self.temporal_conv(x_t.unsqueeze(1)).permute(0, 2, 1)
        ht, _ = self.temporal_lstm(ht)
        ht = ht[:, -1, :]

        if self.mode == "s_only":
            return self.head(hs)
        if self.mode == "t_only":
            return self.head(ht)

        fusion = torch.cat([hs, ht], dim=1)
        if self.mode == "no_attn":
            return self.head(fusion)

        attn_out, _ = self.attn(fusion.unsqueeze(1), fusion.unsqueeze(1), fusion.unsqueeze(1))
        return self.head(attn_out.squeeze(1))


def step5_dsfanet_ablation(args, run_dir: Path, device="cpu"):
    """Run DSFANet component ablation on the base dataset.

    Returns:
        df: pd.DataFrame
    """
    prep = DataPreprocessor(args.base_dataset)
    (x_s_train, x_t_train, y_train), (x_s_test, x_t_test, y_test) = prep.prepare_data()

    val_n = min(max(200, int(0.2 * len(y_train))), len(y_train) - 1)
    x_s_sub, x_t_sub, y_sub = x_s_train[val_n:], x_t_train[val_n:], y_train[val_n:]

    if args.step5_train_max_samples > 0 and len(y_sub) > args.step5_train_max_samples:
        train_idx = stratified_sample_indices(y_sub, args.step5_train_max_samples, seed=1337)
        x_s_sub, x_t_sub, y_sub = x_s_sub[train_idx], x_t_sub[train_idx], y_sub[train_idx]

    if args.step5_eval_max_samples > 0 and len(y_test) > args.step5_eval_max_samples:
        eval_idx = stratified_sample_indices(y_test, args.step5_eval_max_samples, seed=2026)
        x_s_test, x_t_test, y_test = x_s_test[eval_idx], x_t_test[eval_idx], y_test[eval_idx]

    rows = []
    modes = ["full", "s_only", "t_only", "no_attn"]

    for mode in modes:
        model = DSFANetAblation(x_s_sub.shape[1], x_t_sub.shape[1], config.NUM_CLASSES, mode=mode).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        bs = 128
        train_loader, _ = get_dataloaders(
            (x_s_sub, x_t_sub, y_sub),
            (x_s_sub[:1], x_t_sub[:1], y_sub[:1]),
            batch_size=bs,
        )
        for _ in range(3):
            for xs, xt, yy in train_loader:
                xs = xs.to(device, non_blocking=True)
                xt = xt.to(device, non_blocking=True)
                yy = yy.to(device, non_blocking=True)
                optimizer.zero_grad()
                logits = model(xs, xt)
                loss = criterion(logits, yy)
                loss.backward()
                optimizer.step()

        probs = torch_probs(model, x_s_test, x_t_test, "both", device)

        rows.append({"step": "ablation", "dataset": args.base_dataset, "model": f"DSFANet_{mode}", **metric_row(y_test, probs)})

    df = pd.DataFrame(rows)
    out_csv = run_dir / f"summary_step5_dsfanet_ablation_{args.run_id}.csv"
    df.to_csv(out_csv, index=False)

    plt.figure(figsize=(8, 4))
    plt.bar(df["model"], df["acc"])
    plt.xticks(rotation=20)
    plt.ylabel("Accuracy")
    plt.title("DSFANet Ablation Accuracy")
    plt.tight_layout()
    plt.savefig(run_dir / f"chart_step5_dsfanet_ablation_{args.run_id}.png")
    plt.close("all")

    return df


def step6_ensemble_ablation(args, run_dir: Path, device="cpu", registry: dict | None = None, base_pack=None, best_models: dict | None = None):
    """Run ensemble member-subset ablation for voting and stacking.

    Returns:
        df: pd.DataFrame
    """
    if best_models is None:
        best_models = {}
    x_s_train, x_t_train, y_train, x_s_test, x_t_test, y_test = base_pack
    val_n = min(max(200, int(0.2 * len(y_train))), len(y_train) - 1)
    if args.step6_val_max_samples > 0:
        val_n = min(val_n, args.step6_val_max_samples)
    val_idx = stratified_sample_indices(y_train, val_n, seed=4040)
    x_s_val, x_t_val, y_val = x_s_train[val_idx], x_t_train[val_idx], y_train[val_idx]

    eval_n = min(len(y_test), max(1000, args.drift_subset_size))
    if args.step6_eval_max_samples > 0:
        eval_n = min(eval_n, args.step6_eval_max_samples)
    if eval_n < len(y_test):
        eval_idx = stratified_sample_indices(y_test, eval_n, seed=2020)
        x_s_eval, x_t_eval, y_eval = x_s_test[eval_idx], x_t_test[eval_idx], y_test[eval_idx]
    else:
        x_s_eval, x_t_eval, y_eval = x_s_test, x_t_test, y_test

    model_bank = {
        "RandomForest": load_model_from_meta("RandomForest", registry["models"]["RandomForest"], device),
        "SGD": load_model_from_meta("SGD", registry["models"]["SGD"], device),
        "AE": load_model_from_meta("AE", registry["models"]["AE"], device),
        "LSTM": load_model_from_meta("LSTM", registry["models"]["LSTM"], device),
        "DSFANet": load_model_from_meta("DSFANet", registry["models"]["DSFANet"], device),
    }

    best_ae_key = next((k for k in best_models if k.startswith("AE_")), None)
    best_lstm_key = next((k for k in best_models if k.startswith("LSTM_")), None)
    best_dsfa_key = next((k for k in best_models if k.startswith("DSFANet_")), None)

    if best_ae_key:
        model_bank["AE"] = Autoencoder.load_checkpoint(best_models[best_ae_key]["path"], device=str(device))
    if best_lstm_key:
        model_bank["LSTM"] = LSTMClassifier.load_checkpoint(best_models[best_lstm_key]["path"], device=str(device))
    if best_dsfa_key:
        model_bank["DSFANet"] = DSFANet.load_checkpoint(best_models[best_dsfa_key]["path"], device=str(device))

    base_order = ["RandomForest", "SGD", "AE", "LSTM", "DSFANet"]
    subsets = [
        ("all", base_order),
        ("drop_rf", [x for x in base_order if x != "RandomForest"]),
        ("drop_sgd", [x for x in base_order if x != "SGD"]),
        ("drop_ae", [x for x in base_order if x != "AE"]),
        ("drop_lstm", [x for x in base_order if x != "LSTM"]),
        ("drop_dsfanet", [x for x in base_order if x != "DSFANet"]),
        ("traditional_only", ["RandomForest", "SGD"]),
        ("neural_only", ["AE", "LSTM", "DSFANet"]),
    ]

    rows = []
    for subset_name, members in subsets:
        unifier = UnificationLayer()
        voting = VotingEnsemble(unifier=unifier, device=str(device))
        stacking = StackingEnsemble(unifier=unifier, device=str(device))

        for model_name in members:
            meta = registry["models"][model_name]
            voting.add_model(
                model_name,
                model_bank[model_name],
                meta["model_type"],
                meta["input_req"],
                t_stream_dim=meta.get("t_stream_dim"),
            )
            stacking.add_model(
                model_name,
                model_bank[model_name],
                meta["model_type"],
                meta["input_req"],
                t_stream_dim=meta.get("t_stream_dim"),
            )

        voting.calibrate(x_s_val, x_t_val)
        voting_prob = voting.predict(x_s_eval, x_t_eval)
        rows.append(
            {
                "step": "ensemble_ablation",
                "dataset": args.base_dataset,
                "ensemble": "Voting",
                "subset": subset_name,
                "members": ",".join(members),
                "eval_size": int(len(y_eval)),
                **metric_row(y_eval, voting_prob),
            }
        )

        try:
            stacking.fit_meta(x_s_val, x_t_val, y_val)
            stack_prob = stacking.predict(x_s_eval, x_t_eval)
            rows.append(
                {
                    "step": "ensemble_ablation",
                    "dataset": args.base_dataset,
                    "ensemble": "Stacking",
                    "subset": subset_name,
                    "members": ",".join(members),
                    "eval_size": int(len(y_eval)),
                    **metric_row(y_eval, stack_prob),
                }
            )
        except Exception as ex:
            rows.append(
                {
                    "step": "ensemble_ablation",
                    "dataset": args.base_dataset,
                    "ensemble": "Stacking",
                    "subset": subset_name,
                    "members": ",".join(members),
                    "acc": np.nan,
                    "f1": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "ap": np.nan,
                    "error": str(ex),
                }
            )

    df = pd.DataFrame(rows)
    out_csv = run_dir / f"summary_step6_ensemble_ablation_{args.run_id}.csv"
    df.to_csv(out_csv, index=False)

    try:
        plt.figure(figsize=(11, 5))
        chart_df = df.dropna(subset=["ap"]).copy()
        if not chart_df.empty:
            chart_df["label"] = chart_df["ensemble"] + "::" + chart_df["subset"]
            chart_df = chart_df.sort_values("ap", ascending=False)
            plt.bar(chart_df["label"], chart_df["ap"])
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Average Precision")
            plt.title("Step 6 Ensemble Component Ablation (AP)")
            plt.tight_layout()
            plt.savefig(run_dir / f"chart_step6_ensemble_ablation_{args.run_id}.png")
        plt.close("all")
    except Exception:
        plt.close("all")

    return df


def step8_export_for_web(run_dir: Path, args):
    """Export step summaries into web dashboard JSON payloads.
    """
    summary_files = sorted(run_dir.glob("summary_step*.csv"))
    payload = {
        "run_id": args.run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "summary_files": [str(p) for p in summary_files],
        "base_dataset": args.base_dataset,
        "ood_dataset": getattr(args, "ood_dataset", "NF-BoT-IoT-v3.csv"),
        "steps": parse_str_list(str(args.steps)),
    }

    for csv_path in summary_files:
        key = csv_path.stem
        try:
            payload[key] = pd.read_csv(csv_path).head(500).to_dict(orient="records")
        except Exception:
            payload[key] = []

    step3_key = f"summary_step3_retrain_{args.run_id}"
    step4_key = f"summary_step4_best_ensemble_shap_{args.run_id}"

    step3_rows = payload.get(step3_key, [])
    if step3_rows:
        try:
            step3_df = pd.DataFrame(step3_rows)
            cols = ["model", "drift_case", "acc_gain", "ap_gain", "selection_metric", "budget_ratio", "id_ratio"]
            use_cols = [c for c in cols if c in step3_df.columns]
            if use_cols and "acc_gain" in step3_df.columns:
                idx = step3_df.groupby([c for c in ["model", "drift_case"] if c in step3_df.columns])["acc_gain"].idxmax()
                payload["summary_step3_best_by_case"] = step3_df.loc[idx, use_cols].to_dict(orient="records")
        except Exception:
            payload["summary_step3_best_by_case"] = []

    step4_rows = payload.get(step4_key, [])
    if step4_rows:
        try:
            step4_df = pd.DataFrame(step4_rows)
            if {"model", "eval_case", "phase", "acc"}.issubset(set(step4_df.columns)):
                pivot = step4_df.pivot_table(index=["model", "eval_case"], columns="phase", values="acc", aggfunc="mean").reset_index()
                if "after_retrain" in pivot.columns and "before_retrain" in pivot.columns:
                    pivot["acc_delta"] = pivot["after_retrain"] - pivot["before_retrain"]
                payload["summary_step4_phase_compare"] = pivot.fillna(np.nan).to_dict(orient="records")
        except Exception:
            payload["summary_step4_phase_compare"] = []

    out_www = ensure_dir(Path("out") / "www")
    run_json = out_www / f"experiments_{args.run_id}.json"
    run_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    index_path = out_www / "experiments_index.json"
    existing_runs = []
    if index_path.exists():
        try:
            old_index = json.loads(index_path.read_text(encoding="utf-8"))
            existing_runs = old_index.get("runs", []) if isinstance(old_index, dict) else []
        except Exception:
            existing_runs = []

    existing_map = {item.get("run_id"): item for item in existing_runs if isinstance(item, dict) and item.get("run_id")}
    existing_map[args.run_id] = {
        "run_id": args.run_id,
        "generated_at": payload["generated_at"],
        "base_dataset": args.base_dataset,
        "path": str(run_json),
    }

    dataset_latest: dict[str, dict] = {}
    for item in existing_map.values():
        if not isinstance(item, dict):
            continue
        dataset_key = str(item.get("base_dataset") or "")
        if not dataset_key:
            dataset_key = f"__unknown__::{item.get('run_id', '')}"
        prev = dataset_latest.get(dataset_key)
        if prev is None or str(item.get("generated_at", "")) >= str(prev.get("generated_at", "")):
            dataset_latest[dataset_key] = item

    runs_index = sorted(dataset_latest.values(), key=lambda x: x.get("generated_at", ""), reverse=True)
    index_payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "latest_run_id": runs_index[0]["run_id"] if runs_index else args.run_id,
        "runs": runs_index,
    }
    index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")

    all_payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "latest_run_id": args.run_id,
        "runs": [],
    }
    for item in runs_index:
        run_path = item.get("path")
        if not run_path:
            continue
        try:
            run_data = json.loads(Path(run_path).read_text(encoding="utf-8"))
            all_payload["runs"].append(run_data)
        except Exception:
            continue

    all_json = out_www / "experiments_all.json"
    all_json.write_text(json.dumps(all_payload, indent=2), encoding="utf-8")

    latest_json = out_www / "experiments_latest.json"
    latest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run the full experiment pipeline")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--steps", default="1,2,3,4,5,6,7,8", help="Comma-separated steps")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--datasets", default="", help="Comma-separated datasets for step1; empty means use --base-dataset only")
    parser.add_argument("--base-dataset", default="NF-UNSW-NB15-v3.csv")
    parser.add_argument("--ood-dataset", default="NF-BoT-IoT-v3.csv", help="OOD dataset used in step3 retrain cases and step4 ensemble reevaluation.")
    parser.add_argument("--natural-datasets", default="NF-BoT-IoT-v3.csv")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-benign-for-attacks", type=int, default=5000)
    parser.add_argument("--drift-subset-size", type=int, default=3000)
    parser.add_argument("--step5-train-max-samples", type=int, default=200000, help="Cap step5 ablation train samples; 0 means no cap.")
    parser.add_argument("--step5-eval-max-samples", type=int, default=300000, help="Cap step5 ablation eval/test samples; 0 means no cap.")
    parser.add_argument("--step6-val-max-samples", type=int, default=100000, help="Cap step6 calibration/meta-fit validation samples; 0 means no cap.")
    parser.add_argument("--step6-eval-max-samples", type=int, default=30000, help="Cap step6 eval samples after drift-subset rule; 0 means no cap.")
    parser.add_argument("--natural-shift-size", type=int, default=0, help="Optional cap for each natural-shift dataset in step2; 0 means use --drift-subset-size.")
    parser.add_argument("--retrain-metrics", default="random,uncertainty,entropy,gd,ensemble_rank,ensemble_p_value,ensemble_hybrid")
    parser.add_argument("--retrain-budgets", default="0.05,0.1,0.2,0.3")
    parser.add_argument("--retrain-id-ratios", default="0.1,0.3,0.5,0.7")
    parser.add_argument("--ensembles", default="voting,stacking,xgboost", help="Comma-separated ensemble types for step 6")
    parser.add_argument("--epochs", default="20,20,20", help="Comma-separated epochs for AE,LSTM,DSFANet")
    parser.add_argument("--test-size", type=int, default=0, help="If >0, enables test mode using only the first N samples of each dataset")
    args = parser.parse_args()

    if args.datasets and args.datasets.strip():
        args.datasets = parse_str_list(args.datasets)
    else:
        args.datasets = [args.base_dataset]
    if args.natural_datasets:
        args.natural_datasets = parse_str_list(args.natural_datasets)
    args.ensembles = [x.lower() for x in parse_str_list(args.ensembles)]
    args.epochs = parse_int_list(args.epochs)
    if len(args.epochs) != 3:
        raise ValueError("--epochs must provide exactly 3 integers: AE,LSTM,DSFANet")
    steps = set(parse_str_list(args.steps))

    if args.test_size > 0:
        config.TEST_MODE = True
        config.TEST_SIZE = args.test_size
        print(f"Test mode enabled from CLI: using first {args.test_size} samples.")
    else:
        config.TEST_MODE = False

    device = resolve_device(args.device)
    run_dir = ensure_dir(Path("out") / "experiments" / args.run_id)
    run_overview_path = run_dir / f"run_overview_{args.run_id}.csv"

    # print(f"Base dataset: {args.base_dataset}")
    # print(f"Step1 datasets: {args.datasets}")

    summary_rows_map: dict[int, dict[str, int]] = {}
    if run_overview_path.exists():
        try:
            prev_overview = pd.read_csv(run_overview_path)
            for _, r in prev_overview.iterrows():
                step_val = int(r.get("step"))
                rows_val = int(r.get("rows"))
                summary_rows_map[step_val] = {"step": step_val, "rows": rows_val}
        except Exception:
            summary_rows_map = {}

    def add_summary(step_num: int, row_count: int):
        summary_rows_map[int(step_num)] = {"step": int(step_num), "rows": int(row_count)}

    def ensure_exists(path: Path, desc: str):
        if not path.exists():
            raise FileNotFoundError(f"Required {desc} not found: {path}. Run prerequisite steps first.")

    registries = {}
    dataset_packs = {}

    if "1" in steps:
        df1, registries, dataset_packs = step1_benchmarks(args, run_dir, device)
        add_summary(1, len(df1))

    base_key = slug(args.base_dataset)
    needs_base = any(x in steps for x in ["2", "3", "4", "5", "6", "7"])

    if needs_base:
        if not registries:
            reg_path = run_dir / base_key / f"registry_{base_key}.json"
            if "1" not in steps:
                ensure_exists(reg_path, "step1 registry")
            if reg_path.exists():
                registries[base_key] = json.loads(reg_path.read_text(encoding="utf-8"))

        if base_key not in dataset_packs:
            prep = DataPreprocessor(args.base_dataset)
            (x_s_train, x_t_train, y_train), (x_s_test, x_t_test, y_test) = prep.prepare_data()
            dataset_packs[base_key] = (x_s_train, x_t_train, y_train, x_s_test, x_t_test, y_test)
    
    def get_best_models():
        best_path = run_dir / f"best_models_step3_{args.run_id}.json"
        if "3" not in steps:
            ensure_exists(best_path, "step3 best models")
        if best_path.exists():
            return json.loads(best_path.read_text(encoding="utf-8"))
        return {}
    
    def get_transfer_best_models():
        transfer_path = run_dir / f"transfer_best_models_step3_{args.run_id}.json"
        if "3" not in steps:
            ensure_exists(transfer_path, "step3 transfer best models")
        if transfer_path.exists():
            return json.loads(transfer_path.read_text(encoding="utf-8"))
        return {}

    best_models = {}
    transfer_best_models = {}

    if "2" in steps:
        df2 = step2_drift(args, run_dir, device, registries[base_key], dataset_packs[base_key])
        add_summary(2, len(df2))

    if "3" in steps:
        df3, best_models, transfer_best_models = step3_retrain(args, run_dir, device, registries[base_key], dataset_packs[base_key], advs=['pgd', 'gdkde'])
        add_summary(3, len(df3))
    
    if not best_models and any(x in steps for x in ["4", "6", "7"]):
        best_models = get_best_models()
    
    if not transfer_best_models and "7" in steps:
        transfer_best_models = get_transfer_best_models()

    if "4" in steps:
        df4 = step4_best_ensemble_shap(args, run_dir, device, dataset_packs[base_key], best_models, registries[base_key])
        add_summary(4, len(df4))

    if "5" in steps:
        df5 = step5_dsfanet_ablation(args, run_dir, device)
        add_summary(5, len(df5))

    if "6" in steps:
        df6 = step6_ensemble_ablation(args, run_dir, device, registries[base_key], dataset_packs[base_key], best_models)
        add_summary(6, len(df6))

    if "7" in steps:
        df7 = step7_transfer_ensemble_compare(
            args,
            run_dir,
            device,
            registries[base_key],
            dataset_packs[base_key],
            best_models,
            transfer_best_models,
        )
        add_summary(7, len(df7))

    if "8" in steps:
        if "1" not in steps and not any(x in steps for x in ["2", "3", "4", "5", "6", "7"]):
            summary_files = list(run_dir.glob("summary_step*.csv"))
            if not summary_files:
                raise FileNotFoundError(f"No step summary files found under {run_dir}. ")
        step8_export_for_web(run_dir, args)
        add_summary(8, 1)

    summary_rows = [summary_rows_map[k] for k in sorted(summary_rows_map.keys())]
    pd.DataFrame(summary_rows).to_csv(run_overview_path, index=False)
    print(f"Done. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
