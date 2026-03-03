from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from . import config
from .data_loader import DataPreprocessor, get_dataloaders
from .models import Autoencoder, DSFANet, LSTMClassifier
from .runtime import resolve_device


def _train_lstm(
    x_s_train: np.ndarray,
    x_t_train: np.ndarray,
    y_train: np.ndarray,
    x_s_test: np.ndarray,
    x_t_test: np.ndarray,
    y_test: np.ndarray,
    device: str | torch.device = "cpu",
    epochs: int = 3,
) -> LSTMClassifier:
    device = resolve_device(device)
    model = LSTMClassifier(temporal_dim=x_t_train.shape[1], n_classes=config.NUM_CLASSES, device=str(device))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_loader, _ = get_dataloaders(
        (x_s_train, x_t_train, y_train),
        (x_s_test, x_t_test, y_test),
        batch_size=config.BATCH_SIZE,
    )

    model.train()
    for _ in range(epochs):
        for _, x_t, y in train_loader:
            x_t, y = x_t.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x_t)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def _train_autoencoder(
    x_s_train: np.ndarray,
    x_t_train: np.ndarray,
    y_train: np.ndarray,
    x_s_test: np.ndarray,
    x_t_test: np.ndarray,
    y_test: np.ndarray,
    device: str | torch.device = "cpu",
    epochs: int = 3,
) -> Autoencoder:
    device = resolve_device(device)
    model = Autoencoder(input_dim=x_s_train.shape[1], device=str(device))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    train_loader, _ = get_dataloaders(
        (x_s_train, x_t_train, y_train),
        (x_s_test, x_t_test, y_test),
        batch_size=config.BATCH_SIZE,
    )

    model.train()
    for _ in range(epochs):
        for x_s, _, _ in train_loader:
            x_s = x_s.to(device)
            optimizer.zero_grad()
            recon = model(x_s)
            loss = criterion(recon, x_s)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def train_lstm_model(
    x_s_train: np.ndarray,
    x_t_train: np.ndarray,
    y_train: np.ndarray,
    x_s_test: np.ndarray,
    x_t_test: np.ndarray,
    y_test: np.ndarray,
    device: str | torch.device = "cpu",
    epochs: int = 3,
) -> LSTMClassifier:
    return _train_lstm(
        x_s_train=x_s_train,
        x_t_train=x_t_train,
        y_train=y_train,
        x_s_test=x_s_test,
        x_t_test=x_t_test,
        y_test=y_test,
        device=device,
        epochs=epochs,
    )


def train_autoencoder_model(
    x_s_train: np.ndarray,
    x_t_train: np.ndarray,
    y_train: np.ndarray,
    x_s_test: np.ndarray,
    x_t_test: np.ndarray,
    y_test: np.ndarray,
    device: str | torch.device = "cpu",
    epochs: int = 3,
) -> Autoencoder:
    return _train_autoencoder(
        x_s_train=x_s_train,
        x_t_train=x_t_train,
        y_train=y_train,
        x_s_test=x_s_test,
        x_t_test=x_t_test,
        y_test=y_test,
        device=device,
        epochs=epochs,
    )


def _to_jsonable(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in records:
        converted = {}
        for k, v in item.items():
            if isinstance(v, (np.float32, np.float64)):
                converted[k] = float(v)
            elif isinstance(v, (np.int32, np.int64)):
                converted[k] = int(v)
            else:
                converted[k] = v
        out.append(converted)
    return out


def analyze_lstm_shap(
    model: LSTMClassifier,
    x_temporal: np.ndarray,
    temporal_feature_names: list[str],
    out_dir: str | Path,
    background_size: int = 128,
    explain_size: int = 256,
) -> dict[str, Any]:
    try:
        import shap
    except ImportError as ex:
        raise RuntimeError("Missing dependency 'shap'. Install it first.") from ex

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    x_temporal = x_temporal.astype(np.float32)
    background = torch.tensor(x_temporal[:background_size], dtype=torch.float32, device=model.device)
    explain = torch.tensor(x_temporal[:explain_size], dtype=torch.float32, device=model.device)

    class ProbWrapper(nn.Module):
        def __init__(self, base_model: LSTMClassifier):
            super().__init__()
            self.base = base_model

        def forward(self, x_t: torch.Tensor) -> torch.Tensor:
            logits = self.base(x_t)
            probs = torch.softmax(logits, dim=1)
            return probs[:, 1:2]

    wrapped = ProbWrapper(model).to(model.device)
    wrapped.eval()

    explainer = shap.DeepExplainer(wrapped, background)
    shap_values = explainer.shap_values(explain)

    if isinstance(shap_values, list):
        values = shap_values[0]
    else:
        values = shap_values

    if values.ndim == 3:
        values = values[:, :, 0]

    feature_importance = np.mean(np.abs(values), axis=0)
    importance_df = pd.DataFrame(
        {
            "feature": temporal_feature_names,
            "mean_abs_shap": feature_importance,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    per_sample_records: list[dict[str, Any]] = []
    top_k = min(10, len(temporal_feature_names))
    top_features = importance_df["feature"].head(top_k).tolist()

    explain_np = explain.detach().cpu().numpy()
    for i in range(explain_np.shape[0]):
        row: dict[str, Any] = {"sample_id": i}
        for feat in top_features:
            feat_idx = temporal_feature_names.index(feat)
            row[f"value::{feat}"] = explain_np[i, feat_idx]
            row[f"shap::{feat}"] = values[i, feat_idx]
        per_sample_records.append(row)

    importance_csv = out_path / "shap_lstm_importance.csv"
    samples_json = out_path / "shap_lstm_samples.json"
    importance_df.to_csv(importance_csv, index=False)
    samples_json.write_text(json.dumps(_to_jsonable(per_sample_records), indent=2), encoding="utf-8")

    return {
        "model": "lstm",
        "importance_csv": str(importance_csv),
        "samples_json": str(samples_json),
        "top_features": _to_jsonable(importance_df.head(20).to_dict(orient="records")),
    }


def analyze_ae_shap(
    model: Autoencoder,
    x_static: np.ndarray,
    static_feature_names: list[str],
    out_dir: str | Path,
    background_size: int = 64,
    explain_size: int = 128,
    nsamples: int = 120,
) -> dict[str, Any]:
    try:
        import shap
    except ImportError as ex:
        raise RuntimeError("Missing dependency 'shap'. Install it first.") from ex

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    x_static = x_static.astype(np.float32)
    background = x_static[:background_size]
    explain = x_static[:explain_size]

    def recon_error(input_x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor_x = torch.tensor(input_x, dtype=torch.float32, device=model.device)
            recon = model(tensor_x)
            err = torch.mean((recon - tensor_x) ** 2, dim=1)
            return err.detach().cpu().numpy()

    explainer = shap.KernelExplainer(recon_error, background)
    shap_values = explainer.shap_values(explain, nsamples=nsamples)

    values = np.array(shap_values)
    if values.ndim == 3:
        values = values[0]

    feature_importance = np.mean(np.abs(values), axis=0)
    importance_df = pd.DataFrame(
        {
            "feature": static_feature_names,
            "mean_abs_shap": feature_importance,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    per_sample_records: list[dict[str, Any]] = []
    top_k = min(10, len(static_feature_names))
    top_features = importance_df["feature"].head(top_k).tolist()

    for i in range(explain.shape[0]):
        row: dict[str, Any] = {"sample_id": i}
        for feat in top_features:
            feat_idx = static_feature_names.index(feat)
            row[f"value::{feat}"] = float(explain[i, feat_idx])
            row[f"shap::{feat}"] = float(values[i, feat_idx])
        per_sample_records.append(row)

    importance_csv = out_path / "shap_ae_importance.csv"
    samples_json = out_path / "shap_ae_samples.json"
    importance_df.to_csv(importance_csv, index=False)
    samples_json.write_text(json.dumps(_to_jsonable(per_sample_records), indent=2), encoding="utf-8")

    return {
        "model": "autoencoder",
        "importance_csv": str(importance_csv),
        "samples_json": str(samples_json),
        "top_features": _to_jsonable(importance_df.head(20).to_dict(orient="records")),
    }


def analyze_dsfanet_shap(
    model: DSFANet,
    x_static: np.ndarray,
    x_temporal: np.ndarray,
    static_feature_names: list[str],
    temporal_feature_names: list[str],
    out_dir: str | Path,
    background_size: int = 96,
    explain_size: int = 160,
) -> dict[str, Any]:
    try:
        import shap
    except ImportError as ex:
        raise RuntimeError("Missing dependency 'shap'. Install it first.") from ex

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    x_static = x_static.astype(np.float32)
    x_temporal = x_temporal.astype(np.float32)
    x_cat = np.concatenate([x_static, x_temporal], axis=1)
    feature_names = [f"static::{f}" for f in static_feature_names] + [f"temporal::{f}" for f in temporal_feature_names]

    class ProbWrapper(nn.Module):
        def __init__(self, base_model: DSFANet, static_dim: int):
            super().__init__()
            self.base = base_model
            self.static_dim = static_dim

        def forward(self, x_cat_tensor: torch.Tensor) -> torch.Tensor:
            x_s = x_cat_tensor[:, : self.static_dim]
            x_t = x_cat_tensor[:, self.static_dim :]
            logits = self.base(x_s, x_t)
            probs = torch.softmax(logits, dim=1)
            return probs[:, 1:2]

    wrapped = ProbWrapper(model, static_dim=x_static.shape[1]).to(model.device)
    wrapped.eval()

    background = torch.tensor(x_cat[:background_size], dtype=torch.float32, device=model.device)
    explain = torch.tensor(x_cat[:explain_size], dtype=torch.float32, device=model.device)

    values = None
    try:
        explainer = shap.DeepExplainer(wrapped, background)
        shap_values = explainer.shap_values(explain)
        values = shap_values[0] if isinstance(shap_values, list) else shap_values
    except Exception:
        explainer = shap.GradientExplainer(wrapped, background)
        shap_values = explainer.shap_values(explain)
        values = shap_values[0] if isinstance(shap_values, list) else shap_values

    if values.ndim == 3:
        values = values[:, :, 0]

    feature_importance = np.mean(np.abs(values), axis=0)
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": feature_importance,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    top_k = min(12, len(feature_names))
    top_features = importance_df["feature"].head(top_k).tolist()

    explain_np = explain.detach().cpu().numpy()
    per_sample_records: list[dict[str, Any]] = []
    for i in range(explain_np.shape[0]):
        row: dict[str, Any] = {"sample_id": i}
        for feat in top_features:
            feat_idx = feature_names.index(feat)
            row[f"value::{feat}"] = float(explain_np[i, feat_idx])
            row[f"shap::{feat}"] = float(values[i, feat_idx])
        per_sample_records.append(row)

    importance_csv = out_path / "shap_dsfanet_importance.csv"
    samples_json = out_path / "shap_dsfanet_samples.json"
    importance_df.to_csv(importance_csv, index=False)
    samples_json.write_text(json.dumps(_to_jsonable(per_sample_records), indent=2), encoding="utf-8")

    return {
        "model": "dsfanet",
        "importance_csv": str(importance_csv),
        "samples_json": str(samples_json),
        "top_features": _to_jsonable(importance_df.head(20).to_dict(orient="records")),
    }


def run_shap_analysis(
    csv_path: str = "NF-UNSW-NB15-v3.csv",
    out_dir: str | Path = "out/www",
    run_lstm: bool = True,
    run_ae: bool = True,
    run_dsfanet: bool = True,
    device: str | torch.device = "cpu",
    max_train_samples: int = 20000,
) -> dict[str, Any]:
    device = resolve_device(device)
    preprocessor = DataPreprocessor(csv_path)
    (x_s_train, x_t_train, y_train), (x_s_test, x_t_test, y_test) = preprocessor.prepare_data()

    if max_train_samples > 0 and len(y_train) > max_train_samples:
        idx = np.random.RandomState(42).choice(len(y_train), size=max_train_samples, replace=False)
        x_s_train = x_s_train[idx]
        x_t_train = x_t_train[idx]
        y_train = y_train[idx]

    static_feature_names = preprocessor.used_static_cols
    temporal_feature_names = preprocessor.used_temporal_cols

    results: dict[str, Any] = {
        "dataset": csv_path,
        "device": str(device),
        "models": {},
    }

    if run_lstm:
        lstm = _train_lstm(x_s_train, x_t_train, y_train, x_s_test, x_t_test, y_test, device=device, epochs=3)
        results["models"]["lstm"] = analyze_lstm_shap(
            lstm,
            x_t_test,
            temporal_feature_names,
            out_dir=out_dir,
        )

    if run_ae:
        ae = _train_autoencoder(x_s_train, x_t_train, y_train, x_s_test, x_t_test, y_test, device=device, epochs=3)
        results["models"]["autoencoder"] = analyze_ae_shap(
            ae,
            x_s_test,
            static_feature_names,
            out_dir=out_dir,
        )

    if run_dsfanet:
        dsfanet = DSFANet(
            static_dim=x_s_train.shape[1],
            temporal_dim=x_t_train.shape[1],
            n_classes=config.NUM_CLASSES,
            device=str(device),
        )
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(dsfanet.parameters(), lr=config.LEARNING_RATE)
        train_loader, _ = get_dataloaders(
            (x_s_train, x_t_train, y_train),
            (x_s_test, x_t_test, y_test),
            batch_size=config.BATCH_SIZE,
        )
        dsfanet.train()
        for _ in range(2):
            for x_s, x_t, y in train_loader:
                x_s, x_t, y = x_s.to(device), x_t.to(device), y.to(device)
                optimizer.zero_grad()
                logits = dsfanet(x_s, x_t)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
        dsfanet.eval()

        results["models"]["dsfanet"] = analyze_dsfanet_shap(
            model=dsfanet,
            x_static=x_s_test,
            x_temporal=x_t_test,
            static_feature_names=static_feature_names,
            temporal_feature_names=temporal_feature_names,
            out_dir=out_dir,
        )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    summary_path = out_path / "shap_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    results["summary_json"] = str(summary_path)

    return results


if __name__ == "__main__":
    report = run_shap_analysis()
    print(json.dumps(report, indent=2))
