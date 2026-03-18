from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.svm import SVC

from src.attacker import FGSMAttack, PGDAttack
from src.data_loader import DataPreprocessor
from src.models import Autoencoder, DSFANet
from src.models.ensemble import StackingEnsemble, UnificationLayer, VotingEnsemble
from src.runtime import resolve_device
from src.shap_analysis import analyze_ae_shap, analyze_dsfanet_shap, analyze_lstm_shap, train_autoencoder_model, train_lstm_model


def _float(v: Any) -> float:
    if isinstance(v, (np.floating, np.float32, np.float64)):
        return float(v)
    return float(v)


def _evaluate_binary(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    return {
        "accuracy": _float(accuracy_score(y_true, preds)),
        "precision": _float(precision_score(y_true, preds, zero_division=0)),
        "recall": _float(recall_score(y_true, preds, zero_division=0)),
        "f1": _float(f1_score(y_true, preds, zero_division=0)),
        "average_precision": _float(average_precision_score(y_true, probs)),
        "confusion": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "threshold": threshold,
    }


def _dataset_overview(
    x_s_train: np.ndarray,
    x_t_train: np.ndarray,
    y_train: np.ndarray,
    x_s_test: np.ndarray,
    x_t_test: np.ndarray,
    y_test: np.ndarray,
    static_feature_names: list[str],
    temporal_feature_names: list[str],
) -> dict[str, Any]:
    train_counts = np.bincount(y_train.astype(int), minlength=2)
    test_counts = np.bincount(y_test.astype(int), minlength=2)

    def _top_feature_stats(x: np.ndarray, names: list[str], top_n: int = 10) -> list[dict[str, Any]]:
        variances = np.var(x, axis=0)
        order = np.argsort(variances)[::-1][: min(top_n, len(names))]
        rows = []
        for idx in order:
            rows.append(
                {
                    "feature": names[idx],
                    "mean": float(np.mean(x[:, idx])),
                    "std": float(np.std(x[:, idx])),
                    "min": float(np.min(x[:, idx])),
                    "max": float(np.max(x[:, idx])),
                }
            )
        return rows

    return {
        "shape": {
            "train_static": [int(x_s_train.shape[0]), int(x_s_train.shape[1])],
            "train_temporal": [int(x_t_train.shape[0]), int(x_t_train.shape[1])],
            "test_static": [int(x_s_test.shape[0]), int(x_s_test.shape[1])],
            "test_temporal": [int(x_t_test.shape[0]), int(x_t_test.shape[1])],
        },
        "class_distribution": {
            "train": {"benign": int(train_counts[0]), "malicious": int(train_counts[1])},
            "test": {"benign": int(test_counts[0]), "malicious": int(test_counts[1])},
        },
        "feature_stats": {
            "static_top_variance": _top_feature_stats(x_s_test, static_feature_names, top_n=10),
            "temporal_top_variance": _top_feature_stats(x_t_test, temporal_feature_names, top_n=10),
        },
    }


def _train_dsfanet(
    x_s_train: np.ndarray,
    x_t_train: np.ndarray,
    y_train: np.ndarray,
    x_s_test: np.ndarray,
    x_t_test: np.ndarray,
    y_test: np.ndarray,
    device: str | torch.device,
    epochs: int = 2,
) -> DSFANet:
    model = DSFANet(
        static_dim=x_s_train.shape[1],
        temporal_dim=x_t_train.shape[1],
        n_classes=2,
        device=str(device),
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    x_s_t = torch.tensor(x_s_train, dtype=torch.float32)
    x_t_t = torch.tensor(x_t_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)

    batch_size = 128
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(x_s_t.shape[0])
        for i in range(0, x_s_t.shape[0], batch_size):
            idx = perm[i : i + batch_size]
            bx_s = x_s_t[idx].to(device)
            bx_t = x_t_t[idx].to(device)
            by = y_t[idx].to(device)
            optimizer.zero_grad()
            logits = model(bx_s, bx_t)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def _torch_probs(model: torch.nn.Module, x_s: np.ndarray, x_t: np.ndarray, input_req: str, device: str | torch.device) -> np.ndarray:
    batch_size = 1024 if resolve_device(device).type == "cuda" else 4096
    probs_batches: list[np.ndarray] = []
    with torch.no_grad():
        total = x_s.shape[0]
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            if input_req == "both":
                logits = model(
                    torch.tensor(x_s[start:end], dtype=torch.float32, device=device),
                    torch.tensor(x_t[start:end], dtype=torch.float32, device=device),
                )
            elif input_req == "temporal":
                logits = model(torch.tensor(x_t[start:end], dtype=torch.float32, device=device))
            else:
                logits = model(torch.tensor(x_s[start:end], dtype=torch.float32, device=device))
            probs_batches.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())

    if not probs_batches:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(probs_batches, axis=0)


def _make_pr_curve(y_true: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    p_arr, r_arr, t_arr = precision_recall_curve(y_true, probs)
    return {
        "precision": [float(x) for x in p_arr],
        "recall": [float(x) for x in r_arr],
        "thresholds": [float(x) for x in t_arr],
    }


def build_dashboard_data(
    dataset: str,
    out_dir: str | Path = "out/www",
    device: str | torch.device = "cpu",
    max_train_samples: int = 20000,
    shap_background_size: int = 128,
    shap_explain_size: int = 256,
) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = resolve_device(device)
    preprocessor = DataPreprocessor(dataset)
    (x_s_train, x_t_train, y_train), (x_s_test, x_t_test, y_test) = preprocessor.prepare_data()

    if max_train_samples > 0 and len(y_train) > max_train_samples:
        idx = np.random.RandomState(42).choice(len(y_train), size=max_train_samples, replace=False)
        x_s_train = x_s_train[idx]
        x_t_train = x_t_train[idx]
        y_train = y_train[idx]

    val_size = max(1000, int(len(y_train) * 0.2))
    x_s_val, x_t_val, y_val = x_s_train[:val_size], x_t_train[:val_size], y_train[:val_size]
    x_s_train_sub, x_t_train_sub, y_train_sub = x_s_train[val_size:], x_t_train[val_size:], y_train[val_size:]

    lstm_model = train_lstm_model(
        x_s_train=x_s_train_sub,
        x_t_train=x_t_train_sub,
        y_train=y_train_sub,
        x_s_test=x_s_test,
        x_t_test=x_t_test,
        y_test=y_test,
        device=device,
        epochs=3,
    )

    dsfanet_model = _train_dsfanet(
        x_s_train=x_s_train_sub,
        x_t_train=x_t_train_sub,
        y_train=y_train_sub,
        x_s_test=x_s_test,
        x_t_test=x_t_test,
        y_test=y_test,
        device=device,
        epochs=2,
    )

    ae_model = train_autoencoder_model(
        x_s_train=x_s_train,
        x_t_train=x_t_train,
        y_train=y_train,
        x_s_test=x_s_test,
        x_t_test=x_t_test,
        y_test=y_test,
        device=device,
        epochs=3,
    )

    rf_model = RandomForestClassifier(n_estimators=80, max_depth=12, random_state=42)
    rf_model.fit(x_s_train_sub, y_train_sub)

    svm_model = SVC(probability=True, kernel="rbf", max_iter=1500, random_state=42)
    svm_model.fit(x_s_train_sub, y_train_sub)

    lstm_probs = _torch_probs(lstm_model, x_s_test, x_t_test, input_req="temporal", device=device)
    dsfanet_probs = _torch_probs(dsfanet_model, x_s_test, x_t_test, input_req="both", device=device)
    rf_probs = rf_model.predict_proba(x_s_test)[:, 1]
    svm_probs = svm_model.predict_proba(x_s_test)[:, 1]

    with torch.no_grad():
        x_s_test_t = torch.tensor(x_s_test, dtype=torch.float32, device=device)
        ae_recon_test = ae_model(x_s_test_t).detach().cpu().numpy()
        ae_err_test = np.mean((ae_recon_test - x_s_test) ** 2, axis=1)

        x_s_train_t = torch.tensor(x_s_train_sub, dtype=torch.float32, device=device)
        ae_recon_train = ae_model(x_s_train_t).detach().cpu().numpy()
        ae_err_train = np.mean((ae_recon_train - x_s_train_sub) ** 2, axis=1)

    ae_min, ae_max = float(np.min(ae_err_train)), float(np.max(ae_err_train))
    ae_denom = max(ae_max - ae_min, 1e-8)
    ae_probs = np.clip((ae_err_test - ae_min) / ae_denom, 0.0, 1.0)

    unifier = UnificationLayer()
    voting = VotingEnsemble(unifier=unifier, weights={"DSFANet": 2.0, "LSTM": 1.5, "RF": 1.2, "SVM": 1.0, "AE": 1.0}, device=str(device))
    stacking = StackingEnsemble(unifier=unifier, device=str(device))

    models_config = [
        ("DSFANet", dsfanet_model, "classifier", "both"),
        ("LSTM", lstm_model, "classifier", "temporal"),
        ("RF", rf_model, "classifier", "static"),
        ("SVM", svm_model, "classifier", "static"),
        ("AE", ae_model, "anomaly", "static"),
    ]

    for cfg in models_config:
        voting.add_model(*cfg)
        stacking.add_model(*cfg)

    voting.calibrate(x_s_val, x_t_val)
    stacking.fit_meta(x_s_val, x_t_val, y_val)

    voting_probs = voting.predict(x_s_test, x_t_test)
    stacking_probs = stacking.predict(x_s_test, x_t_test)

    static_feature_names = preprocessor.used_static_cols
    temporal_feature_names = preprocessor.used_temporal_cols

    shap_lstm = analyze_lstm_shap(
        model=lstm_model,
        x_temporal=x_t_test,
        temporal_feature_names=temporal_feature_names,
        out_dir=out_path,
        background_size=shap_background_size,
        explain_size=shap_explain_size,
    )

    shap_ae = analyze_ae_shap(
        model=ae_model,
        x_static=x_s_test,
        static_feature_names=static_feature_names,
        out_dir=out_path,
        background_size=min(64, len(x_s_test)),
        explain_size=min(120, len(x_s_test)),
        nsamples=100,
    )

    shap_dsfanet = analyze_dsfanet_shap(
        model=dsfanet_model,
        x_static=x_s_test,
        x_temporal=x_t_test,
        static_feature_names=static_feature_names,
        temporal_feature_names=temporal_feature_names,
        out_dir=out_path,
        background_size=min(96, len(x_s_test)),
        explain_size=min(140, len(x_s_test)),
    )

    model_scores: dict[str, np.ndarray] = {
        "LSTM": lstm_probs,
        "DSFANet": dsfanet_probs,
        "RF": rf_probs,
        "SVM": svm_probs,
        "Autoencoder": ae_probs,
        "VotingEnsemble": voting_probs,
        "StackingEnsemble": stacking_probs,
    }

    benchmark_rows = []
    for name, probs in model_scores.items():
        metrics = _evaluate_binary(y_test, probs)
        benchmark_rows.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "average_precision": metrics["average_precision"],
            }
        )

    benchmark_rows.sort(key=lambda x: x["average_precision"], reverse=True)

    hist_counts, hist_edges = np.histogram(voting_probs, bins=20, range=(0.0, 1.0))
    score_histogram = {
        "bins": [float(x) for x in hist_edges.tolist()],
        "counts": [int(x) for x in hist_counts.tolist()],
    }

    drift_probs = voting_probs
    n_windows = 12
    win_size = max(1, len(drift_probs) // n_windows)
    drift_points = []
    for w in range(n_windows):
        start = w * win_size
        end = len(drift_probs) if w == n_windows - 1 else min(len(drift_probs), (w + 1) * win_size)
        part_prob = drift_probs[start:end]
        part_label = y_test[start:end]
        if len(part_prob) == 0:
            continue
        drift_points.append(
            {
                "window": w + 1,
                "mean_score": float(np.mean(part_prob)),
                "positive_ratio": float(np.mean(part_label == 1)),
                "count": int(len(part_prob)),
            }
        )

    subset_n = min(1500, len(y_test))
    idx_subset = np.random.RandomState(42).choice(len(y_test), size=subset_n, replace=False)
    x_s_subset = torch.tensor(x_s_test[idx_subset], dtype=torch.float32)
    x_t_subset = torch.tensor(x_t_test[idx_subset], dtype=torch.float32)
    y_subset = torch.tensor(y_test[idx_subset], dtype=torch.long)

    fgsm = FGSMAttack(dsfanet_model, device=str(device), epsilon=0.08)
    pgd = PGDAttack(dsfanet_model, device=str(device), epsilon=0.08, steps=6, alpha=0.02)
    adv_fgsm_s, adv_fgsm_t = fgsm.generate(x_s_subset, x_t_subset, y_subset)
    adv_pgd_s, adv_pgd_t = pgd.generate(x_s_subset, x_t_subset, y_subset)

    adv_sets = {
        "clean": (x_s_subset.numpy(), x_t_subset.numpy(), y_subset.numpy()),
        "fgsm": (adv_fgsm_s.detach().cpu().numpy(), adv_fgsm_t.detach().cpu().numpy(), y_subset.numpy()),
        "pgd": (adv_pgd_s.detach().cpu().numpy(), adv_pgd_t.detach().cpu().numpy(), y_subset.numpy()),
    }

    attack_rows: list[dict[str, Any]] = []
    for atk_name, (ax_s, ax_t, ay) in adv_sets.items():
        ds_probs = _torch_probs(dsfanet_model, ax_s, ax_t, input_req="both", device=device)
        vt_probs = voting.predict(ax_s, ax_t)
        ds_metrics = _evaluate_binary(ay, ds_probs)
        vt_metrics = _evaluate_binary(ay, vt_probs)
        attack_rows.append(
            {
                "attack": atk_name,
                "model": "DSFANet",
                "accuracy": ds_metrics["accuracy"],
                "recall": ds_metrics["recall"],
                "f1": ds_metrics["f1"],
                "average_precision": ds_metrics["average_precision"],
            }
        )
        attack_rows.append(
            {
                "attack": atk_name,
                "model": "VotingEnsemble",
                "accuracy": vt_metrics["accuracy"],
                "recall": vt_metrics["recall"],
                "f1": vt_metrics["f1"],
                "average_precision": vt_metrics["average_precision"],
            }
        )

    model_details: dict[str, Any] = {}
    for name, probs in model_scores.items():
        model_details[name] = {
            "metrics": _evaluate_binary(y_test, probs),
            "pr_curve": _make_pr_curve(y_test, probs),
            "score_summary": {
                "mean": float(np.mean(probs)),
                "std": float(np.std(probs)),
                "min": float(np.min(probs)),
                "max": float(np.max(probs)),
            },
        }

    if hasattr(rf_model, "feature_importances_"):
        rf_top = np.argsort(rf_model.feature_importances_)[::-1][: min(15, len(static_feature_names))]
        model_details["RF"]["top_features"] = [
            {
                "feature": static_feature_names[i],
                "importance": float(rf_model.feature_importances_[i]),
            }
            for i in rf_top
        ]
    model_details["LSTM"]["top_features"] = shap_lstm["top_features"][:15]
    model_details["Autoencoder"]["top_features"] = shap_ae["top_features"][:15]
    model_details["DSFANet"]["top_features"] = shap_dsfanet["top_features"][:15]

    top_idx = np.argsort(voting_probs)[::-1][:400]
    alert_rows = []
    sample_details: dict[str, Any] = {}
    for rank, i in enumerate(top_idx, start=1):
        row = {
            "rank": rank,
            "sample_id": int(i),
            "voting_score": float(voting_probs[i]),
            "stacking_score": float(stacking_probs[i]),
            "pred": int(voting_probs[i] >= 0.5),
            "label": int(y_test[i]),
        }
        for feat in temporal_feature_names[:5]:
            feat_idx = temporal_feature_names.index(feat)
            row[f"temporal::{feat}"] = float(x_t_test[i, feat_idx])
        for feat in static_feature_names[:3]:
            feat_idx = static_feature_names.index(feat)
            row[f"static::{feat}"] = float(x_s_test[i, feat_idx])
        alert_rows.append(row)

        sample_details[str(int(i))] = {
            "sample_id": int(i),
            "label": int(y_test[i]),
            "model_scores": {k: float(v[i]) for k, v in model_scores.items()},
            "top_static_features": [
                {"feature": feat, "value": float(x_s_test[i, static_feature_names.index(feat)])}
                for feat in static_feature_names[:10]
            ],
            "top_temporal_features": [
                {"feature": feat, "value": float(x_t_test[i, temporal_feature_names.index(feat)])}
                for feat in temporal_feature_names[:10]
            ],
        }

    alerts_df = pd.DataFrame(alert_rows)
    alerts_csv = out_path / "alerts_top.csv"
    alerts_df.to_csv(alerts_csv, index=False)

    summary_metrics = _evaluate_binary(y_test, voting_probs)
    dataset_report = _dataset_overview(
        x_s_train=x_s_train,
        x_t_train=x_t_train,
        y_train=y_train,
        x_s_test=x_s_test,
        x_t_test=x_t_test,
        y_test=y_test,
        static_feature_names=static_feature_names,
        temporal_feature_names=temporal_feature_names,
    )

    data = {
        "meta": {
            "dataset": dataset,
            "device": str(device),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "primary_model": "VotingEnsemble",
        },
        "dataset_overview": dataset_report,
        "metrics": {
            "accuracy": summary_metrics["accuracy"],
            "precision": summary_metrics["precision"],
            "recall": summary_metrics["recall"],
            "f1": summary_metrics["f1"],
            "average_precision": summary_metrics["average_precision"],
        },
        "confusion": summary_metrics["confusion"],
        "pr_curve": _make_pr_curve(y_test, voting_probs),
        "score_histogram": score_histogram,
        "drift_windows": drift_points,
        "alerts_preview": alert_rows[:200],
        "shap_top_features": shap_lstm["top_features"][:20],
        "shap_by_model": {
            "LSTM": shap_lstm["top_features"][:20],
            "Autoencoder": shap_ae["top_features"][:20],
            "DSFANet": shap_dsfanet["top_features"][:20],
        },
        "benchmark_models": benchmark_rows,
        "attack_results": attack_rows,
        "model_details": model_details,
        "sample_ids": [int(i) for i in top_idx[:200]],
    }

    model_json = out_path / "model_details.json"
    sample_json = out_path / "sample_details.json"
    json_path = out_path / "dashboard_data.json"
    model_json.write_text(json.dumps(model_details, indent=2), encoding="utf-8")
    sample_json.write_text(json.dumps(sample_details, indent=2), encoding="utf-8")
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return {
        "dashboard_json": str(json_path),
        "alerts_csv": str(alerts_csv),
        "model_json": str(model_json),
        "sample_json": str(sample_json),
        "shap": {
            "LSTM": shap_lstm,
            "Autoencoder": shap_ae,
            "DSFANet": shap_dsfanet,
        },
    }


def serve_dashboard(www_dir: str | Path = "www", data_dir: str | Path = "out/www", host: str = "127.0.0.1", port: int = 8000):
    from flask import Flask, jsonify
    from flask_cors import CORS

    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    data_root = Path(data_dir)

    @app.route("/")
    def index():
        return jsonify(
            {
                "message": "IDS dashboard API is running.",
                "endpoints": [
                    "/api/dashboard",
                    "/api/alerts",
                    "/api/models",
                    "/api/model/<name>",
                    "/api/sample/<sample_id>",
                ],
            }
        )

    @app.route("/api/dashboard")
    def api_dashboard():
        path = data_root / "dashboard_data.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        return jsonify(payload)

    @app.route("/api/alerts")
    def api_alerts():
        path = data_root / "alerts_top.csv"
        rows = pd.read_csv(path).fillna(0).to_dict(orient="records")
        return jsonify(rows)

    @app.route("/api/models")
    def api_models():
        path = data_root / "model_details.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        return jsonify(payload)

    @app.route("/api/model/<name>")
    def api_model(name: str):
        path = data_root / "model_details.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = name
        if key not in payload:
            return jsonify({"error": f"model '{name}' not found"}), 404
        return jsonify(payload[key])

    @app.route("/api/sample/<sample_id>")
    def api_sample(sample_id: str):
        path = data_root / "sample_details.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if sample_id not in payload:
            return jsonify({"error": f"sample '{sample_id}' not found"}), 404
        return jsonify(payload[sample_id])

    @app.route("/api/experiments/latest")
    def api_experiments_latest():
        path = data_root / "experiments_latest.json"
        if not path.exists():
            return jsonify({"error": "experiments_latest.json not found. Run experiments_main.py with step 8."}), 404
        payload = json.loads(path.read_text(encoding="utf-8"))
        return jsonify(payload)

    @app.route("/api/experiments/index")
    def api_experiments_index():
        path = data_root / "experiments_index.json"
        if not path.exists():
            return jsonify({"error": "experiments_index.json not found. Run experiments_main.py with step 8."}), 404
        payload = json.loads(path.read_text(encoding="utf-8"))
        return jsonify(payload)

    @app.route("/api/experiments/all")
    def api_experiments_all():
        path = data_root / "experiments_all.json"
        if not path.exists():
            return jsonify({"error": "experiments_all.json not found. Run experiments_main.py with step 8."}), 404
        payload = json.loads(path.read_text(encoding="utf-8"))
        return jsonify(payload)

    app.run(host=host, port=port)


def main():
    parser = argparse.ArgumentParser(description="Offline IDS dashboard pipeline")
    parser.add_argument("--dataset", default="NF-UNSW-NB15-v3.csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--skip-serve", action="store_true")
    parser.add_argument("--serve-only", action="store_true")
    args = parser.parse_args()

    if not args.serve_only:
        report = build_dashboard_data(dataset=args.dataset, device=args.device)
        print(json.dumps(report, indent=2))

    if not args.skip_serve:
        serve_dashboard(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
