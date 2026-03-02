from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from src.data_loader import DataPreprocessor
from src.runtime import resolve_device
from src.shap_analysis import analyze_lstm_shap, train_lstm_model


def _float(v: Any) -> float:
    if isinstance(v, (np.floating, np.float32, np.float64)):
        return float(v)
    return float(v)


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

    model = train_lstm_model(
        x_s_train=x_s_train,
        x_t_train=x_t_train,
        y_train=y_train,
        x_s_test=x_s_test,
        x_t_test=x_t_test,
        y_test=y_test,
        device=device,
        epochs=3,
    )

    with torch.no_grad():
        logits = model(torch.tensor(x_t_test, dtype=torch.float32, device=device))
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

    preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": _float(accuracy_score(y_test, preds)),
        "precision": _float(precision_score(y_test, preds, zero_division=0)),
        "recall": _float(recall_score(y_test, preds, zero_division=0)),
        "f1": _float(f1_score(y_test, preds, zero_division=0)),
        "average_precision": _float(average_precision_score(y_test, probs)),
    }

    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    confusion = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    p_arr, r_arr, t_arr = precision_recall_curve(y_test, probs)
    pr_curve = {
        "precision": [float(x) for x in p_arr],
        "recall": [float(x) for x in r_arr],
        "thresholds": [float(x) for x in t_arr],
    }

    hist_counts, hist_edges = np.histogram(probs, bins=20, range=(0.0, 1.0))
    score_histogram = {
        "bins": [float(x) for x in hist_edges.tolist()],
        "counts": [int(x) for x in hist_counts.tolist()],
    }

    static_feature_names = preprocessor.used_static_cols
    temporal_feature_names = preprocessor.used_temporal_cols

    alert_rows = []
    top_idx = np.argsort(probs)[::-1][:300]
    for rank, i in enumerate(top_idx, start=1):
        row = {
            "rank": rank,
            "sample_id": int(i),
            "score": float(probs[i]),
            "pred": int(preds[i]),
            "label": int(y_test[i]),
        }

        for feat in temporal_feature_names[:5]:
            feat_idx = temporal_feature_names.index(feat)
            row[f"feature::{feat}"] = float(x_t_test[i, feat_idx])

        for feat in static_feature_names[:3]:
            feat_idx = static_feature_names.index(feat)
            row[f"feature::{feat}"] = float(x_s_test[i, feat_idx])

        alert_rows.append(row)

    alerts_df = pd.DataFrame(alert_rows)
    alerts_csv = out_path / "alerts_top.csv"
    alerts_df.to_csv(alerts_csv, index=False)

    n_windows = 10
    win_size = max(1, len(probs) // n_windows)
    drift_points = []
    for w in range(n_windows):
        start = w * win_size
        end = len(probs) if w == n_windows - 1 else min(len(probs), (w + 1) * win_size)
        part_prob = probs[start:end]
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

    shap_report = analyze_lstm_shap(
        model=model,
        x_temporal=x_t_test,
        temporal_feature_names=temporal_feature_names,
        out_dir=out_path,
        background_size=shap_background_size,
        explain_size=shap_explain_size,
    )

    data = {
        "meta": {
            "dataset": dataset,
            "device": str(device),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "model": "LSTMClassifier",
        },
        "metrics": metrics,
        "confusion": confusion,
        "pr_curve": pr_curve,
        "score_histogram": score_histogram,
        "drift_windows": drift_points,
        "alerts_preview": alert_rows[:200],
        "shap_top_features": shap_report["top_features"][:20],
    }

    json_path = out_path / "dashboard_data.json"
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return {
        "dashboard_json": str(json_path),
        "alerts_csv": str(alerts_csv),
        "shap": shap_report,
    }


def serve_dashboard(www_dir: str | Path = "www", data_dir: str | Path = "out/www", host: str = "127.0.0.1", port: int = 8000):
    try:
        from flask import Flask, jsonify, send_from_directory
    except ImportError as ex:
        raise RuntimeError("Missing dependency 'flask'. Install it first.") from ex

    app = Flask(__name__, static_folder=str(www_dir), static_url_path="")
    data_root = Path(data_dir)

    @app.route("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

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

    app.run(host=host, port=port)


def main():
    parser = argparse.ArgumentParser(description="Offline IDS dashboard pipeline")
    parser.add_argument("--dataset", default="NF-UNSW-NB15-v3.csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--skip-serve", action="store_true")
    args = parser.parse_args()

    report = build_dashboard_data(dataset=args.dataset, device=args.device)
    print(json.dumps(report, indent=2))

    if not args.skip_serve:
        serve_dashboard(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
