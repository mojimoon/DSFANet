from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC

from src import config
from src.data_loader import DataPreprocessor, extract_benign_samples
from src.drift_tester import DriftGenerator
from src.models import Autoencoder, DSFANet, LSTMClassifier
from src.models.ensemble import StackingEnsemble, UnificationLayer, VotingEnsemble, XGBoostStackingEnsemble
from src.runtime import resolve_device
from src.shap_analysis import analyze_ae_shap, analyze_dsfanet_shap, analyze_lstm_shap, train_autoencoder_model, train_lstm_model

DEFAULT_DATASETS = [
    "NF-UNSW-NB15-v3.csv",
    "NF-ToN-IoT-v3.csv",
    "NF-CICIDS2018-v3.csv",
]


def slug(text: str) -> str:
    return text.replace(".csv", "").replace(".", "_").replace("-", "_").replace(" ", "_")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_str_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def metric_row(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray | None = None) -> dict:
    if y_pred is None:
        y_pred = (y_prob >= 0.5).astype(int)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "ap": float(average_precision_score(y_true, y_prob)),
    }


def save_predictions(out_dir: Path, dataset_key: str, model_name: str, y_true: np.ndarray, y_prob: np.ndarray):
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
    x_s_train: np.ndarray,
    x_t_train: np.ndarray,
    y_train: np.ndarray,
    device: torch.device,
    epochs: int = 2,
):
    model = DSFANet(
        static_dim=x_s_train.shape[1],
        temporal_dim=x_t_train.shape[1],
        n_classes=config.NUM_CLASSES,
        device=str(device),
    )
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    batch_size = 128
    x_s_t = torch.tensor(x_s_train, dtype=torch.float32, device=device)
    x_t_t = torch.tensor(x_t_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.long, device=device)

    model.train()
    for _ in range(epochs):
        perm = torch.randperm(x_s_t.shape[0], device=device)
        for i in range(0, x_s_t.shape[0], batch_size):
            idx = perm[i : i + batch_size]
            bx_s, bx_t, by = x_s_t[idx], x_t_t[idx], y_t[idx]
            optimizer.zero_grad()
            logits = model(bx_s, bx_t)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def torch_probs(model: nn.Module, x_s: np.ndarray, x_t: np.ndarray, input_req: str, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        if input_req == "both":
            logits = model(
                torch.tensor(x_s, dtype=torch.float32, device=device),
                torch.tensor(x_t, dtype=torch.float32, device=device),
            )
        elif input_req == "temporal":
            logits = model(torch.tensor(x_t, dtype=torch.float32, device=device))
        else:
            logits = model(torch.tensor(x_s, dtype=torch.float32, device=device))

        return torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()


def ae_probs(model: Autoencoder, x_s: np.ndarray, ae_min: float, ae_max: float, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        recon = model(torch.tensor(x_s, dtype=torch.float32, device=device)).detach().cpu().numpy()
    err = np.mean((recon - x_s) ** 2, axis=1)
    denom = max(ae_max - ae_min, 1e-8)
    return np.clip((err - ae_min) / denom, 0.0, 1.0)


def get_model_probs_and_features(
    model_name: str,
    model,
    x_s: np.ndarray,
    x_t: np.ndarray,
    device: torch.device,
    ae_min: float | None = None,
    ae_max: float | None = None,
):
    if model_name == "AE":
        if ae_min is None or ae_max is None:
            with torch.no_grad():
                recon = model(torch.tensor(x_s, dtype=torch.float32, device=device)).detach().cpu().numpy()
            raw_err = np.mean((recon - x_s) ** 2, axis=1)
            ae_min = float(np.min(raw_err))
            ae_max = float(np.max(raw_err))
        probs = ae_probs(model, x_s, ae_min, ae_max, device)
        features = x_s
        return probs, features

    if model_name == "LSTM":
        probs = torch_probs(model, x_s, x_t, "temporal", device)
        with torch.no_grad():
            xt = torch.tensor(x_t, dtype=torch.float32, device=device).unsqueeze(-1)
            h_seq, _ = model.lstm(xt)
            features = h_seq[:, -1, :].detach().cpu().numpy()
        return probs, features

    probs = torch_probs(model, x_s, x_t, "both", device)
    if hasattr(model, "extract_features"):
        with torch.no_grad():
            features = model.extract_features(
                torch.tensor(x_s, dtype=torch.float32, device=device),
                torch.tensor(x_t, dtype=torch.float32, device=device),
            ).detach().cpu().numpy()
    else:
        features = np.concatenate([x_s, x_t], axis=1)
    return probs, features


def get_raw_score(model, model_type: str, input_req: str, x_s: np.ndarray, x_t: np.ndarray, device: torch.device) -> np.ndarray:
    if isinstance(model, nn.Module):
        if model_type == "classifier":
            return torch_probs(model, x_s, x_t, input_req=input_req, device=device)
        with torch.no_grad():
            recon = model(torch.tensor(x_s, dtype=torch.float32, device=device)).detach().cpu().numpy()
        return np.mean((recon - x_s) ** 2, axis=1)

    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_s)[:, 1]
    return model.predict(x_s)


def unify_scores(raw_scores: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    low = stats["min"]
    high = stats["max"]
    if high == low:
        high += 1e-6
    unified = (raw_scores - low) / (high - low)
    return np.clip(unified, 0.0, 1.0)


def train_and_eval_dataset(
    dataset: str,
    run_dir: Path,
    device: torch.device,
    max_train_samples: int,
    include_xgboost: bool,
):
    dataset_key = slug(dataset)
    ds_dir = ensure_dir(run_dir / dataset_key)
    model_dir = ensure_dir(ds_dir / "models")
    pred_dir = ensure_dir(ds_dir / "predictions")

    prep = DataPreprocessor(dataset)
    (x_s_train, x_t_train, y_train), (x_s_test, x_t_test, y_test) = prep.prepare_data()

    if max_train_samples > 0 and len(y_train) > max_train_samples:
        idx = np.random.RandomState(42).choice(len(y_train), size=max_train_samples, replace=False)
        x_s_train, x_t_train, y_train = x_s_train[idx], x_t_train[idx], y_train[idx]

    val_n = max(1000, int(0.2 * len(y_train)))
    x_s_val, x_t_val, y_val = x_s_train[:val_n], x_t_train[:val_n], y_train[:val_n]
    x_s_sub, x_t_sub, y_sub = x_s_train[val_n:], x_t_train[val_n:], y_train[val_n:]

    models = {}
    model_meta = {}

    rf = RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42)
    rf.fit(x_s_sub, y_sub)
    rf_path = model_dir / f"{dataset_key}_rf.joblib"
    joblib.dump(rf, rf_path)
    models["RandomForest"] = rf
    model_meta["RandomForest"] = {"path": str(rf_path), "model_type": "classifier", "input_req": "static", "kind": "sklearn"}

    svm = SVC(probability=True, kernel="rbf", max_iter=2000, random_state=42)
    svm.fit(x_s_sub, y_sub)
    svm_path = model_dir / f"{dataset_key}_svm.joblib"
    joblib.dump(svm, svm_path)
    models["SVM"] = svm
    model_meta["SVM"] = {"path": str(svm_path), "model_type": "classifier", "input_req": "static", "kind": "sklearn"}

    ae = train_autoencoder_model(x_s_sub, x_t_sub, y_sub, x_s_test, x_t_test, y_test, device=device, epochs=3)
    ae_path = ae.save_checkpoint(filename=f"{dataset_key}_ae.pt", checkpoint_dir=model_dir)
    with torch.no_grad():
        ae_train_recon = ae(torch.tensor(x_s_sub, dtype=torch.float32, device=device)).detach().cpu().numpy()
    ae_train_err = np.mean((ae_train_recon - x_s_sub) ** 2, axis=1)
    models["AE"] = ae
    model_meta["AE"] = {
        "path": str(ae_path),
        "model_type": "anomaly",
        "input_req": "static",
        "kind": "torch",
        "ae_min": float(np.min(ae_train_err)),
        "ae_max": float(np.max(ae_train_err)),
    }

    lstm = train_lstm_model(x_s_sub, x_t_sub, y_sub, x_s_test, x_t_test, y_test, device=device, epochs=3)
    lstm_path = lstm.save_checkpoint(filename=f"{dataset_key}_lstm.pt", checkpoint_dir=model_dir)
    models["LSTM"] = lstm
    model_meta["LSTM"] = {"path": str(lstm_path), "model_type": "classifier", "input_req": "temporal", "kind": "torch"}

    dsfa = train_dsfanet(x_s_sub, x_t_sub, y_sub, device=device, epochs=3)
    dsfa_path = dsfa.save_checkpoint(filename=f"{dataset_key}_dsfanet.pt", checkpoint_dir=model_dir)
    models["DSFANet"] = dsfa
    model_meta["DSFANet"] = {"path": str(dsfa_path), "model_type": "classifier", "input_req": "both", "kind": "torch"}

    rows = []
    prob_bank = {}

    for name, model in models.items():
        meta = model_meta[name]
        if name == "AE":
            probs = ae_probs(model, x_s_test, meta["ae_min"], meta["ae_max"], device=device)
        else:
            probs = get_raw_score(model, meta["model_type"], meta["input_req"], x_s_test, x_t_test, device)

        prob_bank[name] = probs
        save_predictions(pred_dir, dataset_key, name, y_test, probs)
        m = metric_row(y_test, probs)
        rows.append({"step": "baseline", "dataset": dataset, "model": name, **m})

    unifier = UnificationLayer()
    voting = VotingEnsemble(unifier=unifier, device=str(device))
    stacking = StackingEnsemble(unifier=unifier, device=str(device))

    base_configs = [
        ("RandomForest", models["RandomForest"], "classifier", "static"),
        ("SVM", models["SVM"], "classifier", "static"),
        ("AE", models["AE"], "anomaly", "static"),
        ("LSTM", models["LSTM"], "classifier", "temporal"),
        ("DSFANet", models["DSFANet"], "classifier", "both"),
    ]
    for cfg in base_configs:
        voting.add_model(*cfg)
        stacking.add_model(*cfg)

    voting.calibrate(x_s_val, x_t_val)
    stacking.fit_meta(x_s_val, x_t_val, y_val)

    voting_probs = voting.predict(x_s_test, x_t_test)
    stacking_probs = stacking.predict(x_s_test, x_t_test)
    save_predictions(pred_dir, dataset_key, "Voting", y_test, voting_probs)
    save_predictions(pred_dir, dataset_key, "Stacking", y_test, stacking_probs)
    rows.append({"step": "ensemble", "dataset": dataset, "model": "Voting", **metric_row(y_test, voting_probs)})
    rows.append({"step": "ensemble", "dataset": dataset, "model": "Stacking", **metric_row(y_test, stacking_probs)})

    ensemble_packages = []
    stack_pack = {
        "name": "Stacking",
        "model_order": [cfg[0] for cfg in base_configs],
        "model_info": {k: model_meta[k] for k in [cfg[0] for cfg in base_configs]},
        "unifier_stats": unifier.stats,
        "meta_learner": stacking.meta_learner,
    }
    stack_path = model_dir / f"{dataset_key}_stacking_pack.joblib"
    joblib.dump(stack_pack, stack_path)
    ensemble_packages.append({"name": "Stacking", "path": str(stack_path)})

    if include_xgboost:
        try:
            xgb_ens = XGBoostStackingEnsemble(unifier=unifier, device=str(device))
            for cfg in base_configs:
                xgb_ens.add_model(*cfg)
            xgb_ens.fit_meta(x_s_val, x_t_val, y_val)
            xgb_probs = xgb_ens.predict(x_s_test, x_t_test)
            save_predictions(pred_dir, dataset_key, "XGBoostStacking", y_test, xgb_probs)
            rows.append({"step": "ensemble", "dataset": dataset, "model": "XGBoostStacking", **metric_row(y_test, xgb_probs)})

            xgb_pack = {
                "name": "XGBoostStacking",
                "model_order": [cfg[0] for cfg in base_configs],
                "model_info": {k: model_meta[k] for k in [cfg[0] for cfg in base_configs]},
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
        "temporal_features": prep.used_temporal_cols,
        "models": model_meta,
        "ensembles": ensemble_packages,
    }
    registry_path = ds_dir / f"registry_{dataset_key}.json"
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    return rows, registry, (x_s_train, x_t_train, y_train, x_s_test, x_t_test, y_test)


def load_model_from_meta(model_name: str, meta: dict, device: torch.device):
    if meta["kind"] == "torch":
        if model_name == "AE":
            return Autoencoder.load_checkpoint(meta["path"], device=str(device))
        if model_name == "LSTM":
            return LSTMClassifier.load_checkpoint(meta["path"], device=str(device))
        if model_name == "DSFANet":
            return DSFANet.load_checkpoint(meta["path"], device=str(device))
        raise ValueError(f"Unsupported torch model: {model_name}")

    return joblib.load(meta["path"])


def predict_from_package(pack: dict, loaded_models: dict, x_s: np.ndarray, x_t: np.ndarray, device: torch.device):
    feats = []
    for name in pack["model_order"]:
        m = loaded_models[name]
        meta = pack["model_info"][name]
        if name == "AE":
            raw = get_raw_score(m, "anomaly", "static", x_s, x_t, device)
        else:
            raw = get_raw_score(m, meta["model_type"], meta["input_req"], x_s, x_t, device)
        unified = unify_scores(raw, pack["unifier_stats"][name])
        feats.append(unified)

    x_meta = np.column_stack(feats)
    return pack["meta_learner"].predict_proba(x_meta)[:, 1]


def step1_benchmarks(args, run_dir: Path, device: torch.device):
    all_rows = []
    registries = {}
    dataset_packs = {}

    for ds in args.datasets:
        rows, registry, data_pack = train_and_eval_dataset(
            dataset=ds,
            run_dir=run_dir,
            device=device,
            max_train_samples=args.max_train_samples,
            include_xgboost=args.include_xgboost,
        )
        all_rows.extend(rows)
        registries[registry["dataset_key"]] = registry
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
        plt.savefig(run_dir / f"chart_step1_ap_{args.run_id}.png")
    plt.close("all")

    return df, registries, dataset_packs


def step2_drift(args, run_dir: Path, device: torch.device, registry: dict, base_pack):
    x_s_train, x_t_train, y_train, x_s_test, x_t_test, y_test = base_pack
    drifter = DriftGenerator()

    loaded_models = {}
    for name, meta in registry["models"].items():
        loaded_models[name] = load_model_from_meta(name, meta, device)

    stack_pack_path = next((x["path"] for x in registry["ensembles"] if x["name"] == "Stacking"), None)
    xgb_pack_path = next((x["path"] for x in registry["ensembles"] if x["name"] == "XGBoostStacking"), None)
    stack_pack = joblib.load(stack_pack_path) if stack_pack_path else None
    xgb_pack = joblib.load(xgb_pack_path) if xgb_pack_path else None

    benign_s, benign_t = extract_benign_samples(args.base_dataset, max_samples=args.max_benign_for_attacks)

    drift_cases = {
        "clean": (x_s_test, x_t_test, y_test),
        "label_shift": drifter.simulate_label_shift(x_s_test, x_t_test, y_test, target_malicious_ratio=0.8),
        "corruption": (
            drifter.simulate_corruption(x_s_test, noise_type="gaussian", severity=0.1),
            drifter.simulate_corruption(x_t_test, noise_type="gaussian", severity=0.1),
            y_test,
        ),
    }

    for natural_ds in args.natural_datasets:
        try:
            n_s, n_t, n_y = drifter.load_natural_shift_data(natural_ds)
            if n_s.shape[1] == x_s_test.shape[1] and n_t.shape[1] == x_t_test.shape[1]:
                drift_cases[f"natural_{slug(natural_ds)}"] = (n_s, n_t, n_y)
        except Exception:
            continue

    for adv in ["fgsm", "pgd", "mimicry", "gdkde"]:
        sub_n = min(args.drift_subset_size, len(y_test))
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
        ae_meta = registry["models"]["AE"]
        ae_prob = ae_probs(loaded_models["AE"], dxs, ae_meta["ae_min"], ae_meta["ae_max"], device)
        lstm_prob = torch_probs(loaded_models["LSTM"], dxs, dxt, "temporal", device)
        dsfa_prob = torch_probs(loaded_models["DSFANet"], dxs, dxt, "both", device)

        for mname, probs in [("AE", ae_prob), ("LSTM", lstm_prob), ("DSFANet", dsfa_prob)]:
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


def select_indices_by_metric(
    metric: str,
    probs: np.ndarray,
    budget_ratio: float,
    features: np.ndarray | None = None,
):
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    probs = np.clip(probs, 0.0, 1.0)

    n = len(probs)
    k = max(1, int(round(n * budget_ratio)))

    if metric == "random":
        return np.random.choice(n, size=k, replace=False)

    if metric in ["uncertainty", "deep_gini"]:
        score = 1.0 - np.abs(probs - 0.5) * 2
    elif metric == "entropy":
        p = np.clip(probs, 1e-8, 1 - 1e-8)
        score = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    elif metric == "gd":
        if features is None:
            score = probs
        else:
            center = np.mean(features, axis=0, keepdims=True)
            score = np.linalg.norm(features - center, axis=1)
    elif metric == "ensemble_rank":
        p = np.clip(probs, 1e-8, 1 - 1e-8)
        uncertainty = 1.0 - np.abs(probs - 0.5) * 2
        entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        rank_u = np.argsort(np.argsort(uncertainty))
        rank_e = np.argsort(np.argsort(entropy))
        score = (rank_u + rank_e) / 2.0
    elif metric == "ensemble_hybrid":
        p = np.clip(probs, 1e-8, 1 - 1e-8)
        uncertainty = 1.0 - np.abs(probs - 0.5) * 2
        entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        if features is None:
            diversity = np.zeros_like(uncertainty)
        else:
            center = np.mean(features, axis=0, keepdims=True)
            diversity = np.linalg.norm(features - center, axis=1)
            diversity = diversity / (np.max(diversity) + 1e-8)
        score = 0.45 * uncertainty + 0.45 * (entropy / (np.max(entropy) + 1e-8)) + 0.10 * diversity
    else:
        score = probs

    return np.argsort(score)[::-1][:k]


def retrain_model_generic(
    model_name: str,
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
    device: torch.device,
):
    model = deepcopy(model)

    probs, features = get_model_probs_and_features(
        model_name=model_name,
        model=model,
        x_s=drift_s,
        x_t=drift_t,
        device=device,
    )

    selected = select_indices_by_metric(metric, probs, budget_ratio, features=features)
    id_count = int(max(1, round(len(selected) * id_ratio)))
    id_count = min(id_count, len(y_train))
    replay_idx = np.random.choice(len(y_train), size=id_count, replace=False)

    if model_name == "AE":
        retrain_s = np.concatenate([drift_s[selected], x_s_train[replay_idx]])
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

        p_after, _ = get_model_probs_and_features(model_name, model, drift_s, drift_t, device)
        return model, p_after

    retrain_s = np.concatenate([drift_s[selected], x_s_train[replay_idx]])
    retrain_t = np.concatenate([drift_t[selected], x_t_train[replay_idx]])
    retrain_y = np.concatenate([drift_y[selected], y_train[replay_idx]])

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()

    bs = 64
    for _ in range(3):
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
            optimizer.step()

    p, _ = get_model_probs_and_features(model_name, model, drift_s, drift_t, device)
    return model, p


def step3_retrain(args, run_dir: Path, device: torch.device, registry: dict, base_pack):
    x_s_train, x_t_train, y_train, x_s_test, x_t_test, y_test = base_pack

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

    drift_cases = {}
    for adv in ["pgd", "gdkde"]:
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
        drift_cases[adv] = (adv_s, adv_t, adv_y)

    retrain_dir = ensure_dir(run_dir / "retrain_models")
    rows = []
    best_models = {}

    metrics_list = parse_str_list(args.retrain_metrics)
    budgets = parse_float_list(args.retrain_budgets)
    id_ratios = parse_float_list(args.retrain_id_ratios)

    for model_name, model in models.items():
        for attack_name, (dxs, dxt, dy) in drift_cases.items():
            if model_name == "AE":
                ae_meta = registry["models"]["AE"]
                before_prob = ae_probs(model, dxs, ae_meta["ae_min"], ae_meta["ae_max"], device)
            elif model_name == "LSTM":
                before_prob = torch_probs(model, dxs, dxt, "temporal", device)
            else:
                before_prob = torch_probs(model, dxs, dxt, "both", device)
            before_acc = accuracy_score(dy, (before_prob >= 0.5).astype(int))

            best_gain = -1e9
            best_state = None
            best_tag = ""

            for metric in metrics_list:
                for budget in budgets:
                    for id_ratio in id_ratios:
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
                            device,
                        )
                        after_acc = accuracy_score(dy, (after_prob >= 0.5).astype(int))
                        gain = float(after_acc - before_acc)
                        row = {
                            "step": "retrain",
                            "dataset": args.base_dataset,
                            "model": model_name,
                            "attack": attack_name,
                            "selection_metric": metric,
                            "budget_ratio": budget,
                            "id_ratio": id_ratio,
                            "before_acc": float(before_acc),
                            "after_acc": float(after_acc),
                            "acc_gain": gain,
                        }
                        rows.append(row)

                        if gain > best_gain:
                            best_gain = gain
                            best_tag = f"{model_name}_{attack_name}_{metric}_b{budget:.2f}_id{id_ratio:.2f}"
                            if isinstance(retrained, nn.Module):
                                best_state = deepcopy(retrained.state_dict())
                            else:
                                best_state = deepcopy(retrained)

            if best_state is not None:
                if isinstance(model, nn.Module):
                    model_copy = deepcopy(model)
                    model_copy.load_state_dict(best_state)
                    filename = f"best_{slug(best_tag)}_{args.run_id}.pt"
                    saved = model_copy.save_checkpoint(filename=filename, checkpoint_dir=retrain_dir)
                else:
                    filename = retrain_dir / f"best_{slug(best_tag)}_{args.run_id}.joblib"
                    joblib.dump(best_state, filename)
                    saved = str(filename)
                best_models[f"{model_name}_{attack_name}"] = {
                    "path": saved,
                    "acc_gain": best_gain,
                    "attack": attack_name,
                    "model": model_name,
                }

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / f"summary_step3_retrain_{args.run_id}.csv", index=False)
    (run_dir / f"best_models_step3_{args.run_id}.json").write_text(json.dumps(best_models, indent=2), encoding="utf-8")
    return df, best_models


def step4_best_ensemble_shap(args, run_dir: Path, device: torch.device, base_pack, best_models: dict, registry: dict):
    x_s_train, x_t_train, y_train, x_s_test, x_t_test, y_test = base_pack
    val_n = max(1000, int(0.2 * len(y_train)))
    x_s_val, x_t_val, y_val = x_s_train[:val_n], x_t_train[:val_n], y_train[:val_n]

    best_ae_key = next((k for k in best_models if k.startswith("AE_")), None)
    best_lstm_key = next((k for k in best_models if k.startswith("LSTM_")), None)
    best_dsfa_key = next((k for k in best_models if k.startswith("DSFANet_")), None)

    ae_model = Autoencoder.load_checkpoint(best_models[best_ae_key]["path"], device=str(device)) if best_ae_key else load_model_from_meta("AE", registry["models"]["AE"], device)
    lstm_model = LSTMClassifier.load_checkpoint(best_models[best_lstm_key]["path"], device=str(device)) if best_lstm_key else load_model_from_meta("LSTM", registry["models"]["LSTM"], device)
    dsfa_model = DSFANet.load_checkpoint(best_models[best_dsfa_key]["path"], device=str(device)) if best_dsfa_key else load_model_from_meta("DSFANet", registry["models"]["DSFANet"], device)

    rf = load_model_from_meta("RandomForest", registry["models"]["RandomForest"], device)
    svm = load_model_from_meta("SVM", registry["models"]["SVM"], device)

    unifier = UnificationLayer()
    stacking = StackingEnsemble(unifier=unifier, device=str(device))
    stacking.add_model("RandomForest", rf, "classifier", "static")
    stacking.add_model("SVM", svm, "classifier", "static")
    stacking.add_model("AE", ae_model, "anomaly", "static")
    stacking.add_model("LSTM", lstm_model, "classifier", "temporal")
    stacking.add_model("DSFANet", dsfa_model, "classifier", "both")
    stacking.calibrate(x_s_val, x_t_val)
    stacking.fit_meta(x_s_val, x_t_val, y_val)
    stack_prob = stacking.predict(x_s_test, x_t_test)

    rows = [{"step": "best_ensemble", "dataset": args.base_dataset, "model": "Stacking_best", **metric_row(y_test, stack_prob)}]

    try:
        xgb = XGBoostStackingEnsemble(unifier=unifier, device=str(device))
        xgb.add_model("RandomForest", rf, "classifier", "static")
        xgb.add_model("SVM", svm, "classifier", "static")
        xgb.add_model("AE", ae_model, "anomaly", "static")
        xgb.add_model("LSTM", lstm_model, "classifier", "temporal")
        xgb.add_model("DSFANet", dsfa_model, "classifier", "both")
        xgb.calibrate(x_s_val, x_t_val)
        xgb.fit_meta(x_s_val, x_t_val, y_val)
        xgb_prob = xgb.predict(x_s_test, x_t_test)
        rows.append({"step": "best_ensemble", "dataset": args.base_dataset, "model": "XGBoostStacking_best", **metric_row(y_test, xgb_prob)})
    except Exception as ex:
        rows.append({"step": "best_ensemble", "dataset": args.base_dataset, "model": "XGBoostStacking_best", "acc": np.nan, "f1": np.nan, "precision": np.nan, "recall": np.nan, "ap": np.nan, "error": str(ex)})

    shap_dir = ensure_dir(run_dir / "shap_best_models")
    prep = DataPreprocessor(args.base_dataset)
    prep.prepare_data()
    s_features = prep.used_static_cols
    t_features = prep.used_temporal_cols

    shap_lstm = analyze_lstm_shap(lstm_model, x_t_test, t_features, out_dir=shap_dir)
    shap_ae = analyze_ae_shap(ae_model, x_s_test, s_features, out_dir=shap_dir)
    shap_ds = analyze_dsfanet_shap(dsfa_model, x_s_test, x_t_test, s_features, t_features, out_dir=shap_dir)
    (shap_dir / f"shap_best_summary_{args.run_id}.json").write_text(json.dumps({"LSTM": shap_lstm, "AE": shap_ae, "DSFANet": shap_ds}, indent=2), encoding="utf-8")

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / f"summary_step4_best_ensemble_shap_{args.run_id}.csv", index=False)
    return df


class DSFANetAblation(nn.Module):
    def __init__(self, static_dim: int, temporal_dim: int, n_classes: int, mode: str = "full"):
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


def step5_dsfanet_ablation(args, run_dir: Path, device: torch.device):
    prep = DataPreprocessor(args.base_dataset)
    (x_s_train, x_t_train, y_train), (x_s_test, x_t_test, y_test) = prep.prepare_data()

    val_n = max(1000, int(0.2 * len(y_train)))
    x_s_sub, x_t_sub, y_sub = x_s_train[val_n:], x_t_train[val_n:], y_train[val_n:]

    rows = []
    modes = ["full", "s_only", "t_only", "no_attn"]

    for mode in modes:
        model = DSFANetAblation(x_s_sub.shape[1], x_t_sub.shape[1], config.NUM_CLASSES, mode=mode).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        bs = 128
        for _ in range(3):
            perm = np.random.permutation(len(y_sub))
            for i in range(0, len(y_sub), bs):
                idx = perm[i : i + bs]
                xs = torch.tensor(x_s_sub[idx], dtype=torch.float32, device=device)
                xt = torch.tensor(x_t_sub[idx], dtype=torch.float32, device=device)
                yy = torch.tensor(y_sub[idx], dtype=torch.long, device=device)
                optimizer.zero_grad()
                logits = model(xs, xt)
                loss = criterion(logits, yy)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(
                torch.tensor(x_s_test, dtype=torch.float32, device=device),
                torch.tensor(x_t_test, dtype=torch.float32, device=device),
            )
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

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


def step6_export_for_web(run_dir: Path, args):
    summary_files = sorted(run_dir.glob("summary_step*.csv"))
    payload = {
        "run_id": args.run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "summary_files": [str(p) for p in summary_files],
    }

    for csv_path in summary_files:
        key = csv_path.stem
        try:
            payload[key] = pd.read_csv(csv_path).head(500).to_dict(orient="records")
        except Exception:
            payload[key] = []

    out_www = ensure_dir(Path("out") / "www")
    latest_json = out_www / "experiments_latest.json"
    latest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run the full experiment pipeline")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--steps", default="1,2,3,4,5,6", help="Comma-separated steps")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--base-dataset", default="NF-UNSW-NB15-v3.csv")
    parser.add_argument("--natural-datasets", default="NF-ToN-IoT-v3.csv,NF-CICIDS2018-v3.csv")
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--max-benign-for-attacks", type=int, default=5000)
    parser.add_argument("--drift-subset-size", type=int, default=3000)
    parser.add_argument("--retrain-metrics", default="random,uncertainty,entropy,gd,ensemble_rank,ensemble_hybrid")
    parser.add_argument("--retrain-budgets", default="0.1,0.2,0.3")
    parser.add_argument("--retrain-id-ratios", default="0.25,0.5,0.75")
    parser.add_argument("--include-xgboost", action="store_true")
    args = parser.parse_args()

    args.datasets = parse_str_list(args.datasets)
    args.natural_datasets = parse_str_list(args.natural_datasets)
    steps = set(parse_str_list(args.steps))

    device = resolve_device(args.device)
    run_dir = ensure_dir(Path("out") / "experiments" / args.run_id)

    summary_rows = []
    registries = {}
    dataset_packs = {}

    if "1" in steps:
        df1, registries, dataset_packs = step1_benchmarks(args, run_dir, device)
        summary_rows.append({"step": 1, "rows": len(df1)})

    base_key = slug(args.base_dataset)
    needs_base = any(x in steps for x in ["2", "3", "4", "5"])

    if needs_base:
        if not registries:
            reg_path = run_dir / base_key / f"registry_{base_key}.json"
            if reg_path.exists():
                registries[base_key] = json.loads(reg_path.read_text(encoding="utf-8"))

        if base_key not in dataset_packs:
            prep = DataPreprocessor(args.base_dataset)
            (x_s_train, x_t_train, y_train), (x_s_test, x_t_test, y_test) = prep.prepare_data()
            dataset_packs[base_key] = (x_s_train, x_t_train, y_train, x_s_test, x_t_test, y_test)

    best_models = {}

    if "2" in steps and base_key in registries:
        df2 = step2_drift(args, run_dir, device, registries[base_key], dataset_packs[base_key])
        summary_rows.append({"step": 2, "rows": len(df2)})

    if "3" in steps and base_key in registries:
        df3, best_models = step3_retrain(args, run_dir, device, registries[base_key], dataset_packs[base_key])
        summary_rows.append({"step": 3, "rows": len(df3)})

    if "4" in steps and base_key in registries:
        if not best_models:
            best_path = run_dir / f"best_models_step3_{args.run_id}.json"
            if best_path.exists():
                best_models = json.loads(best_path.read_text(encoding="utf-8"))
        df4 = step4_best_ensemble_shap(args, run_dir, device, dataset_packs[base_key], best_models, registries[base_key])
        summary_rows.append({"step": 4, "rows": len(df4)})

    if "5" in steps:
        df5 = step5_dsfanet_ablation(args, run_dir, device)
        summary_rows.append({"step": 5, "rows": len(df5)})

    if "6" in steps:
        step6_export_for_web(run_dir, args)
        summary_rows.append({"step": 6, "rows": 1})

    pd.DataFrame(summary_rows).to_csv(run_dir / f"run_overview_{args.run_id}.csv", index=False)
    print(f"Done. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
