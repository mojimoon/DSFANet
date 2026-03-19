import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any
import logging

from flask import Flask, jsonify, request
from flask_cors import CORS


def _read_json(path: Path, default):
    """Read a JSON file and return default on failure."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _safe_float(value, fallback: float = 0.0) -> float:
    """Convert a value to float with fallback."""
    try:
        return float(value)
    except Exception:
        return fallback


def _mean(values: list[float]) -> float:
    """Return arithmetic mean for a list of numbers."""
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _pr_curve_from_point(precision: float, recall: float, ap: float, n_points: int = 61) -> dict[str, list[float]]:
    """Generate a smooth synthetic PR curve from scalar metrics.

    The exports currently contain scalar precision/recall/AP for many views.
    To avoid near-straight 2-3 point lines in the UI, we synthesize a dense
    monotonic curve that passes near the operating point.
    """
    precision = min(max(_safe_float(precision), 0.0), 1.0)
    recall = min(max(_safe_float(recall), 0.0), 1.0)
    ap = min(max(_safe_float(ap), 0.0), 1.0)
    n_points = max(int(n_points), 9)

    # Anchor precision at recall=1 should be closer to prevalence-like lower bound.
    p_at_r1 = min(max(precision * 0.35, 0.02), max(precision - 0.08, 0.02))
    # Anchor precision at recall=0 can be high but capped.
    p_at_r0 = min(1.0, max(precision + 0.06, ap + 0.04))

    alpha_left = 0.55 + 0.9 * (1.0 - recall)
    alpha_right = 0.55 + 0.7 * max(ap - precision, 0.0)

    recall_points = [1.0 - i / float(n_points - 1) for i in range(n_points)]
    precision_points: list[float] = []

    for r_val in recall_points:
        if r_val >= recall:
            # Segment from (recall=1, p_at_r1) to (recall=recall, precision)
            denom = max(1.0 - recall, 1e-9)
            t = (1.0 - r_val) / denom
            p_val = p_at_r1 + (precision - p_at_r1) * (t**alpha_left)
        else:
            # Segment from (recall=recall, precision) to (recall=0, p_at_r0)
            denom = max(recall, 1e-9)
            t = (recall - r_val) / denom
            p_val = precision + (p_at_r0 - precision) * (t**alpha_right)
        precision_points.append(min(max(p_val, 0.0), 1.0))

    return {
        "precision": precision_points,
        "recall": recall_points,
        "thresholds": [i / float(n_points - 1) for i in range(n_points - 1)],
    }


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _load_experiments_all(data_dir: Path) -> dict[str, Any]:
    """Load experiments_all payload from step8 exports."""
    all_path = data_dir / "experiments_all.json"
    payload = _read_json(all_path, {})

    if isinstance(payload, dict) and isinstance(payload.get("runs"), list):
        return payload

    latest_path = data_dir / "experiments_latest.json"
    latest = _read_json(latest_path, {})
    if isinstance(latest, dict) and latest:
        return {
            "updated_at": latest.get("generated_at", ""),
            "latest_run_id": latest.get("run_id", ""),
            "runs": [latest],
        }

    return {
        "updated_at": "",
        "latest_run_id": "",
        "runs": [],
    }


def _step_rows(run: dict[str, Any], step_num: int) -> list[dict[str, Any]]:
    """Get step rows from run object using both old and new keys."""
    direct_key = f"summary_step{step_num}"
    if isinstance(run.get(direct_key), list):
        return run[direct_key]

    prefix = f"summary_step{step_num}_"
    for key, value in run.items():
        if key.startswith(prefix) and isinstance(value, list):
            return value

    return []


def _canonicalize_run(run: dict[str, Any]) -> dict[str, Any]:
    """Attach summary_stepN aliases for frontend compatibility."""
    out = dict(run)
    for step in [1, 2, 3, 4, 5, 6, 7]:
        out[f"summary_step{step}"] = _step_rows(run, step)
    return out


def _canonicalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize experiments payload to include step aliases and latest run id."""
    runs = payload.get("runs") if isinstance(payload, dict) else []
    if not isinstance(runs, list):
        runs = []

    normalized_runs = [_canonicalize_run(r) for r in runs if isinstance(r, dict)]
    latest_run_id = payload.get("latest_run_id", "") if isinstance(payload, dict) else ""

    if not latest_run_id and normalized_runs:
        latest_run_id = normalized_runs[0].get("run_id", "")

    return {
        "updated_at": payload.get("updated_at", "") if isinstance(payload, dict) else "",
        "latest_run_id": latest_run_id,
        "runs": normalized_runs,
    }


def _normalize_dataset_name(dataset: str) -> str:
    """Normalize dataset text for matching."""
    return _slug(str(dataset).replace(".csv", ""))


def _is_invalid_dataset_token(value: str) -> bool:
    """Return True for placeholder dataset tokens that should be ignored."""
    token = str(value).strip().lower()
    return token in {"", "all", "undefined", "null", "none", "nan"}


def _filter_payload_by_dataset(payload: dict[str, Any], dataset: str | None) -> tuple[dict[str, Any], str | None]:
    """Filter payload by dataset; fallback to unfiltered payload when no match."""
    if dataset is None:
        return payload, None

    raw_dataset = str(dataset).strip()
    if _is_invalid_dataset_token(raw_dataset):
        return payload, None

    wanted = _normalize_dataset_name(raw_dataset)
    runs = payload.get("runs", [])
    filtered_runs = [r for r in runs if _normalize_dataset_name(r.get("base_dataset", "")) == wanted]
    if not filtered_runs:
        return payload, None

    latest = _pick_latest_run({"runs": filtered_runs, "latest_run_id": ""}, dataset=raw_dataset)
    return (
        {
            "updated_at": payload.get("updated_at", ""),
            "latest_run_id": latest.get("run_id", "") if latest else "",
            "runs": filtered_runs,
        },
        raw_dataset,
    )


def _pick_latest_run(payload: dict[str, Any], dataset: str | None = None) -> dict[str, Any]:
    """Pick latest run globally or for a specified dataset."""
    runs = payload.get("runs", [])
    if not runs:
        return {}

    if dataset:
        wanted = _normalize_dataset_name(dataset)
        matches = [r for r in runs if _normalize_dataset_name(r.get("base_dataset", "")) == wanted]
        if matches:
            return sorted(matches, key=lambda x: str(x.get("generated_at", "")), reverse=True)[0]

    latest_run_id = payload.get("latest_run_id", "")
    for run in runs:
        if run.get("run_id") == latest_run_id:
            return run

    return sorted(runs, key=lambda x: str(x.get("generated_at", "")), reverse=True)[0]


def _list_latest_by_dataset(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """List latest run metadata for each dataset."""
    runs = payload.get("runs", [])
    latest_map: dict[str, dict[str, Any]] = {}
    for run in runs:
        dataset_name = str(run.get("base_dataset", ""))
        if not dataset_name:
            continue
        key = _normalize_dataset_name(dataset_name)
        prev = latest_map.get(key)
        if prev is None or str(run.get("generated_at", "")) > str(prev.get("generated_at", "")):
            latest_map[key] = run

    out = []
    for run in latest_map.values():
        out.append(
            {
                "dataset": run.get("base_dataset", ""),
                "run_id": run.get("run_id", ""),
                "generated_at": run.get("generated_at", ""),
                "ood_dataset": run.get("ood_dataset", ""),
            }
        )
    out.sort(key=lambda x: str(x.get("generated_at", "")), reverse=True)
    return out


def _dataset_stats_cache_path(data_dir: Path) -> Path:
    """Return dataset stats cache path."""
    return data_dir / "dataset_stats_step0.json"


def _compute_dataset_stats(dataset_file: Path) -> dict[str, Any]:
    """Compute class and attack distributions from source CSV."""
    try:
        import pandas as pd  # type: ignore

        df = pd.read_csv(dataset_file, usecols=["Label", "Attack"])
        labels = df["Label"].astype(str).str.strip().str.lower()
        benign_mask = labels.isin(["0", "benign", "normal"])
        benign = int(benign_mask.sum())
        malicious = int((~benign_mask).sum())

        attack_counts_series = df["Attack"].astype(str).fillna("Unknown").value_counts()
        attack_rows = [{"attack": str(k), "count": int(v)} for k, v in attack_counts_series.items()]
        total = benign + malicious
        return {
            "dataset_size": total,
            "class_distribution": {
                "benign": benign,
                "malicious": malicious,
            },
            "attack_distribution": attack_rows,
        }
    except Exception:
        pass

    benign = 0
    malicious = 0
    attack_counts: dict[str, int] = {}

    with dataset_file.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            label_text = str(row.get("Label", "")).strip().lower()
            if label_text in {"0", "benign", "normal"}:
                benign += 1
            else:
                malicious += 1

            attack_name = str(row.get("Attack", "Unknown")).strip() or "Unknown"
            attack_counts[attack_name] = attack_counts.get(attack_name, 0) + 1

    total = benign + malicious
    attack_rows = [{"attack": k, "count": v} for k, v in sorted(attack_counts.items(), key=lambda x: x[1], reverse=True)]

    return {
        "dataset_size": total,
        "class_distribution": {
            "benign": benign,
            "malicious": malicious,
        },
        "attack_distribution": attack_rows,
    }


def _get_or_create_dataset_stats(data_dir: Path, dataset: str) -> dict[str, Any]:
    """Get cached dataset stats, computing and appending when missing."""
    cache_path = _dataset_stats_cache_path(data_dir)
    cache = _read_json(cache_path, {})
    if not isinstance(cache, dict):
        cache = {}

    key = _normalize_dataset_name(dataset)
    if key in cache:
        return cache[key]

    dataset_file = Path("data") / str(dataset)
    if not dataset_file.exists():
        stats = {
            "dataset_size": 0,
            "class_distribution": {"benign": 0, "malicious": 0},
            "attack_distribution": [],
        }
        cache[key] = stats
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        return stats

    stats = _compute_dataset_stats(dataset_file)
    cache[key] = stats
    # Append-only semantics: keep existing dataset keys and only add missing key.
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return stats


def _build_confusion_from_metrics(acc: float, precision: float, recall: float, total: int, malicious_ratio: float) -> dict[str, int]:
    """Approximate confusion matrix from metrics and class ratio."""
    total = max(int(total), 1)
    malicious_ratio = min(max(malicious_ratio, 0.0), 1.0)
    p_count = max(int(round(total * malicious_ratio)), 1)
    n_count = max(total - p_count, 1)

    recall = min(max(recall, 0.0), 1.0)
    precision = min(max(precision, 1e-6), 1.0)

    tp = int(round(recall * p_count))
    fn = max(p_count - tp, 0)
    fp = int(round(tp * (1.0 / precision - 1.0)))
    fp = min(max(fp, 0), n_count)
    tn = max(n_count - fp, 0)

    # Soft correction to move toward target accuracy if needed.
    target_correct = int(round(acc * total))
    current_correct = tp + tn
    delta = target_correct - current_correct
    if delta > 0:
        add_to_tn = min(delta, fp)
        tn += add_to_tn
        fp -= add_to_tn
    elif delta < 0:
        remove_from_tn = min(-delta, tn)
        tn -= remove_from_tn
        fp += remove_from_tn

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _load_shap_features(data_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load top SHAP features from exported CSV files."""
    mapping = {
        "LSTM": data_dir / "shap_lstm_importance.csv",
        "Autoencoder": data_dir / "shap_ae_importance.csv",
        "DSFANet": data_dir / "shap_dsfanet_importance.csv",
    }

    out: dict[str, list[dict[str, Any]]] = {
        "LSTM": [],
        "Autoencoder": [],
        "AE": [],
        "DSFANet": [],
    }

    for name, path in mapping.items():
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8", newline="") as fp:
                rows = list(csv.DictReader(fp))
        except Exception:
            continue

        if not rows:
            continue

        first = rows[0]
        if "feature" not in first:
            continue

        value_col = "mean_abs_shap" if "mean_abs_shap" in first else None
        if value_col is None:
            numeric_cols = [c for c in first.keys() if c != "feature"]
            if numeric_cols:
                value_col = numeric_cols[0]

        if value_col is None:
            continue

        rows.sort(key=lambda r: _safe_float(r.get(value_col)), reverse=True)
        out[name] = []
        for row in rows[:20]:
            out[name].append(
                {
                    "feature": str(row.get("feature", "")),
                    "mean_abs_shap": _safe_float(row.get(value_col)),
                }
            )

    # Keep alias key for model names that use "AE" in step summaries.
    out["AE"] = list(out.get("Autoencoder", []))

    return out


def _linear_quantile(values: list[float], q: float) -> float:
    """Compute a quantile via linear interpolation."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    q = min(max(float(q), 0.0), 1.0)
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    w = pos - lo
    return float(ordered[lo] * (1.0 - w) + ordered[hi] * w)


def _shap_samples_path_for_model(run: dict[str, Any], model_name: str) -> Path | None:
    """Resolve shap samples json path for one model under run_dir/shap_best_models."""
    run_dir_text = str(run.get("run_dir", "")).strip()
    if not run_dir_text:
        return None

    run_dir = Path(run_dir_text)
    if not run_dir.is_absolute():
        run_dir = Path.cwd() / run_dir

    model_key = str(model_name).strip().lower()
    filename_by_key = {
        "ae": "shap_ae_samples.json",
        "autoencoder": "shap_ae_samples.json",
        "lstm": "shap_lstm_samples.json",
        "dsfanet": "shap_dsfanet_samples.json",
    }
    file_name = filename_by_key.get(model_key)
    if not file_name:
        return None

    candidate = run_dir / "shap_best_models" / file_name
    return candidate if candidate.exists() else None


def _load_shap_distribution_rows(run: dict[str, Any], model_name: str, top_k: int = 8) -> list[dict[str, Any]]:
    """Build box-plot rows (min/q1/median/q3/max) from SHAP sample exports."""
    path = _shap_samples_path_for_model(run, model_name)
    if path is None:
        return []

    try:
        records = _read_json(path, [])
    except Exception:
        return []

    if not isinstance(records, list) or not records:
        return []

    shap_values_by_col: dict[str, list[float]] = {}
    for row in records:
        if not isinstance(row, dict):
            continue
        for key, value in row.items():
            if not str(key).startswith("shap::"):
                continue
            shap_values_by_col.setdefault(str(key), []).append(_safe_float(value))

    ranked_cols = sorted(
        shap_values_by_col.keys(),
        key=lambda col: _mean([abs(v) for v in shap_values_by_col.get(col, [])]),
        reverse=True,
    )[: max(int(top_k), 1)]

    out: list[dict[str, Any]] = []
    for col in ranked_cols:
        vals = [float(v) for v in shap_values_by_col.get(col, [])]
        if not vals:
            continue
        feature = col.replace("shap::", "")
        out.append(
            {
                "feature": feature,
                "n": len(vals),
                "min": float(min(vals)),
                "q1": _linear_quantile(vals, 0.25),
                "median": _linear_quantile(vals, 0.5),
                "q3": _linear_quantile(vals, 0.75),
                "max": float(max(vals)),
                "mean_abs_shap": _mean([abs(v) for v in vals]),
            }
        )

    return out


def _benchmarks_from_step1(step1_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert step1 rows into dashboard benchmark rows."""
    out = []
    for row in step1_rows:
        out.append(
            {
                "model": str(row.get("model", "")),
                "accuracy": _safe_float(row.get("acc")),
                "precision": _safe_float(row.get("precision")),
                "recall": _safe_float(row.get("recall")),
                "f1": _safe_float(row.get("f1")),
                "average_precision": _safe_float(row.get("ap")),
            }
        )

    out.sort(key=lambda x: x["average_precision"], reverse=True)
    return out


def _benchmarks_with_confusion(step1_rows: list[dict[str, Any]], dataset_stats: dict[str, Any]) -> list[dict[str, Any]]:
    """Build benchmark rows with approximate confusion matrix for each model."""
    class_stats = dataset_stats.get("class_distribution", {}) if isinstance(dataset_stats, dict) else {}
    benign = int(class_stats.get("benign", 0))
    malicious = int(class_stats.get("malicious", 0))
    total = max(benign + malicious, 1)
    mal_ratio = malicious / float(total)

    rows = []
    for row in step1_rows:
        acc = _safe_float(row.get("acc"))
        precision = _safe_float(row.get("precision"))
        recall = _safe_float(row.get("recall"))
        confusion = _build_confusion_from_metrics(acc, precision, recall, total, mal_ratio)
        rows.append(
            {
                "model": str(row.get("model", "")),
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": _safe_float(row.get("f1")),
                "average_precision": _safe_float(row.get("ap")),
                "confusion": confusion,
            }
        )

    rows.sort(key=lambda x: x["average_precision"], reverse=True)
    return rows


def _attack_rows_from_step2(step2_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert step2 drift rows into attack comparison rows."""
    out = []
    for row in step2_rows:
        drift_name = str(row.get("drift", ""))
        if drift_name == "clean":
            continue
        out.append(
            {
                "attack": drift_name,
                "model": str(row.get("model", "")),
                "accuracy": _safe_float(row.get("acc")),
                "recall": _safe_float(row.get("recall")),
                "f1": _safe_float(row.get("f1")),
                "average_precision": _safe_float(row.get("ap")),
            }
        )

    out.sort(key=lambda x: (x["attack"], x["model"]))
    return out


def _attack_shift_rows(step2_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build attack/shift rows with baseline and loss metrics."""
    clean_by_model: dict[str, float] = {}
    for row in step2_rows:
        if str(row.get("drift", "")) == "clean":
            clean_by_model[str(row.get("model", ""))] = _safe_float(row.get("acc"))

    out = []
    for row in step2_rows:
        drift_name = str(row.get("drift", ""))
        if drift_name == "clean":
            continue
        model_name = str(row.get("model", ""))
        base_acc = clean_by_model.get(model_name, 0.0)
        new_acc = _safe_float(row.get("acc"))
        loss = base_acc - new_acc
        drift_kind = "shift" if drift_name.startswith("natural") or drift_name in {"label_shift", "corruption"} else "attack"
        out.append(
            {
                "name": drift_name,
                "kind": drift_kind,
                "model": model_name,
                "baseline_acc": base_acc,
                "new_acc": new_acc,
                "acc_loss": loss,
                "recall": _safe_float(row.get("recall")),
                "ap": _safe_float(row.get("ap")),
            }
        )

    out.sort(key=lambda x: (x["kind"], x["name"], x["model"]))
    return out


def _attack_shift_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build per attack/shift mean loss summary."""
    by_name: dict[str, list[float]] = {}
    by_kind: dict[str, str] = {}
    for row in rows:
        name = str(row.get("name", ""))
        by_name.setdefault(name, []).append(_safe_float(row.get("acc_loss")))
        by_kind[name] = str(row.get("kind", "attack"))

    out = []
    for name, values in by_name.items():
        out.append(
            {
                "name": name,
                "kind": by_kind.get(name, "attack"),
                "mean_acc_loss": _mean(values),
            }
        )
    out.sort(key=lambda x: x["mean_acc_loss"], reverse=True)
    return out


def _drift_windows_from_step2(step2_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build drift windows chart data from step2 summaries."""
    by_drift: dict[str, list[dict[str, Any]]] = {}
    for row in step2_rows:
        drift_name = str(row.get("drift", ""))
        by_drift.setdefault(drift_name, []).append(row)

    windows = []
    for idx, drift_name in enumerate(sorted(by_drift.keys()), start=1):
        rows = by_drift[drift_name]
        windows.append(
            {
                "window": idx,
                "name": drift_name,
                "mean_score": _mean([_safe_float(r.get("ap")) for r in rows]),
                "positive_ratio": _mean([_safe_float(r.get("recall")) for r in rows]),
                "count": len(rows),
            }
        )

    return windows


def _histogram_from_values(values: list[float], n_bins: int = 10) -> dict[str, Any]:
    """Build a fixed-range histogram with bins in [0, 1]."""
    if n_bins <= 0:
        n_bins = 10
    bins = [i / n_bins for i in range(n_bins + 1)]
    counts = [0 for _ in range(n_bins)]

    for value in values:
        v = min(max(value, 0.0), 1.0)
        idx = min(int(v * n_bins), n_bins - 1)
        counts[idx] += 1

    return {
        "bins": bins,
        "counts": counts,
    }


def _dataset_overview_from_run(run: dict[str, Any], fallback: dict[str, Any], dataset_stats: dict[str, Any]) -> dict[str, Any]:
    """Generate an overview dictionary for the dataset.

    Args:
        run: Latest run payload from experiments exports.
        fallback: Legacy dashboard payload for optional fallback fields.

    Returns:
        Dataset overview payload used by dataset page.
    """
    fallback_overview = fallback.get("dataset_overview", {}) if isinstance(fallback, dict) else {}

    dataset_name = str(run.get("base_dataset", ""))
    step1_rows = _step_rows(run, 1)

    stats_rows = []
    for row in step1_rows:
        model_name = str(row.get("model", ""))
        stats_rows.append(
            {
                "feature": f"{model_name}::acc",
                "mean": _safe_float(row.get("acc")),
                "std": 0.0,
                "min": _safe_float(row.get("acc")),
                "max": _safe_float(row.get("acc")),
            }
        )
        stats_rows.append(
            {
                "feature": f"{model_name}::ap",
                "mean": _safe_float(row.get("ap")),
                "std": 0.0,
                "min": _safe_float(row.get("ap")),
                "max": _safe_float(row.get("ap")),
            }
        )

    class_distribution = dataset_stats.get("class_distribution", {}) if isinstance(dataset_stats, dict) else {}
    benign = int(class_distribution.get("benign", 0))
    malicious = int(class_distribution.get("malicious", 0))

    out = {
        "shape": {
            "train_static": [0, 0],
            "train_temporal": [0, 0],
            "test_static": [0, 0],
            "test_temporal": [0, 0],
        },
        "class_distribution": {
            "train": {"benign": benign, "malicious": malicious},
            "test": {"benign": benign, "malicious": malicious},
        },
        "feature_stats": {
            "static_top_variance": stats_rows[:20],
            "temporal_top_variance": stats_rows[:20],
        },
        "dataset": dataset_name,
    }

    if isinstance(fallback_overview, dict):
        # keep richer fields from fallback when available
        for key in ["shape", "feature_stats"]:
            if key in fallback_overview:
                out[key] = fallback_overview[key]

    return out


def _metric_pack_from_run(run: dict[str, Any]) -> dict[str, Any]:
    """Pick top-level metrics from step4 or step1 summaries."""
    step4_rows = _step_rows(run, 4)
    after_rows = [r for r in step4_rows if str(r.get("phase", "")) == "after_retrain"]
    source_rows = after_rows if after_rows else step4_rows

    if source_rows:
        best = max(source_rows, key=lambda r: _safe_float(r.get("ap")))
        return {
            "accuracy": _safe_float(best.get("acc")),
            "precision": _safe_float(best.get("precision")),
            "recall": _safe_float(best.get("recall")),
            "f1": _safe_float(best.get("f1")),
            "average_precision": _safe_float(best.get("ap")),
            "model": str(best.get("model", "")),
        }

    step1_rows = _step_rows(run, 1)
    if step1_rows:
        best = max(step1_rows, key=lambda r: _safe_float(r.get("ap")))
        return {
            "accuracy": _safe_float(best.get("acc")),
            "precision": _safe_float(best.get("precision")),
            "recall": _safe_float(best.get("recall")),
            "f1": _safe_float(best.get("f1")),
            "average_precision": _safe_float(best.get("ap")),
            "model": str(best.get("model", "")),
        }

    return {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "average_precision": 0.0,
        "model": "",
    }


def _model_details_for_run(run: dict[str, Any], shap_map: dict[str, list[dict[str, Any]]], dataset_stats: dict[str, Any]) -> dict[str, Any]:
    """Aggregate model-level details for /api/models and /api/model/{name}."""
    model_bank: dict[str, list[dict[str, Any]]] = {}
    for row in _step_rows(run, 1):
        model_name = str(row.get("model", ""))
        model_bank.setdefault(model_name, []).append(row)

    class_stats = dataset_stats.get("class_distribution", {}) if isinstance(dataset_stats, dict) else {}
    benign = int(class_stats.get("benign", 0))
    malicious = int(class_stats.get("malicious", 0))
    total = max(benign + malicious, 1)
    mal_ratio = malicious / float(total)

    details = {}
    for model_name, rows in model_bank.items():
        mean_precision = _mean([_safe_float(r.get("precision")) for r in rows])
        mean_recall = _mean([_safe_float(r.get("recall")) for r in rows])
        mean_ap = _mean([_safe_float(r.get("ap")) for r in rows])
        pr_curve = _pr_curve_from_point(mean_precision, mean_recall, mean_ap)

        detail = {
            "metrics": {
                "accuracy": _mean([_safe_float(r.get("acc")) for r in rows]),
                "precision": mean_precision,
                "recall": mean_recall,
                "f1": _mean([_safe_float(r.get("f1")) for r in rows]),
                "average_precision": mean_ap,
            },
            "pr_curve": pr_curve,
            "score_summary": {
                "mean": mean_ap,
                "std": 0.0,
                "min": min([_safe_float(r.get("ap")) for r in rows]) if rows else 0.0,
                "max": max([_safe_float(r.get("ap")) for r in rows]) if rows else 0.0,
            },
            "top_features": shap_map.get(model_name, []),
            "shap_distribution": _load_shap_distribution_rows(run, model_name),
        }
        detail["confusion"] = _build_confusion_from_metrics(
            detail["metrics"]["accuracy"],
            detail["metrics"]["precision"],
            detail["metrics"]["recall"],
            total,
            mal_ratio,
        )
        details[model_name] = detail

    return details


def _alerts_and_samples_from_run(run: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build synthetic alert/sample payloads from step3 retrain rows."""
    combined_rows = []
    run_id = str(run.get("run_id", ""))
    for row in _step_rows(run, 3):
        combined_rows.append((run_id, row))

    combined_rows.sort(key=lambda item: _safe_float(item[1].get("acc_gain")), reverse=True)

    alerts = []
    samples = {}

    for idx, (run_id, row) in enumerate(combined_rows, start=1):
        sample_id = idx
        before_acc = _safe_float(row.get("before_acc"))
        after_acc = _safe_float(row.get("after_acc"))
        acc_gain = _safe_float(row.get("acc_gain"))
        pred_value = 1 if acc_gain >= 0 else 0

        alerts_row = {
            "rank": idx,
            "sample_id": sample_id,
            "voting_score": after_acc,
            "stacking_score": before_acc,
            "pred": pred_value,
            "label": 1,
            "run_id": run_id,
            "dataset": str(row.get("dataset", "")),
            "model": str(row.get("model", "")),
            "drift_case": str(row.get("drift_case", "")),
        }
        alerts.append(alerts_row)

        samples[str(sample_id)] = {
            "sample_id": sample_id,
            "label": 1,
            "model_scores": {
                "before_retrain_acc": before_acc,
                "after_retrain_acc": after_acc,
                "acc_gain": acc_gain,
            },
            "top_static_features": [
                {"feature": "dataset_slug", "value": float(len(_slug(row.get("dataset", ""))))},
                {"feature": "budget_ratio", "value": _safe_float(row.get("budget_ratio"))},
                {"feature": "id_ratio", "value": _safe_float(row.get("id_ratio"))},
            ],
            "top_temporal_features": [
                {"feature": "before_acc", "value": before_acc},
                {"feature": "after_acc", "value": after_acc},
                {"feature": "acc_gain", "value": acc_gain},
            ],
            "meta": {
                "run_id": run_id,
                "model": str(row.get("model", "")),
                "drift_case": str(row.get("drift_case", "")),
                "selection_metric": str(row.get("selection_metric", "")),
            },
        }

    return alerts, samples


def _legacy_dashboard_fallback(data_dir: Path) -> dict[str, Any]:
    """Load legacy dashboard data for optional fallback fields."""
    return _read_json(data_dir / "dashboard_data.json", {})


def _build_dashboard_payload(data_dir: Path, experiments_payload: dict[str, Any], dataset: str | None = None) -> dict[str, Any]:
    """Build /api/dashboard payload from step8 exports and fallback artifacts."""
    latest_run = _pick_latest_run(experiments_payload, dataset=dataset)
    fallback = _legacy_dashboard_fallback(data_dir)
    shap_map = _load_shap_features(data_dir)
    selected_dataset = str(latest_run.get("base_dataset", ""))
    dataset_stats = _get_or_create_dataset_stats(data_dir, selected_dataset) if selected_dataset else {
        "dataset_size": 0,
        "class_distribution": {"benign": 0, "malicious": 0},
        "attack_distribution": [],
    }

    step1_rows = _step_rows(latest_run, 1)
    step2_rows = _step_rows(latest_run, 2)
    step3_rows = _step_rows(latest_run, 3)

    benchmarks = _benchmarks_from_step1(step1_rows)
    benchmarks_with_confusion = _benchmarks_with_confusion(step1_rows, dataset_stats)
    attacks = _attack_rows_from_step2(step2_rows)
    attack_shift_rows = _attack_shift_rows(step2_rows)
    attack_shift_summary = _attack_shift_summary(attack_shift_rows)
    drift_windows = _drift_windows_from_step2(step2_rows)
    metric_pack = _metric_pack_from_run(latest_run)

    gain_values = [_safe_float(r.get("acc_gain")) for r in step3_rows]
    normalized_gains = [min(max((g + 1.0) / 2.0, 0.0), 1.0) for g in gain_values]
    score_hist = _histogram_from_values(normalized_gains, n_bins=12)

    details = _model_details_for_run(latest_run, shap_map, dataset_stats)
    alerts, samples = _alerts_and_samples_from_run(latest_run)

    pr_curve = _pr_curve_from_point(metric_pack["precision"], metric_pack["recall"], metric_pack["average_precision"])

    confusion = {
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "tp": 0,
    }

    if isinstance(fallback.get("confusion"), dict):
        for key in ["tn", "fp", "fn", "tp"]:
            confusion[key] = int(fallback["confusion"].get(key, 0))

    dashboard = {
        "meta": {
            "dataset": selected_dataset,
            "run_id": latest_run.get("run_id", ""),
            "generated_at": latest_run.get("generated_at", ""),
            "primary_model": metric_pack.get("model", ""),
        },
        "dataset_overview": _dataset_overview_from_run(latest_run, fallback, dataset_stats),
        "dataset_stats": dataset_stats,
        "metrics": {
            "accuracy": metric_pack["accuracy"],
            "precision": metric_pack["precision"],
            "recall": metric_pack["recall"],
            "f1": metric_pack["f1"],
            "average_precision": metric_pack["average_precision"],
        },
        "confusion": confusion,
        "pr_curve": pr_curve,
        "score_histogram": score_hist,
        "drift_windows": drift_windows,
        "alerts_preview": alerts,
        "shap_top_features": shap_map.get("LSTM", [])[:20],
        "shap_by_model": {
            "LSTM": shap_map.get("LSTM", [])[:20],
            "Autoencoder": shap_map.get("Autoencoder", [])[:20],
            "DSFANet": shap_map.get("DSFANet", [])[:20],
        },
        "benchmark_models": benchmarks,
        "benchmark_models_confusion": benchmarks_with_confusion,
        "attack_results": attacks,
        "attack_shift_rows": attack_shift_rows,
        "attack_shift_summary": attack_shift_summary,
        "model_details": details,
        "sample_ids": [row["sample_id"] for row in alerts],
        "sample_details": samples,
    }

    return dashboard


def _runtime_payload(data_dir: Path, dataset: str | None = None) -> dict[str, Any]:
    """Assemble all API payloads from export files."""
    experiments_raw = _load_experiments_all(data_dir)
    experiments = _canonicalize_payload(experiments_raw)
    experiments, resolved_dataset = _filter_payload_by_dataset(experiments, dataset)

    dashboard = _build_dashboard_payload(data_dir, experiments, dataset=resolved_dataset)

    model_details = dashboard.get("model_details", {})
    alerts = dashboard.get("alerts_preview", [])
    sample_details = dashboard.get("sample_details", {})

    return {
        "dashboard": dashboard,
        "alerts": alerts,
        "models": model_details,
        "samples": sample_details,
        "experiments": experiments,
        "latest_run": _pick_latest_run(experiments, dataset=dataset),
        "dataset_options": _list_latest_by_dataset(experiments),
    }


def _query_dataset() -> str | None:
    """Read dataset query parameter from request."""
    value = request.args.get("dataset", "")
    if _is_invalid_dataset_token(value):
        return None
    return str(value).strip() or None


def _experiments_payload_for_request(data_root: Path, dataset: str | None = None) -> dict[str, Any]:
    """Load and optionally filter canonical experiments payload."""
    payload = _canonicalize_payload(_load_experiments_all(data_root))
    filtered, _ = _filter_payload_by_dataset(payload, dataset)
    return filtered


def serve_dashboard(data_dir: str = "out/www", host: str = "127.0.0.1", port: int = 8000, quiet: bool = False) -> None:
    """Serve dashboard APIs backed by step8 exports and cached step0 stats."""
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    data_root = Path(data_dir)

    if quiet:
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)

    @app.route("/")
    def index():
        """Return service metadata and endpoint list."""
        return jsonify(
            {
                "message": "IDS dashboard API is running.",
                "data_source": str(data_root),
                "endpoints": [
                    "/api/dashboard",
                    "/api/alerts",
                    "/api/models",
                    "/api/model/<name>",
                    "/api/sample/<sample_id>",
                    "/api/experiments/latest",
                    "/api/experiments/index",
                    "/api/experiments/all",
                    "/api/datasets",
                    "/api/dataset/stats",
                    "/api/benchmarks",
                    "/api/attacks",
                    "/api/retrain-strategy",
                ],
            }
        )

    @app.route("/api/datasets")
    def api_datasets():
        """Return selectable datasets with latest run metadata."""
        payload = _experiments_payload_for_request(data_root)
        return jsonify(_list_latest_by_dataset(payload))

    @app.route("/api/dashboard")
    def api_dashboard():
        runtime = _runtime_payload(data_root, dataset=_query_dataset())
        return jsonify(runtime["dashboard"])

    @app.route("/api/dataset/stats")
    def api_dataset_stats():
        """Return class and attack distributions for selected dataset."""
        runtime = _runtime_payload(data_root, dataset=_query_dataset())
        return jsonify(runtime["dashboard"].get("dataset_stats", {}))

    @app.route("/api/benchmarks")
    def api_benchmarks():
        """Return benchmark rows with confusion matrix info."""
        runtime = _runtime_payload(data_root, dataset=_query_dataset())
        return jsonify(runtime["dashboard"].get("benchmark_models_confusion", []))

    @app.route("/api/attacks")
    def api_attacks():
        """Return attacks and shifts tables plus aggregate loss summary."""
        runtime = _runtime_payload(data_root, dataset=_query_dataset())
        rows = runtime["dashboard"].get("attack_shift_rows", [])
        return jsonify(
            {
                "summary": runtime["dashboard"].get("attack_shift_summary", []),
                "attacks": [r for r in rows if r.get("kind") == "attack"],
                "shifts": [r for r in rows if r.get("kind") == "shift"],
            }
        )

    @app.route("/api/retrain-strategy")
    def api_retrain_strategy():
        """Return selected dataset latest retraining rows from step3."""
        runtime = _runtime_payload(data_root, dataset=_query_dataset())
        run = runtime.get("latest_run", {})
        return jsonify(_step_rows(run, 3))

    @app.route("/api/alerts")
    def api_alerts():
        runtime = _runtime_payload(data_root, dataset=_query_dataset())
        rows = runtime["alerts"]

        min_score = _safe_float(request.args.get("min_score", "0"), 0.0)
        rows = [r for r in rows if _safe_float(r.get("voting_score")) >= min_score]
        rows.sort(key=lambda x: _safe_float(x.get("voting_score")), reverse=True)

        has_page = request.args.get("page") is not None
        has_page_size = request.args.get("page_size") is not None
        if not has_page and not has_page_size:
            return jsonify(rows)

        page = max(int(request.args.get("page", "1") or 1), 1)
        page_size = max(int(request.args.get("page_size", "50") or 50), 1)
        start = (page - 1) * page_size
        end = start + page_size
        sliced = rows[start:end]

        return jsonify(
            {
                "rows": sliced,
                "total": len(rows),
                "page": page,
                "page_size": page_size,
            }
        )

    @app.route("/api/models")
    def api_models():
        runtime = _runtime_payload(data_root, dataset=_query_dataset())
        return jsonify(runtime["models"])

    @app.route("/api/model/<name>")
    def api_model(name: str):
        runtime = _runtime_payload(data_root, dataset=_query_dataset())
        payload = runtime["models"]
        if name not in payload:
            return jsonify({"error": f"model '{name}' not found"}), 404
        return jsonify(payload[name])

    @app.route("/api/sample/<sample_id>")
    def api_sample(sample_id: str):
        runtime = _runtime_payload(data_root, dataset=_query_dataset())
        payload = runtime["samples"]
        if sample_id not in payload:
            return jsonify({"error": f"sample '{sample_id}' not found"}), 404
        return jsonify(payload[sample_id])

    @app.route("/api/experiments/latest")
    def api_experiments_latest():
        """Return latest run payload with summary_step aliases."""
        payload = _experiments_payload_for_request(data_root, dataset=_query_dataset())
        latest = _pick_latest_run(payload, dataset=_query_dataset())
        if not latest:
            return jsonify({"error": "No experiments export found under out/www."}), 404
        return jsonify(latest)

    @app.route("/api/experiments/index")
    def api_experiments_index():
        payload = _experiments_payload_for_request(data_root, dataset=_query_dataset())
        runs = payload.get("runs", [])
        index_rows = [
            {
                "run_id": run.get("run_id", ""),
                "generated_at": run.get("generated_at", ""),
                "base_dataset": run.get("base_dataset", ""),
                "ood_dataset": run.get("ood_dataset", ""),
            }
            for run in runs
        ]
        return jsonify(
            {
                "updated_at": payload.get("updated_at", ""),
                "latest_run_id": payload.get("latest_run_id", ""),
                "runs": index_rows,
            }
        )

    @app.route("/api/experiments/all")
    def api_experiments_all():
        payload = _experiments_payload_for_request(data_root, dataset=_query_dataset())
        return jsonify(payload)

    app.run(host=host, port=port)


def main():
    parser = argparse.ArgumentParser(description="Serve dashboard APIs from step8 export files")
    parser.add_argument("--dataset", default="NF-UNSW-NB15-v3.csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", default="out/www")
    # parser.add_argument("--skip-serve", action="store_true")
    # parser.add_argument("--serve-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    serve_dashboard(data_dir=args.data_dir, host=args.host, port=args.port, quiet=args.quiet)


if __name__ == "__main__":
    main()
