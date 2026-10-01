"""Run frozen validation experiments and save models without evaluating TEST."""
from __future__ import annotations

import argparse
import platform
import shutil
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from src.common import PROJECT_ROOT, file_sha256, read_json, timestamp, write_json
from src.data import load_prepared_data, prepare_data
from src.features import PixelScaler
from src.metrics import score_model

MLP_CONFIGURATIONS = [
    {"id": "mlp_64", "hidden_layer_sizes": (64,), "alpha": 0.0001, "early_stopping": True},
    {"id": "mlp_128", "hidden_layer_sizes": (128,), "alpha": 0.0001, "early_stopping": True},
    {"id": "mlp_64_32", "hidden_layer_sizes": (64, 32), "alpha": 0.0001, "early_stopping": True},
    {"id": "mlp_128_64", "hidden_layer_sizes": (128, 64), "alpha": 0.0001, "early_stopping": True},
    {"id": "mlp_128_64_alpha", "hidden_layer_sizes": (128, 64), "alpha": 0.01, "early_stopping": True},
    {"id": "mlp_no_early_stop", "hidden_layer_sizes": (128, 64), "alpha": 0.0001, "early_stopping": False},
]
SELECTION_METRIC = "validation accuracy, then macro-F1, then lower log loss; only early_stopping=True MLP eligible"


def make_mlp(configuration, seed):
    return Pipeline([("pixels", PixelScaler()), ("classifier", MLPClassifier(
        hidden_layer_sizes=tuple(configuration["hidden_layer_sizes"]), alpha=configuration["alpha"],
        early_stopping=configuration["early_stopping"], activation="relu", solver="adam",
        random_state=seed, max_iter=500, n_iter_no_change=25, tol=0.0001,
        validation_fraction=0.15, learning_rate_init=0.001, batch_size=64))])


def make_baseline(seed):
    return Pipeline([("pixels", PixelScaler()), ("classifier", LogisticRegression(
        C=1.0, max_iter=2000, random_state=seed, solver="lbfgs"))])


def archive_previous_run(root, reason):
    archive = root / "reports/archive" / timestamp().replace(":", "-")
    archive.mkdir(parents=True, exist_ok=False)
    for name in ("models", "reports/figures"):
        source = root / name
        if source.exists():
            shutil.copytree(source, archive / name)
    for name in ("evaluation.json", "experiments.json", "learning_curve.json", "model_card.md", "results.md"):
        source = root / "reports" / name
        if source.exists():
            shutil.copy2(source, archive / name)
    write_json(archive / "audit.json", {"reason": reason, "archived_at": timestamp(),
               "warning": "Test has already been inspected. A rerun is not a new independent test."})
    (root / "reports/evaluation.json").unlink(missing_ok=True)
    return str(archive.relative_to(root))


def _learning_curves(root, models, X, y, seed):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    results = {"scope": "TRAIN only,3 stratified folds; no external validation/test", "folds": 3, "seed": seed, "models": {}}
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            sizes, train_scores, fold_scores = learning_curve(
                clone(model), X, y, cv=cv, scoring="accuracy", train_sizes=[0.3, 0.6, 1.0],
                shuffle=True, random_state=seed, n_jobs=1, error_score="raise")
        result = {"train_sizes": sizes.tolist(), "train_mean": train_scores.mean(axis=1).tolist(),
                  "train_std": train_scores.std(axis=1).tolist(), "cv_mean": fold_scores.mean(axis=1).tolist(),
                  "cv_std": fold_scores.std(axis=1).tolist(), "cv_fold_scores": fold_scores.tolist()}
        results["models"][name] = result
        line, = ax.plot(sizes, result["cv_mean"], "o-", label=f"{name} CV")
        mean, std = fold_scores.mean(axis=1), fold_scores.std(axis=1)
        ax.fill_between(sizes, mean - std, mean + std, alpha=0.13, color=line.get_color())
        ax.plot(sizes, result["train_mean"], "--", color=line.get_color(), label=f"{name} train")
    ax.set(xlabel="Training samples per fold", ylabel="Accuracy", ylim=(0.65, 1.01), title="Learning curve — TRAIN only")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(root / "reports/figures/learning_curve.png", dpi=160)
    plt.close(fig)
    write_json(root / "reports/learning_curve.json", results)
    return results


def train(root=PROJECT_ROOT, force=False, reason=None):
    root = Path(root)
    audit = None
    if (root / "reports/evaluation.json").exists() or (root / "data/test_access.json").exists():
        if not force or not reason or not reason.strip():
            raise RuntimeError("Test results already exist. Training again requires --force --reason and archives the previous run.")
        audit = archive_previous_run(root, reason.strip())
    if not (root / "data/splits.json").exists():
        prepare_data(root)
    X, y, splits, manifest = load_prepared_data(root)
    seed = read_json(root / "data/splits.json")["seed"]
    train_indices, validation_indices = splits["train"], splits["validation"]
    # Do not predict on, score, plot or select using the test indices here.
    X_train, y_train = X[train_indices], y[train_indices]
    X_validation, y_validation = X[validation_indices], y[validation_indices]
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "reports/figures").mkdir(parents=True, exist_ok=True)
    fitted, experiments = {}, []
    baseline_config = {"id": "logistic", "C": 1.0, "max_iter": 2000, "solver": "lbfgs"}
    all_configs = [baseline_config] + MLP_CONFIGURATIONS
    for configuration in all_configs:
        config = configuration.copy()
        identifier = config.pop("id")
        kind = "baseline" if identifier == "logistic" else "mlp"
        pipeline = make_baseline(seed) if kind == "baseline" else make_mlp(config, seed)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
            warnings.simplefilter("always", ConvergenceWarning)
            pipeline.fit(X_train, y_train)
            training_metrics, _, _ = score_model(pipeline, X_train, y_train)
            validation_metrics, _, _ = score_model(pipeline, X_validation, y_validation)
        fitted[identifier] = pipeline
        record = {"id": identifier, "name": "Logistic Regression" if kind == "baseline" else identifier,
                  "kind": kind, "configuration": config,
                  "eligible_for_selection": kind == "mlp" and config["early_stopping"],
                  "train": training_metrics, "validation": validation_metrics,
                  "n_iter": int(np.max(pipeline[-1].n_iter_)), "fit_seconds": time.perf_counter() - started,
                  "warnings": [str(w.message) for w in captured]}
        experiments.append(record)
        print(f"{identifier:22s} validation accuracy={validation_metrics['accuracy']:.4f} macro-F1={validation_metrics['macro_f1']:.4f}", flush=True)
    eligible = [e for e in experiments if e["eligible_for_selection"]]
    best = max(eligible, key=lambda e: (e["validation"]["accuracy"], e["validation"]["macro_f1"], -e["validation"]["log_loss"]))
    selected = fitted[best["id"]]
    # Save the exact candidate scored on validation: no hidden refit changes.
    joblib.dump(selected, root / "models/digit_pipeline.joblib", compress=3)
    joblib.dump(fitted["logistic"], root / "models/baseline_pipeline.joblib", compress=3)
    with threadpool_limits(limits=1):
        curves = _learning_curves(root, {"baseline": fitted["logistic"], "mlp": selected}, X_train, y_train, seed)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(selected[-1].loss_curve_, color="#2869c8")
    axes[0].set(xlabel="Epoch", ylabel="Training loss", title="Selected MLP loss history")
    axes[1].plot(selected[-1].validation_scores_, color="#168678")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Internal early-stopping validation")
    fig.tight_layout()
    fig.savefig(root / "reports/figures/training_history.png", dpi=160)
    plt.close(fig)
    payload = {"generated_at": timestamp(), "selection_metric": SELECTION_METRIC,
               "selected_experiment": best["id"], "experiments": experiments, "learning_curve": curves,
               "protocol": "Fit on train only; external validation selects; test remains unseen.",
               "rerun_archive": audit}
    write_json(root / "reports/experiments.json", payload)
    pd.DataFrame([{**{k: e[k] for k in ("id", "kind", "n_iter", "fit_seconds", "eligible_for_selection")},
                   **{f"validation_{k}": v for k, v in e["validation"].items()},
                   "configuration": str(e["configuration"])} for e in experiments]).to_csv(root / "reports/experiments.csv", index=False)
    metadata = {
        "model_name": "MLPClassifier", "selected_experiment": best["id"], "configuration": best["configuration"],
        "common_configuration": {"activation": "relu", "solver": "adam", "max_iter": 500,
                                 "validation_fraction": 0.15, "n_iter_no_change": 25, "batch_size": 64},
        "seed": seed, "split": {k: len(v) for k, v in splits.items()}, "trained_at": timestamp(),
        "sklearn_version": sklearn.__version__, "python_version": platform.python_version(),
        "dataset_sha256": manifest["sha256"], "split_sha256": file_sha256(root / "data/splits.json"),
        "model_sha256": file_sha256(root / "models/digit_pipeline.joblib"),
        "baseline_sha256": file_sha256(root / "models/baseline_pipeline.joblib"),
        "preprocessing": "8×8 grayscale pixels0..16; PixelScaler divides by16 in both training and API",
        "canvas_adapter": "White strokes on black280×280; crop, aspect-preserving center, area mean to8×8, intensity×16/255",
        "selection_metric": SELECTION_METRIC, "validation_metrics": best["validation"],
        "n_iter": best["n_iter"], "source": manifest["source"], "license": manifest["license"],
        "limitations": ["Dữ liệu nhỏ:1.797 ảnh chuẩn hóa8×8; độ chính xác test không đại diện mọi nét vẽ canvas.",
                        "Canvas dùng bộ chuyển ảnh cố định; vẫn có khác biệt miền dữ liệu với Digits gốc.",
                        "Chỉ nhận một chữ số0–9; không hỗ trợ chữ, nhiều chữ số, ảnh tài liệu hay mục đích pháp lý.",
                        "Xác suất chưa hiệu chỉnh; độ tin cậy cao không bảo đảm đầu vào nằm trong miền.",
                        "Không có ID người viết để kiểm chứng tổng quát hóa theo người viết độc lập."],
        "rerun_after_test_inspection": bool(audit), "rerun_reason": reason if audit else None,
    }
    write_json(root / "models/model_metadata.json", metadata)
    print(f"Frozen selected model: {best['id']}. Next: python -m src.evaluate", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reason", help="Audit reason required when retraining after test results exist")
    args = parser.parse_args()
    train(args.root, args.force, args.reason)


if __name__ == "__main__":
    main()
