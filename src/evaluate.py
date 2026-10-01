"""Evaluate frozen baseline and MLP on TEST once, then write error analysis."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from threadpoolctl import threadpool_limits

from src.common import PROJECT_ROOT, file_sha256, read_json, timestamp, write_json
from src.data import load_prepared_data
from src.metrics import score_model


def evaluate(root=PROJECT_ROOT, force=False, reason=None):
    root = Path(root)
    destination = root / "reports/evaluation.json"
    access_path = root / "data/test_access.json"
    access_log = read_json(access_path) if access_path.exists() else {"events": []}
    prior = None
    previously_accessed = destination.exists() or access_path.exists()
    if previously_accessed:
        if not force or not reason or not reason.strip():
            raise RuntimeError("Final test already evaluated. Read reports/evaluation.json; a repeat requires --force --reason.")
        prior = read_json(destination) if destination.exists() else None
    metadata = read_json(root / "models/model_metadata.json")
    for filename, expected in (("digit_pipeline.joblib", metadata["model_sha256"]),
                               ("baseline_pipeline.joblib", metadata["baseline_sha256"])):
        if file_sha256(root / "models" / filename) != expected:
            raise ValueError("Model checksum changed after selection; refusing test evaluation.")
    if file_sha256(root / "data/splits.json") != metadata["split_sha256"]:
        raise ValueError("Frozen split changed after model selection.")
    X, y, splits, manifest = load_prepared_data(root)
    if manifest["sha256"] != metadata["dataset_sha256"]:
        raise ValueError("Model and dataset checksums differ.")
    # Write BEFORE reading/scoring test samples: failures cannot erase access history.
    access_log["events"].append({"accessed_at": timestamp(), "status": "started",
                                 "dataset_sha256": manifest["sha256"],
                                 "model_sha256": metadata["model_sha256"],
                                 "repeat_reason": reason if previously_accessed else None})
    write_json(access_path, access_log)
    indices = splits["test"]
    X_test, y_test = X[indices], y[indices]
    baseline = joblib.load(root / "models/baseline_pipeline.joblib")
    mlp = joblib.load(root / "models/digit_pipeline.joblib")
    models = {}
    with threadpool_limits(limits=1):
        for key, pipeline, name in (("baseline", baseline, "Logistic Regression"), ("mlp", mlp, "MLPClassifier")):
            scores, predicted, probabilities = score_model(pipeline, X_test, y_test, detailed=True)
            models[key] = {"name": name, **scores}
            if key == "mlp":
                mlp_predicted, mlp_probabilities = predicted, probabilities
    confusion = np.asarray(models["mlp"]["confusion_matrix"])
    pairs = confusion + confusion.T
    np.fill_diagonal(pairs, 0)
    a, b = np.unravel_index(np.argmax(pairs), pairs.shape)
    if a > b:
        a, b = b, a
    pair = {"digits": [int(a), int(b)], "count": int(pairs[a, b]),
            "a_to_b": int(confusion[a, b]), "b_to_a": int(confusion[b, a])} if pairs.max() else None
    wrong = np.flatnonzero(mlp_predicted != y_test)
    errors = [{"sample_id": int(indices[i]), "true_label": int(y_test[i]),
               "predicted_label": int(mlp_predicted[i]), "confidence": float(mlp_probabilities[i].max()),
               "pixels": X_test[i].reshape(8, 8).tolist()} for i in wrong]
    figures = root / "reports/figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key in zip(axes, ("baseline", "mlp")):
        ConfusionMatrixDisplay(np.asarray(models[key]["confusion_matrix"]), display_labels=range(10)).plot(
            ax=ax, colorbar=False, cmap="Blues", values_format="d")
        ax.set_title(f"{models[key]['name']} — TEST n={len(indices)}")
    fig.tight_layout()
    fig.savefig(figures / "confusion_matrix.png", dpi=180)
    plt.close(fig)
    render_error_examples(root, errors, pair)
    result = {
        "evaluated_at": timestamp(), "n_test": len(indices), "split": "test", "models": models,
        "accuracy_delta": models["mlp"]["accuracy"] - models["baseline"]["accuracy"],
        "most_confused_pair": pair, "errors": errors,
        "figures": {name: f"/reports/figures/{name}.png" for name in (
            "confusion_matrix", "error_examples", "learning_curve", "training_history", "class_distribution", "train_samples")},
        "integrity": {"dataset_sha256": manifest["sha256"], "model_sha256": metadata["model_sha256"],
                      "split_sha256": metadata["split_sha256"], "selection": "validation only; model hashes verified before test",
                      "repeated_evaluation": previously_accessed, "rerun_after_test_inspection": metadata.get("rerun_after_test_inspection", False)},
        "repeat_reason": reason if previously_accessed else None,
    }
    if prior:
        archive = root / "reports/archive" / (timestamp().replace(":", "-") + "-evaluation")
        write_json(archive / "evaluation.json", prior)
    write_json(destination, result)
    _write_report(root, result, metadata)
    access_log["events"][-1]["status"] = "complete"
    write_json(access_path, access_log)
    for key, scores in models.items():
        print(f"TEST {key}: accuracy={scores['accuracy']:.4f}; macro-F1={scores['macro_f1']:.4f}; log loss={scores['log_loss']:.4f}")
    return result


def render_error_examples(root, errors, pair):
    """Render already saved error records; no model or test scoring is needed."""
    pair_errors = [e for e in errors if pair and {e["true_label"], e["predicted_label"]} == set(pair["digits"])]
    chosen = (pair_errors + [e for e in errors if e not in pair_errors])[:12]
    fig, axes = plt.subplots(2, 6, figsize=(12, 6))
    for ax in axes.flat:
        ax.axis("off")
    for ax, item in zip(axes.flat, chosen):
        ax.imshow(item["pixels"], cmap="gray", vmin=0, vmax=16)
        ax.set_title(f"True {item['true_label']} → pred {item['predicted_label']}\n{item['confidence']:.1%} · ID {item['sample_id']}", fontsize=9)
    fig.suptitle("MLP test errors — most-confused pair shown first" if chosen else "No MLP errors on this test split")
    fig.subplots_adjust(top=0.87, bottom=0.04, hspace=0.45, wspace=0.2)
    fig.savefig(Path(root) / "reports/figures/error_examples.png", dpi=180)
    plt.close(fig)


def _write_report(root, result, metadata):
    baseline, mlp = result["models"]["baseline"], result["models"]["mlp"]
    pair = result["most_confused_pair"]
    lines = ["# Kết quả chạy thực tế", "", f"Thời điểm đánh giá (UTC): {result['evaluated_at']}.", "",
             f"Split stratified seed {metadata['seed']}: {metadata['split']}. Test được giữ riêng khi chọn cấu hình.", "",
             "| Mô hình | Accuracy | Macro-F1 | Log loss |", "|---|---:|---:|---:|",
             f"| Logistic Regression | {baseline['accuracy']:.4f} | {baseline['macro_f1']:.4f} | {baseline['log_loss']:.4f} |",
             f"| MLP | {mlp['accuracy']:.4f} | {mlp['macro_f1']:.4f} | {mlp['log_loss']:.4f} |", "",
             f"MLP chênh {result['accuracy_delta'] * 100:+.2f} điểm phần trăm accuracy so với baseline trên cùng {result['n_test']} ảnh test.",
             "Đây là kết quả một split nhỏ, không chứng minh MLP luôn tốt hơn baseline trên dữ liệu khác.", "",
             f"Cấu hình chọn bằng validation: `{metadata['selected_experiment']}` — `{metadata['configuration']}`.",
             "Xem reports/experiments.csv cho so sánh baseline/MLP, một/hai lớp ẩn và alpha/early stopping; evaluation.json chứa phân tích lỗi.", "",
             f"Cặp nhầm nhiều nhất: {pair['digits'][0]} ↔ {pair['digits'][1]}, tổng {pair['count']} lỗi hai chiều." if pair else "Không có lỗi phân loại trên test này.",
             f"Tổng {len(result['errors'])} ảnh MLP dự đoán sai được lưu trong evaluation.json; ví dụ ở figures/error_examples.png.",
             "Không sửa siêu tham số hoặc bộ chuyển canvas dựa trên các lỗi test này.", "",
             "Learning curve: trung bình và độ lệch chuẩn 3 fold chỉ trên TRAIN; không phải khoảng tin cậy của test.",
             "Phân tích hình ảnh có thể gợi ý nét mờ/biến thể hình dạng; không đủ dữ liệu để kết luận nguyên nhân lỗi.", "",
             "Nguồn dữ liệu: [scikit-learn Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html),",
             "[UCI Optical Digits](https://archive.ics.uci.edu/dataset/80/optical%2Brecognition%2Bof%2Bhandwritten%2Bdigits)."]
    (root / "reports/results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    card = ["# Model card", "", "Mục đích: học tập nhận dạng một chữ số viết tay 0–9, ảnh xám 8×8.", "",
            f"Mô hình: MLPClassifier `{metadata['selected_experiment']}`; `{metadata['configuration']}`.",
            f"Dữ liệu: 1.797 ảnh Digits; {metadata['license']}; seed {metadata['seed']}; sklearn {metadata['sklearn_version']}.",
            f"Tiền xử lý: {metadata['preprocessing']}.", f"Canvas: {metadata['canvas_adapter']}.", "",
            f"Kết quả test: Accuracy {mlp['accuracy']:.4f}, macro-F1 {mlp['macro_f1']:.4f}, log loss {mlp['log_loss']:.4f}.", "",
            "Giới hạn:", "", *[f"- {item}" for item in metadata["limitations"]], "",
            "API từ chối dữ liệu sai schema/ngoài miền; không phát hiện đầy đủ đầu vào ngoài miền (ví dụ chữ cái).",
            "Không thu thập hoặc lưu nét vẽ của người dùng; ảnh được xử lý trong bộ nhớ phục vụ request.", "",
            f"SHA256 model: `{metadata['model_sha256']}`.", f"SHA256 dữ liệu: `{metadata['dataset_sha256']}`."]
    (root / "reports/model_card.md").write_text("\n".join(card) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reason", help="Required audit reason for repeating an already completed test evaluation")
    args = parser.parse_args()
    evaluate(args.root, args.force, args.reason)


if __name__ == "__main__":
    main()
