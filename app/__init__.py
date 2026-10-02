"""Flask serving layer for the offline-trained handwritten digit models.

The application loads local, trusted joblib artifacts once during startup.
Training and evaluation live in ``src`` and never run inside a request.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.exceptions import BadRequest, HTTPException, RequestEntityTooLarge

from src.features import canvas_to_pixels, validate_pixels

MAX_REQUEST_BYTES = 2 * 1024 * 1024
LOW_CONFIDENCE_THRESHOLD = 0.75
DOMAIN_WARNING = (
    "Mô hình học từ scikit-learn Digits: ảnh xám 8×8, pixel 0–16. "
    "Ảnh vẽ tay mới có thể khác dữ liệu huấn luyện; xác suất cao không bảo đảm đúng. "
    "Ứng dụng phục vụ học tập, không dùng cho tài liệu pháp lý."
)


def _read_json(path: Path, diagnostics: list[str]) -> Any | None:
    """Read an optional report without preventing the web pages from opening."""
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError):
        diagnostics.append(f"Không đọc được {path.name}; hãy tạo lại artifact offline.")
        return None


def _load_pipeline(path: Path, diagnostics: list[str]) -> Any | None:
    if not path.is_file():
        return None
    try:
        model = joblib.load(path)
        if not callable(getattr(model, "predict", None)) or not callable(
            getattr(model, "predict_proba", None)
        ):
            raise ValueError("Artifact is not a probabilistic classifier")
        return model
    except Exception:
        diagnostics.append(f"Không nạp được {path.name}; hãy huấn luyện lại offline.")
        return None


def create_app(project_root: str | Path | None = None, testing: bool = False) -> Flask:
    """Create an app with artifacts resolved relative to a portable project root.

    Tests can provide a temporary root containing models and reports. Templates
    and static assets always come from the installed application package.
    """
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    app = Flask(__name__)
    app.config.update(TESTING=testing, MAX_CONTENT_LENGTH=MAX_REQUEST_BYTES)

    diagnostics: list[str] = []
    model = _load_pipeline(root / "models" / "digit_pipeline.joblib", diagnostics)
    baseline = _load_pipeline(root / "models" / "baseline_pipeline.joblib", diagnostics)
    metadata = _read_json(root / "models" / "model_metadata.json", diagnostics)
    evaluation = _read_json(root / "reports" / "evaluation.json", diagnostics)
    experiments = _read_json(root / "reports" / "experiments.json", diagnostics)
    demo_samples = _read_json(root / "data" / "demo_samples.json", diagnostics)
    if metadata is not None and not isinstance(metadata, dict):
        diagnostics.append("model_metadata.json phải là một đối tượng JSON.")
        metadata = None

    app.extensions["digit_service"] = {
        "model": model,
        "baseline": baseline,
        "metadata": metadata,
        "evaluation": evaluation,
        "experiments": experiments,
        "demo_samples": demo_samples,
        "diagnostics": diagnostics,
    }

    def error_response(code: str, message: str, status: int):
        return jsonify(error={"code": code, "message": message}), status

    def service_status() -> str:
        if model is not None:
            return "ready"
        return "artifact_error" if diagnostics else "model_missing"

    def service_message() -> str | None:
        if model is None:
            return (
                "Chưa có mô hình sẵn sàng. Chạy python -m src.train rồi "
                "python -m src.evaluate và khởi động lại ứng dụng."
            )
        return " ".join(diagnostics) if diagnostics else None

    @app.get("/")
    def index():
        return render_template("index.html", active_page="home")

    @app.get("/recognize")
    def recognize():
        return render_template("recognize.html", active_page="recognize")

    @app.get("/dashboard")
    def dashboard():
        return render_template("dashboard.html", active_page="dashboard")

    @app.get("/reports/figures/<path:filename>")
    def report_figure(filename: str):
        # send_from_directory checks traversal and restricts reads to this folder.
        return send_from_directory(root / "reports" / "figures", filename)

    @app.get("/health")
    def health():
        return jsonify(status=service_status(), model_loaded=model is not None)

    @app.get("/api/model")
    def model_information():
        return jsonify(
            status=service_status(),
            model_loaded=model is not None,
            metadata=metadata,
            evaluation=evaluation,
            experiments=experiments,
            message=service_message(),
        )

    @app.get("/api/examples")
    def examples():
        if demo_samples is None:
            return jsonify(
                examples=[], source="train",
                message="Chạy python -m src.data để tạo các mẫu demo từ tập train.",
            )
        if not isinstance(demo_samples, list):
            return error_response("invalid_artifact", "Tệp mẫu demo không hợp lệ; hãy tạo lại dữ liệu.", 503)
        # Validate saved examples too, so corrupted artifacts never reach the UI.
        clean_samples = []
        try:
            for sample in demo_samples:
                if not isinstance(sample, dict):
                    raise ValueError("Invalid demo example")
                label = sample["label"]
                if isinstance(label, bool) or not isinstance(label, int) or label not in range(10):
                    raise ValueError("Invalid demo label")
                pixels = validate_pixels(sample["pixels"])
                clean_samples.append({
                    "pixels": np.asarray(pixels).reshape(8, 8).tolist(),
                    "label": label,
                    "id": sample.get("id"),
                })
        except (ValueError, TypeError, OverflowError, KeyError):
            return error_response("invalid_artifact", "Tệp mẫu demo không hợp lệ; hãy tạo lại dữ liệu.", 503)
        return jsonify(examples=clean_samples, source="train", message=None)

    @app.post("/api/digit")
    def predict_digit():
        # Parsing and validation precede availability checks, even without a model.
        if not request.is_json:
            return error_response("invalid_json", "Gửi JSON với Content-Type: application/json.", 400)
        try:
            payload = request.get_json()
        except BadRequest:
            return error_response("invalid_json", "Nội dung JSON không hợp lệ.", 400)
        if not isinstance(payload, dict) or set(payload) not in ({"pixels"}, {"canvas"}):
            return error_response(
                "invalid_input", "JSON phải chứa đúng một trường pixels hoặc canvas, không có trường khác.", 422
            )
        try:
            pixels = (
                validate_pixels(payload["pixels"])
                if "pixels" in payload else canvas_to_pixels(payload["canvas"])
            )
        except (ValueError, TypeError, OverflowError) as exc:
            message = "Pixel quá lớn; hãy dùng đúng miền giá trị của ảnh." if isinstance(exc, OverflowError) else str(exc)
            return error_response("invalid_input", message, 422)
        if model is None:
            return error_response("model_unavailable", service_message() or "Mô hình chưa sẵn sàng.", 503)

        sample = np.asarray(pixels, dtype=float).reshape(1, 64)
        predicted = np.asarray(model.predict(sample)).reshape(-1)
        probabilities = np.asarray(model.predict_proba(sample), dtype=float)
        classes = np.asarray(model.classes_).reshape(-1)
        # Refuse incompatible artifacts instead of inventing a usable prediction.
        if (
            predicted.size != 1
            or probabilities.shape != (1, classes.size)
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or np.any(probabilities > 1)
            or not np.isclose(probabilities.sum(), 1.0, atol=1e-6)
            or any(isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) not in range(10) for value in classes)
            or len(set(int(value) for value in classes)) != classes.size
            or int(predicted[0]) != predicted[0]
            or int(predicted[0]) not in range(10)
        ):
            app.logger.error("Model artifact returned an incompatible prediction")
            return error_response("model_unavailable", "Artifact mô hình không tương thích; hãy huấn luyện lại offline.", 503)
        all_probabilities = np.zeros(10, dtype=float)
        for digit, probability in zip(classes, probabilities[0]):
            all_probabilities[int(digit)] = float(probability)
        prediction = int(predicted[0])
        confidence = float(all_probabilities[prediction])
        warnings = [DOMAIN_WARNING]
        if "canvas" in payload:
            warnings.append(
                "Canvas được căn giữa và thu nhỏ về 8×8; nét vẽ, vị trí và độ dày ảnh hưởng kết quả."
            )
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            warnings.append("Độ tin cậy dưới 75%; hãy kiểm tra ảnh 8×8 và thử vẽ rõ hơn.")
        return jsonify(
            prediction=prediction,
            probabilities=[{"digit": digit, "probability": float(all_probabilities[digit])} for digit in range(10)],
            pixels=sample.reshape(8, 8).tolist(),
            confidence=confidence,
            model_name=(metadata or {}).get("model_name", "MLPClassifier"),
            warnings=warnings,
        )

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return error_response("payload_too_large", "Nội dung yêu cầu vượt giới hạn 2 MiB.", 413)

    @app.errorhandler(HTTPException)
    def http_error(error):
        if request.path.startswith("/api/") or request.path == "/health":
            return error_response("http_error", error.description, error.code or 500)
        return error

    @app.errorhandler(Exception)
    def unexpected_error(error):
        app.logger.exception("Unexpected application error", exc_info=error)
        if request.path.startswith("/api/") or request.path == "/health":
            return error_response("internal_error", "Không xử lý được yêu cầu; kiểm tra log của ứng dụng.", 500)
        return "Ứng dụng gặp lỗi. Kiểm tra log để biết chi tiết.", 500

    return app
