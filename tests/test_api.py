"""Behavior tests for HTTP validation, model loading, and serving contracts."""

import json

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from app import MAX_REQUEST_BYTES, create_app


@pytest.fixture
def pixels():
    result = np.zeros((8, 8), dtype=float)
    result[1:7, 3:5] = 16
    return result.tolist()


@pytest.fixture
def artifact_root(tmp_path, pixels):
    (tmp_path / "models").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "data").mkdir()
    # All classes occur; deterministic predictions need no expensive training.
    classifier = DummyClassifier(strategy="constant", constant=3)
    classifier.fit(np.zeros((10, 64)), np.arange(10))
    joblib.dump(classifier, tmp_path / "models" / "digit_pipeline.joblib")
    (tmp_path / "models" / "model_metadata.json").write_text(json.dumps({"model_name": "test-model"}), encoding="utf-8")
    (tmp_path / "reports" / "evaluation.json").write_text(json.dumps({"split": "test"}), encoding="utf-8")
    (tmp_path / "reports" / "experiments.json").write_text(json.dumps({"split": "validation"}), encoding="utf-8")
    (tmp_path / "data" / "demo_samples.json").write_text(json.dumps([{"id": 17, "label": 3, "pixels": pixels}]), encoding="utf-8")
    return tmp_path


@pytest.fixture
def client(artifact_root):
    return create_app(artifact_root, testing=True).test_client()


def test_prediction_schema_and_class_order(client, pixels):
    response = client.post("/api/digit", json={"pixels": pixels})
    assert response.status_code == 200
    body = response.get_json()
    assert body["prediction"] == 3
    assert body["model_name"] == "test-model"
    assert body["confidence"] == 1.0
    assert body["pixels"] == pixels
    assert [item["digit"] for item in body["probabilities"]] == list(range(10))
    assert sum(item["probability"] for item in body["probabilities"]) == pytest.approx(1)
    assert body["warnings"]


def test_flat_pixels_are_supported(client, pixels):
    response = client.post("/api/digit", json={"pixels": np.asarray(pixels).ravel().tolist()})
    assert response.status_code == 200
    assert response.get_json()["pixels"] == pixels


def test_canvas_is_processed_and_warns(client):
    canvas = np.zeros((280, 280))
    canvas[50:230, 125:155] = 255
    response = client.post("/api/digit", json={"canvas": canvas.tolist()})
    assert response.status_code == 200
    body = response.get_json()
    assert np.asarray(body["pixels"]).shape == (8, 8)
    assert np.asarray(body["pixels"]).min() >= 0
    assert np.asarray(body["pixels"]).max() <= 16
    assert len(body["warnings"]) >= 2


@pytest.mark.parametrize("payload", [
    None, [], {}, {"extra": 1}, {"pixels": [1] * 63}, {"pixels": [17] * 64},
    {"pixels": [-1] * 64}, {"pixels": [True] * 64}, {"pixels": ["1"] * 64},
    {"pixels": [0] * 64}, {"pixels": [float("nan")] * 64}, {"pixels": [float("inf")] * 64},
    {"pixels": [10 ** 400] * 64},
    {"pixels": [1] * 64, "canvas": []}, {"pixels": [1] * 64, "label": 3},
    {"canvas": [[0] * 8] * 8},
])
def test_invalid_schema_is_422(client, payload):
    # Serialize null explicitly: Flask's json=None means no request body.
    response = client.post("/api/digit", data=json.dumps(payload), content_type="application/json")
    assert response.status_code == 422
    assert response.get_json()["error"]["code"] == "invalid_input"


@pytest.mark.parametrize("body,content_type", [("{broken", "application/json"), ("{}", "text/plain"), ("", "application/json")])
def test_bad_json_is_400(client, body, content_type):
    response = client.post("/api/digit", data=body, content_type=content_type)
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_json"


def test_large_request_is_413(client):
    response = client.post("/api/digit", data=b"x" * (MAX_REQUEST_BYTES + 1), content_type="application/json")
    assert response.status_code == 413
    assert response.get_json()["error"]["code"] == "payload_too_large"


def test_absent_model_still_validates_and_returns_actionable_503(tmp_path, pixels):
    client = create_app(tmp_path, testing=True).test_client()
    assert client.post("/api/digit", json={"pixels": []}).status_code == 422
    response = client.post("/api/digit", json={"pixels": pixels})
    assert response.status_code == 503
    assert "python -m src.train" in response.get_json()["error"]["message"]
    assert client.get("/health").get_json() == {"status": "model_missing", "model_loaded": False}
    assert client.get("/api/examples").get_json()["examples"] == []


def test_corrupt_model_does_not_prevent_startup(tmp_path, pixels):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "digit_pipeline.joblib").write_bytes(b"invalid-joblib")
    client = create_app(tmp_path, testing=True).test_client()
    assert client.get("/health").get_json()["status"] == "artifact_error"
    assert client.post("/api/digit", json={"pixels": pixels}).status_code == 503


def test_model_and_examples_metadata(client):
    body = client.get("/api/model").get_json()
    assert body["status"] == "ready"
    assert body["evaluation"]["split"] == "test"
    assert body["experiments"]["split"] == "validation"
    examples = client.get("/api/examples").get_json()
    assert examples["source"] == "train"
    assert examples["examples"][0]["id"] == 17
    assert examples["examples"][0]["label"] == 3


def test_model_is_loaded_once_at_startup(artifact_root, monkeypatch, pixels):
    original_load = joblib.load
    loaded = []
    def track_load(path):
        loaded.append(path)
        return original_load(path)
    monkeypatch.setattr(joblib, "load", track_load)
    client = create_app(artifact_root, testing=True).test_client()
    client.post("/api/digit", json={"pixels": pixels})
    client.post("/api/digit", json={"pixels": pixels})
    assert len(loaded) == 1


def test_api_route_errors_are_json(client):
    response = client.get("/api/digit")
    assert response.status_code == 405
    assert response.get_json()["error"]["code"] == "http_error"


def test_figure_route_cannot_escape_directory(client):
    response = client.get("/reports/figures/../evaluation.json")
    assert response.status_code == 404
