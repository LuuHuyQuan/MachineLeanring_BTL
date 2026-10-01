"""Protect test isolation without training or scoring the real project test set."""

import pytest
import numpy as np

from src import evaluate as evaluation_module
from src import train as training_module
from src.common import file_sha256, read_json, write_json


def _forbidden(*_args, **_kwargs):
    raise AssertionError("A refusal guard must run before model loading, data access, or scoring.")


@pytest.mark.parametrize("force,reason", [
    (False, None), (False, "repeat requested"), (True, None), (True, ""), (True, "   "),
])
def test_existing_test_report_refuses_evaluation_before_loading(tmp_path, monkeypatch, force, reason):
    destination = tmp_path / "reports/evaluation.json"
    destination.parent.mkdir()
    # Even a damaged report records prior test exposure; it must not trigger a rerun.
    destination.write_text("not valid JSON", encoding="utf-8")
    monkeypatch.setattr(evaluation_module, "read_json", _forbidden)
    monkeypatch.setattr(evaluation_module.joblib, "load", _forbidden)
    monkeypatch.setattr(evaluation_module, "load_prepared_data", _forbidden)
    monkeypatch.setattr(evaluation_module, "score_model", _forbidden)

    with pytest.raises(RuntimeError, match="Final test already evaluated"):
        evaluation_module.evaluate(tmp_path, force=force, reason=reason)

    assert destination.read_text(encoding="utf-8") == "not valid JSON"
    assert not (tmp_path / "reports/archive").exists()


@pytest.mark.parametrize("force,reason", [
    (False, None), (False, "retrain requested"), (True, None), (True, ""), (True, "   "),
])
def test_retraining_after_test_requires_explicit_audit_before_data_access(tmp_path, monkeypatch, force, reason):
    destination = tmp_path / "reports/evaluation.json"
    destination.parent.mkdir()
    destination.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(training_module, "prepare_data", _forbidden)
    monkeypatch.setattr(training_module, "load_prepared_data", _forbidden)
    monkeypatch.setattr(training_module, "make_baseline", _forbidden)
    monkeypatch.setattr(training_module, "archive_previous_run", _forbidden)

    with pytest.raises(RuntimeError, match="Test results already exist"):
        training_module.train(tmp_path, force=force, reason=reason)

    assert destination.read_text(encoding="utf-8") == "{}"
    assert not (tmp_path / "models").exists()


@pytest.fixture
def frozen_artifacts(tmp_path):
    """Only opaque bytes and hashes are required to test the pre-score guards."""
    model_folder = tmp_path / "models"
    model_folder.mkdir()
    (tmp_path / "data").mkdir()
    (model_folder / "digit_pipeline.joblib").write_bytes(b"frozen-mlp")
    (model_folder / "baseline_pipeline.joblib").write_bytes(b"frozen-baseline")
    (tmp_path / "data/splits.json").write_text("{}", encoding="utf-8")
    write_json(model_folder / "model_metadata.json", {
        "model_sha256": file_sha256(model_folder / "digit_pipeline.joblib"),
        "baseline_sha256": file_sha256(model_folder / "baseline_pipeline.joblib"),
        "split_sha256": file_sha256(tmp_path / "data/splits.json"),
        "dataset_sha256": "not-needed-before-refusal",
    })
    return tmp_path


@pytest.mark.parametrize("artifact", ["digit_pipeline.joblib", "baseline_pipeline.joblib"])
def test_tampered_model_is_rejected_before_deserialization_and_test_scoring(frozen_artifacts, monkeypatch, artifact):
    (frozen_artifacts / "models" / artifact).write_bytes(b"changed-after-validation-selection")
    monkeypatch.setattr(evaluation_module.joblib, "load", _forbidden)
    monkeypatch.setattr(evaluation_module, "load_prepared_data", _forbidden)
    monkeypatch.setattr(evaluation_module, "score_model", _forbidden)

    with pytest.raises(ValueError, match="Model checksum changed after selection"):
        evaluation_module.evaluate(frozen_artifacts)

    assert not (frozen_artifacts / "reports/evaluation.json").exists()


def test_changed_split_is_rejected_before_data_access_or_test_scoring(frozen_artifacts, monkeypatch):
    (frozen_artifacts / "data/splits.json").write_text('{"changed":true}', encoding="utf-8")
    monkeypatch.setattr(evaluation_module.joblib, "load", _forbidden)
    monkeypatch.setattr(evaluation_module, "load_prepared_data", _forbidden)
    monkeypatch.setattr(evaluation_module, "score_model", _forbidden)

    with pytest.raises(ValueError, match="Frozen split changed after model selection"):
        evaluation_module.evaluate(frozen_artifacts)

    assert not (frozen_artifacts / "reports/evaluation.json").exists()


@pytest.mark.parametrize("events", [[], [{"status": "started", "accessed_at": "initial-access"}]])
def test_existing_access_marker_refuses_evaluation_when_result_is_missing(tmp_path, monkeypatch, events):
    write_json(tmp_path / "data/test_access.json", {"events": events})
    monkeypatch.setattr(evaluation_module.joblib, "load", _forbidden)
    monkeypatch.setattr(evaluation_module, "load_prepared_data", _forbidden)
    monkeypatch.setattr(evaluation_module, "score_model", _forbidden)

    with pytest.raises(RuntimeError, match="Final test already evaluated"):
        evaluation_module.evaluate(tmp_path)

    assert read_json(tmp_path / "data/test_access.json")["events"] == events
    assert not (tmp_path / "reports/evaluation.json").exists()


def test_existing_access_marker_refuses_retraining_when_result_is_missing(tmp_path, monkeypatch):
    write_json(tmp_path / "data/test_access.json", {"events": [{"status": "started"}]})
    monkeypatch.setattr(training_module, "prepare_data", _forbidden)
    monkeypatch.setattr(training_module, "load_prepared_data", _forbidden)
    monkeypatch.setattr(training_module, "archive_previous_run", _forbidden)

    with pytest.raises(RuntimeError, match="Test results already exist"):
        training_module.train(tmp_path)

    assert read_json(tmp_path / "data/test_access.json")["events"] == [{"status": "started"}]


def test_failed_forced_evaluation_preserves_access_history(frozen_artifacts, monkeypatch):
    marker = frozen_artifacts / "data/test_access.json"
    write_json(marker, {"events": [{"status": "started", "accessed_at": "initial-access"}]})
    # Only two synthetic rows are accessed; no real project data or scores are used.
    monkeypatch.setattr(evaluation_module, "load_prepared_data", lambda _root: (
        np.zeros((2, 64)), np.asarray([0, 1]), {"test": np.asarray([0, 1])},
        {"sha256": "not-needed-before-refusal"},
    ))
    def fail_deserialization(_path):
        raise RuntimeError("simulated deserialization failure")
    monkeypatch.setattr(evaluation_module.joblib, "load", fail_deserialization)
    monkeypatch.setattr(evaluation_module, "score_model", _forbidden)

    with pytest.raises(RuntimeError, match="simulated deserialization failure"):
        evaluation_module.evaluate(frozen_artifacts, force=True, reason="verify failure audit")

    events = read_json(marker)["events"]
    assert len(events) == 2
    assert events[0]["accessed_at"] == "initial-access"
    assert events[1]["status"] == "started"
    assert events[1]["repeat_reason"] == "verify failure audit"
    assert not (frozen_artifacts / "reports/evaluation.json").exists()
    # A failed repeat must not silently permit the next default evaluation.
    with pytest.raises(RuntimeError, match="Final test already evaluated"):
        evaluation_module.evaluate(frozen_artifacts)


def test_failed_forced_retraining_preserves_access_history(frozen_artifacts, monkeypatch):
    marker = frozen_artifacts / "data/test_access.json"
    write_json(marker, {"events": [{"status": "completed"}]})
    write_json(frozen_artifacts / "reports/evaluation.json", {"test": "already inspected"})
    def fail_training_data(_root):
        raise RuntimeError("simulated cached data failure")
    monkeypatch.setattr(training_module, "load_prepared_data", fail_training_data)
    monkeypatch.setattr(training_module, "make_baseline", _forbidden)

    with pytest.raises(RuntimeError, match="simulated cached data failure"):
        training_module.train(frozen_artifacts, force=True, reason="verify retraining failure audit")

    assert read_json(marker)["events"] == [{"status": "completed"}]
    assert not (frozen_artifacts / "reports/evaluation.json").exists()
    assert list((frozen_artifacts / "reports/archive").glob("*/audit.json"))
    # Archiving the old result before a failure must not erase the refusal guard.
    with pytest.raises(RuntimeError, match="Test results already exist"):
        training_module.train(frozen_artifacts)
