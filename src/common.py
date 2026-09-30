"""Small shared helpers for portable artifact paths and audit metadata."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _serialize(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                   default=_serialize, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_sha256(X, y) -> str:
    """Hash canonical array bytes, independent of NPZ compression metadata."""
    digest = hashlib.sha256()
    digest.update(np.asarray(X, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(y, dtype="<i8").tobytes(order="C"))
    return digest.hexdigest()
