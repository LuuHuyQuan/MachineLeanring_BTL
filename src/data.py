"""Load bundled Digits, audit its quality, freeze splits and explore TRAIN only."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from src.common import PROJECT_ROOT, SEED, dataset_sha256, read_json, timestamp, write_json


def create_split_indices(y, seed=SEED):
    indices = np.arange(len(y))
    train_val, test = train_test_split(indices, test_size=0.2, stratify=y, random_state=seed)
    train, validation = train_test_split(train_val, test_size=0.25,
                                       stratify=np.asarray(y)[train_val], random_state=seed)
    return {"train": train, "validation": validation, "test": test}


def load_dataset():
    digits = load_digits()
    X, y = digits.data.astype(np.float64), digits.target.astype(np.int64)
    if X.shape != (1797, 64) or y.shape != (1797,) or not np.isfinite(X).all():
        raise ValueError("Digits schema or dataset size differs from the frozen specification.")
    if X.min() < 0 or X.max() > 16 or set(y) != set(range(10)):
        raise ValueError("Digits values violate the documented schema.")
    return X, y


def load_prepared_data(root=PROJECT_ROOT):
    root = Path(root)
    raw_path = root / "data/raw/digits.npz"
    if not raw_path.exists() or not (root / "data/splits.json").exists():
        raise FileNotFoundError("Run python -m src.data before training or evaluation.")
    with np.load(raw_path, allow_pickle=False) as raw:
        X, y = raw["X"], raw["y"]
    manifest = read_json(root / "data/manifest.json")
    split_payload = read_json(root / "data/splits.json")
    checksum = dataset_sha256(X, y)
    if checksum != manifest["sha256"] or checksum != split_payload["dataset_sha256"]:
        raise ValueError("Dataset checksum differs from the frozen split.")
    if X.shape != (1797, 64) or y.shape != (1797,) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("Cached dataset violates its frozen schema.")
    if not np.isfinite(X).all() or X.min() < 0 or X.max() > 16 or set(y) != set(range(10)):
        raise ValueError("Cached dataset has invalid pixels or labels.")
    splits = {}
    for key in ("train", "validation", "test"):
        indices = np.asarray(split_payload[key])
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or not len(indices):
            raise ValueError("Split indices must be nonempty one-dimensional integer arrays.")
        splits[key] = indices.astype(np.int64)
    merged = np.concatenate(list(splits.values()))
    if len(merged) != len(y) or not np.array_equal(np.sort(merged), np.arange(len(y))):
        raise ValueError("Splits must be disjoint and cover every sample exactly once.")
    return X, y, splits, manifest


def prepare_data(root=PROJECT_ROOT, seed=SEED):
    root = Path(root)
    X, y = load_dataset()
    checksum = dataset_sha256(X, y)
    splits = create_split_indices(y, seed)
    frozen_path = root / "data/splits.json"
    if frozen_path.exists():
        frozen = read_json(frozen_path)
        if frozen["seed"] != seed or frozen["dataset_sha256"] != checksum:
            raise ValueError("An existing frozen split differs. Use a separate project directory for a new study.")
    raw_path = root / "data/raw/digits.npz"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw_path, X=X, y=y)
    if not frozen_path.exists():
        write_json(frozen_path, {"seed": seed, "dataset_sha256": checksum,
                                "strategy": "stratified60/20/20", **splits})
    manifest_path = root / "data/manifest.json"
    if not manifest_path.exists():
        write_json(manifest_path, {
            "name": "scikit-learn Digits", "loaded_at": timestamp(), "source_access_date": "2026-09-30",
            "source": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
            "original_source": "https://archive.ics.uci.edu/dataset/80/optical%2Brecognition%2Bof%2Bhandwritten%2Bdigits",
            "license": "CC BY4.0 (UCI)", "sklearn_version": sklearn.__version__,
            "shape": list(X.shape), "sha256": checksum, "hash_format": "float64 little-endian X + int64 little-endian y",
            "acquisition": "sklearn.datasets.load_digits bundled dataset; no external download required",
            "files": ["data/raw/digits.npz", "data/splits.json"],
        })
    # Reuse the persisted partition rather than recomputing EDA/demo membership.
    X, y, splits, _ = load_prepared_data(root)
    _, counts = np.unique(X, axis=0, return_counts=True)
    quality = {
        "rows": len(y), "features": 64, "missing_values": int(np.isnan(X).sum()),
        "duplicate_images": int(np.sum(counts - 1)), "out_of_range_pixels": int(((X < 0) | (X > 16)).sum()),
        "pixel_min": float(X.min()), "pixel_max": float(X.max()), "removed_rows": [],
        "class_counts": {str(i): int((y == i).sum()) for i in range(10)},
        "split_class_counts": {name: {str(i): int((y[idx] == i).sum()) for i in range(10)}
                               for name, idx in splits.items()},
        "decisions": ["Keep all valid images; do not learn preprocessing from validation or test.",
                      "Use constant /16 scaling from dataset schema; stratify because ten classes have slightly different counts.",
                      "No writer IDs available in the sklearn subset; writer-disjoint generalization is not demonstrated."],
    }
    write_json(root / "reports/data_quality.json", quality)
    train = splits["train"]
    demo = [{"id": int(train[np.flatnonzero(y[train] == digit)[0]]), "label": digit,
             "pixels": X[train[np.flatnonzero(y[train] == digit)[0]]].reshape(8, 8)} for digit in range(10)]
    write_json(root / "data/demo_samples.json", demo)
    figures = root / "reports/figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(10), [(y[train] == digit).sum() for digit in range(10)], color="#2869c8")
    ax.set(xlabel="Digit class", ylabel="Training samples", title="Class distribution — TRAIN only", xticks=range(10))
    fig.tight_layout()
    fig.savefig(figures / "class_distribution.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 5, figsize=(9, 4))
    for item, ax in zip(demo, axes.flat):
        ax.imshow(item["pixels"], cmap="gray", vmin=0, vmax=16)
        ax.set_title(f"Digit {item['label']} · ID{item['id']}")
        ax.axis("off")
    fig.suptitle("Demo images — TRAIN only")
    fig.tight_layout()
    fig.savefig(figures / "train_samples.png", dpi=160)
    plt.close(fig)
    print(f"Digits ready: {len(y)} images; " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    return quality


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    prepare_data(args.root, args.seed)


if __name__ == "__main__":
    main()
