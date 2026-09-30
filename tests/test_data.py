"""Test split guarantees using synthetic labels, without opening held-out data."""

import json

import numpy as np
import pytest

from src.common import dataset_sha256
from src.data import create_split_indices, load_prepared_data


@pytest.fixture
def prepared_cache(tmp_path):
    """Synthetic cache with valid Digits schema, never a real held-out image."""
    features = np.resize(np.arange(17, dtype=float), 1797 * 64).reshape(1797, 64)
    labels = np.resize(np.arange(10, dtype=np.int64), 1797)
    splits = create_split_indices(labels, seed=42)
    checksum = dataset_sha256(features, labels)
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    np.savez_compressed(raw / "digits.npz", X=features, y=labels)
    manifest = {"sha256": checksum, "shape": [1797, 64]}
    split_payload = {
        "seed": 42,
        "strategy": "stratified60/20/20",
        "dataset_sha256": checksum,
        **{name: values.tolist() for name, values in splits.items()},
    }
    (tmp_path / "data" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "data" / "splits.json").write_text(json.dumps(split_payload), encoding="utf-8")
    return tmp_path, features, labels


def test_split_is_a_complete_disjoint_partition_with_expected_sizes():
    labels = np.resize(np.arange(10), 1797)
    splits = create_split_indices(labels, seed=42)
    assert set(splits) == {"train", "validation", "test"}
    all_indices = []
    for name, expected_fraction in [("train", 0.6), ("validation", 0.2), ("test", 0.2)]:
        indices = np.asarray(splits[name])
        assert indices.ndim == 1
        assert np.issubdtype(indices.dtype, np.integer)
        assert abs(len(indices) - len(labels) * expected_fraction) <= 2
        assert np.all((0 <= indices) & (indices < len(labels)))
        all_indices.extend(indices.tolist())
    assert len(all_indices) == len(labels)
    assert len(set(all_indices)) == len(labels)
    assert set(all_indices) == set(range(len(labels)))


def test_split_stratifies_each_class_on_unbalanced_labels():
    labels = np.concatenate([np.full(40 + digit * 7, digit) for digit in range(10)])
    splits = create_split_indices(labels, seed=42)
    global_fractions = np.bincount(labels, minlength=10) / len(labels)
    for indices in splits.values():
        counts = np.bincount(labels[np.asarray(indices)], minlength=10)
        assert np.all(counts > 0)
        np.testing.assert_allclose(counts / counts.sum(), global_fractions, atol=0.015, rtol=0)


def test_same_seed_repeats_split_and_changed_seed_changes_membership():
    labels = np.resize(np.arange(10), 1000)
    first = create_split_indices(labels, seed=42)
    repeat = create_split_indices(labels, seed=42)
    changed = create_split_indices(labels, seed=43)
    for name in first:
        np.testing.assert_array_equal(first[name], repeat[name])
    assert set(first["test"]) != set(changed["test"])


def test_prepared_cache_loads_verified_arrays_and_partition(prepared_cache):
    root, expected_features, expected_labels = prepared_cache
    features, labels, splits, manifest = load_prepared_data(root)
    np.testing.assert_array_equal(features, expected_features)
    np.testing.assert_array_equal(labels, expected_labels)
    assert sum(map(len, splits.values())) == len(labels)
    assert manifest["sha256"] == dataset_sha256(features, labels)


def test_prepared_cache_rejects_changed_pixel_without_updated_checksum(prepared_cache):
    root, features, labels = prepared_cache
    changed = features.copy()
    changed[0, 0] = 1
    np.savez_compressed(root / "data" / "raw" / "digits.npz", X=changed, y=labels)
    with pytest.raises(ValueError):
        load_prepared_data(root)


def test_prepared_cache_rejects_overlapping_split_membership(prepared_cache):
    root, _, _ = prepared_cache
    split_path = root / "data" / "splits.json"
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    payload["validation"][0] = payload["train"][0]
    split_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_prepared_data(root)
