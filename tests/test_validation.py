from pathlib import Path

import pytest

from biofoundation_v1.evaluation.io import sha256_file
from biofoundation_v1.evaluation.metrics import binary_metrics
from biofoundation_v1.preflight import _check_fivefold_subject_split, _require_hash


def _valid_folds():
    return {
        f"fold_{index}": {
            "train_subjects": [f"train_{index}"],
            "val_subjects": [f"val_{index}"],
            "test_subjects": [f"test_{index}"],
        }
        for index in range(1, 6)
    }


def test_fivefold_subject_split_is_complete_and_disjoint():
    _check_fivefold_subject_split({"folds": _valid_folds()}, "synthetic manifest")


def test_fivefold_subject_overlap_fails_loudly():
    folds = _valid_folds()
    folds["fold_2"]["test_subjects"] = ["test_1"]
    with pytest.raises(RuntimeError, match="test subjects overlap"):
        _check_fivefold_subject_split({"folds": folds}, "synthetic manifest")


def test_binary_metrics_use_predictions_and_scores():
    metrics = binary_metrics(
        [
            {"y_true": 0, "y_pred": 0, "score": 0.1, "subject_id": "s1"},
            {"y_true": 1, "y_pred": 1, "score": 0.9, "subject_id": "s2"},
        ]
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["confusion_matrix"] == [[1, 0], [0, 1]]


def test_sha256_validation_fails_on_changed_asset(tmp_path: Path):
    path = tmp_path / "asset.bin"
    path.write_bytes(b"locked")
    expected = sha256_file(path)
    _require_hash(path, expected, "synthetic asset")
    path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _require_hash(path, expected, "synthetic asset")
