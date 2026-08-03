from pathlib import Path

import numpy as np
import pytest

from vestibular_fusion.cli import build_parser
from vestibular_fusion.data.monifeixing import fit_prototypes
from vestibular_fusion.data.types import AuditMetadata, FeatureBank, SubjectRecord
from vestibular_fusion.data.vrq import choose_score_threshold, select_fixed_view_configs
from vestibular_fusion.evaluation.trained import (
    FOLD_IDS,
    _checkpoint_paths,
    _validate_checkpoint_metadata,
)
from vestibular_fusion.training.data import FoldProtocol


def _protocol() -> FoldProtocol:
    return FoldProtocol(
        source_subjects=("source_train", "source_val"),
        test_subjects=("test_1",),
        calibration_train_subjects=("source_train",),
        calibration_val_subjects=("source_val",),
        source_examples=(),
        test_examples=(),
    )


def _payload() -> dict:
    return {
        "checkpoint_schema": "trained_fold_refit_v1",
        "dataset": "vrq",
        "fold_id": "fold_1",
        "training_seed": 1001,
        "severity_weight": 0.3,
        "severity_windows_per_session": 5,
        "encoder_frozen": True,
        "refit_protocol": "source_validation_selection_then_source_refit",
        "best_epoch": 3,
        "source_subjects": ["source_train", "source_val"],
        "test_subjects": ["test_1"],
        "model_state_dict": {},
        "severity_head_state_dict": {},
    }


def test_trained_checkpoint_metadata_requires_complete_training_checkpoint():
    payload = _payload()
    _validate_checkpoint_metadata(payload, "vrq", "fold_1", _protocol())
    del payload["severity_head_state_dict"]
    with pytest.raises(RuntimeError, match="severity_head_state_dict"):
        _validate_checkpoint_metadata(payload, "vrq", "fold_1", _protocol())


def test_fold_protocol_requires_source_only_disjoint_calibration_partition():
    with pytest.raises(ValueError, match="partition outer source"):
        FoldProtocol(
            source_subjects=("source_1", "source_2"),
            test_subjects=("test_1",),
            calibration_train_subjects=("source_1",),
            calibration_val_subjects=(),
            source_examples=(),
            test_examples=(),
        )


def test_checkpoint_root_requires_all_five_folds(tmp_path: Path):
    for fold_id in FOLD_IDS:
        path = tmp_path / fold_id / "checkpoint.pt"
        path.parent.mkdir()
        path.touch()
    assert set(_checkpoint_paths(tmp_path)) == set(FOLD_IDS)
    (tmp_path / "fold_5" / "checkpoint.pt").unlink()
    with pytest.raises(FileNotFoundError, match="fold_1..fold_5"):
        _checkpoint_paths(tmp_path)


def test_a1_threshold_selection_is_source_only_and_deterministic():
    source_rows = [
        {"subject_id": "s1", "y_true": 0, "mambakan_score": 0.20},
        {"subject_id": "s1", "y_true": 1, "mambakan_score": 0.60},
        {"subject_id": "s2", "y_true": 0, "mambakan_score": 0.35},
        {"subject_id": "s2", "y_true": 1, "mambakan_score": 0.55},
    ]
    first = choose_score_threshold(source_rows)
    second = choose_score_threshold(list(reversed(source_rows)))
    assert first == second
    assert 0.35 < first[0] <= 0.55


def test_monifeixing_prototypes_use_locked_float64_normalization():
    class Record:
        labels = np.asarray([0, 0, 1, 1])

    class Bank:
        records = {"s1": Record(), "s2": Record()}

    embeddings = {
        "s1": np.asarray(
            [[1.0, 1e-4], [1.0, 2e-4], [1.0, 3e-4], [1.0, 4e-4]],
            dtype=np.float32,
        ),
        "s2": np.asarray(
            [[1.0, 5e-4], [1.0, 6e-4], [1.0, 7e-4], [1.0, 8e-4]],
            dtype=np.float32,
        ),
    }
    actual = fit_prototypes(embeddings, Bank(), ["s1", "s2"])
    centers = np.asarray(
        [
            np.stack(
                [values[Bank.records[subject].labels == label].mean(axis=0) for label in (0, 1)]
            )
            for subject, values in embeddings.items()
        ],
        dtype=np.float64,
    )
    centers /= np.linalg.norm(centers, axis=-1, keepdims=True)
    expected = centers.mean(axis=0)
    expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
    np.testing.assert_array_equal(actual, expected.astype(np.float32))


def test_cli_exposes_trained_evaluate_command():
    args = build_parser().parse_args(
        [
            "evaluate",
            "--config",
            "configs/paths.local.json",
            "--dataset",
            "vrq",
            "--checkpoint-root",
            "outputs/training/vrq",
        ]
    )
    assert args.command == "evaluate"
    assert args.dataset == "vrq"


def test_fixed_view_selection_is_source_only_and_deterministic():
    from types import SimpleNamespace

    records = {}
    samples = []
    features = {family: {} for family in ("spectral_topography", "covariance_tangent")}
    sample_index = 0
    for subject_index, subject in enumerate(("s1", "s2", "s3", "s4")):
        labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
        values = np.asarray(
            [[float(label), float(subject_index), float(window)] for window, label in enumerate(labels)],
            dtype=np.float64,
        )
        records[subject] = SubjectRecord(
            windows=values.astype(np.float32),
            tokens=values.astype(np.float32),
            labels=labels,
            sessions=["rest01"] * 3 + ["task"] * 3,
        )
        for local_index in range(len(labels)):
            samples.append(
                SimpleNamespace(
                    sample_index=sample_index,
                    subject_id=subject,
                    local_index=local_index,
                )
            )
            sample_index += 1
        for family in features:
            features[family][subject] = values.copy()
    bank = FeatureBank(
        records=records,
        samples=samples,
        encoder_state={},
        audit=AuditMetadata({}, "frozen", {}),
    )

    first = select_fixed_view_configs(features, bank, ["s1", "s2"], ["s3", "s4"], 1001)
    second = select_fixed_view_configs(features, bank, ["s1", "s2"], ["s3", "s4"], 1001)

    assert first == second
    configs, report = first
    assert set(configs) == {"spectral_topography", "covariance_tangent"}
    assert report["selection_source"] == "v1_raw_features_source_train_validation"
    assert all(item["candidate_count"] == 24 for item in report["families"].values())
