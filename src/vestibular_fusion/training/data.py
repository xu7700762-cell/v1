from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ..data import city, monifeixing, vrq
from ..data.types import FeatureBank
from ..evaluation.io import read_csv, read_json
from ..evaluation.metrics import subject_sort_key


@dataclass(frozen=True)
class SeverityExample:
    subject_id: str
    reference_session: str
    task_session: str
    label: int
    weight: float = 1.0


def weighted_examples(
    examples: list[SeverityExample], *, path_weighting: bool
) -> tuple[SeverityExample, ...]:
    """Match the historical source-severity weighting protocol."""
    if not examples:
        raise ValueError("Severity weighting requires at least one example")
    labels = np.asarray([int(example.label) for example in examples], dtype=np.int64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Severity weighting requires both source classes")
    weights = np.ones(len(examples), dtype=np.float64)
    if path_weighting:
        counts = {
            subject: sum(example.subject_id == subject for example in examples)
            for subject in {example.subject_id for example in examples}
        }
        weights = np.asarray(
            [1.0 / counts[example.subject_id] for example in examples], dtype=np.float64
        )
    for label in (0, 1):
        mask = labels == label
        weights[mask] *= 0.5 / weights[mask].sum()
    weights *= len(weights) / weights.sum()
    return tuple(
        SeverityExample(
            example.subject_id,
            example.reference_session,
            example.task_session,
            example.label,
            float(weight),
        )
        for example, weight in zip(examples, weights)
    )


@dataclass(frozen=True)
class FoldProtocol:
    source_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]
    calibration_train_subjects: tuple[str, ...]
    calibration_val_subjects: tuple[str, ...]
    source_examples: tuple[SeverityExample, ...]
    test_examples: tuple[SeverityExample, ...]

    def __post_init__(self) -> None:
        source = set(self.source_subjects)
        test = set(self.test_subjects)
        calibration_train = set(self.calibration_train_subjects)
        calibration_val = set(self.calibration_val_subjects)
        if source & test:
            raise ValueError("Source and outer-test subjects must be identity-disjoint")
        if calibration_train & calibration_val:
            raise ValueError("Calibration-train and calibration-val subjects must be disjoint")
        if calibration_train | calibration_val != source:
            raise ValueError("Source-only calibration subjects must partition outer source")


@dataclass
class TrainingDataset:
    bank: FeatureBank
    folds: dict[str, FoldProtocol]


def _select_examples(
    examples: list[SeverityExample], subjects: list[str] | tuple[str, ...]
) -> tuple[SeverityExample, ...]:
    selected = set(subjects)
    return tuple(example for example in examples if example.subject_id in selected)


def _load_monifeixing(config: dict) -> TrainingDataset:
    root = (
        Path(config["paths"]["asset_root"])
        / "vr_ssq_regression"
        / "artifacts_fair_joint_lambda0p3"
        / "monifeixing"
        / "lambda0p3"
        / "seed42"
        / "full"
    )
    report = read_json(root / "report.json")
    labels = {
        str(row["subject_id"]): int(row["y_true"])
        for row in read_csv(root / "severity_predictions.csv")
    }
    examples = [
        SeverityExample(subject, "rest1", "rest2", labels[subject])
        for subject in sorted(labels, key=subject_sort_key)
    ]
    folds = {}
    for fold_id, split in report["identity_audit"]["folds"].items():
        source = tuple(str(subject) for subject in split["source_outer_train_subjects"])
        test = tuple(str(subject) for subject in split["test_subjects"])
        folds[fold_id] = FoldProtocol(
            source_subjects=source,
            test_subjects=test,
            calibration_train_subjects=tuple(
                str(subject) for subject in split["source_train_subjects"]
            ),
            calibration_val_subjects=tuple(
                str(subject) for subject in split["source_val_subjects"]
            ),
            source_examples=_select_examples(examples, source),
            test_examples=_select_examples(examples, test),
        )
    bank = monifeixing.build_raw_bank(
        Path(config["paths"]["monifeixing_data_root"]),
        Path(config["paths"]["monifeixing_initial_encoder"]),
        torch.device("cpu"),
    )
    return TrainingDataset(bank=bank, folds=folds)


def _load_vrq(config: dict, device: torch.device) -> TrainingDataset:
    root = (
        Path(config["paths"]["asset_root"])
        / "vr_ssq_regression"
        / "artifacts_fair_joint_lambda0p3"
        / "vrq"
        / "seed_42"
        / "main"
        / "full"
    )
    manifest = read_json(root / "audit_manifest.json")
    payload = manifest["run_fingerprint_payload"]
    args = SimpleNamespace(
        pretrain_ckpt=str(config["paths"]["pretrain_checkpoint"]),
        data_root=str(config["paths"]["vrq_data_root"]),
        mat_key=payload["mat_key"],
        encoder_backend="native",
        ea_mode="subject_unlabeled",
        encode_batch_size=64,
        record_storage_dtype=np.float16,
    )
    protocols = [vrq.SubjectProtocol(**row) for row in manifest["subject_protocols"]]
    task_sessions = {row.subject_id: row.final_task for row in protocols}
    examples = [
        SeverityExample(
            subject,
            "rest01",
            task_sessions[subject],
            int(metadata["ssq_label"]),
        )
        for subject, metadata in sorted(
            manifest["audit"]["subjects"].items(), key=lambda item: subject_sort_key(item[0])
        )
        if subject in task_sessions
    ]
    folds = {}
    for fold_id, split in manifest["folds"].items():
        source = tuple(
            sorted(split["train_subjects"] + split["val_subjects"], key=subject_sort_key)
        )
        test = tuple(str(subject) for subject in split["test_subjects"])
        folds[fold_id] = FoldProtocol(
            source_subjects=source,
            test_subjects=test,
            calibration_train_subjects=tuple(
                str(subject) for subject in split["train_subjects"]
            ),
            calibration_val_subjects=tuple(
                str(subject) for subject in split["val_subjects"]
            ),
            source_examples=_select_examples(examples, source),
            test_examples=_select_examples(examples, test),
        )
    bank = vrq.build_feature_bank(args, device, manifest["audit"], protocols)
    return TrainingDataset(bank=bank, folds=folds)


def _load_city(config: dict, device: torch.device) -> TrainingDataset:
    root = (
        Path(config["paths"]["asset_root"])
        / "vr_ssq_regression"
        / "artifacts_city_a3_lambda_sweep_strict"
        / "audit"
        / "audit_manifest.json"
    )
    manifest = read_json(root)
    audit = copy.deepcopy(manifest["audit"])
    data_root = Path(config["paths"]["city_data_root"])
    for metadata in audit["subjects"].values():
        if metadata.get("included"):
            metadata["mat_path"] = str(data_root / Path(metadata["mat_path"]).name)
    args = SimpleNamespace(
        pretrain_ckpt=str(config["paths"]["pretrain_checkpoint"]),
        encoder_backend="native",
        mat_key="data256",
        ea_mode="subject_unlabeled",
        encode_batch_size=64,
        record_storage_dtype=np.float16,
    )
    aliases = {}
    for subject, metadata in audit["subjects"].items():
        for segment in metadata.get("segments", []):
            if segment.get("path_score") is not None:
                aliases[(subject, int(segment["route_order"]))] = city.session_alias(
                    segment, metadata["anchor_session"]
                )
    examples = [
        SeverityExample(
            str(row["subject_id"]),
            "rest01",
            aliases[(str(row["subject_id"]), int(row["route_order"]))],
            int(row["path_label"]),
        )
        for row in audit["path_labels"]
    ]
    folds = {}
    for fold_id, split in manifest["fold_manifest"]["folds"].items():
        source = tuple(
            sorted(split["train_subjects"] + split["val_subjects"], key=subject_sort_key)
        )
        test = tuple(str(subject) for subject in split["test_subjects"])
        folds[fold_id] = FoldProtocol(
            source_subjects=source,
            test_subjects=test,
            calibration_train_subjects=tuple(
                str(subject) for subject in split["train_subjects"]
            ),
            calibration_val_subjects=tuple(
                str(subject) for subject in split["val_subjects"]
            ),
            source_examples=_select_examples(examples, source),
            test_examples=_select_examples(examples, test),
        )
    bank = city.build_feature_bank(args, device, audit)
    return TrainingDataset(bank=bank, folds=folds)


def load_training_dataset(
    config: dict, dataset: str, device: torch.device
) -> TrainingDataset:
    if dataset == "monifeixing":
        return _load_monifeixing(config)
    if dataset == "vrq":
        return _load_vrq(config, device)
    if dataset == "city":
        return _load_city(config, device)
    raise ValueError(f"Unsupported dataset: {dataset}")
