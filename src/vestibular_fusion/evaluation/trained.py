from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from ..data import monifeixing, vrq
from ..model.a1 import DirectionalMambaKAN, load_checkpoint_state_dict
from ..model.encoder import TemporalEncoder
from ..training.data import FoldProtocol, load_training_dataset
from .fusion import source_r4_rows, state_rows, uniform_anchor_mask
from .io import read_csv, read_json, sha256_file, write_csv, write_json, write_npz
from .metrics import binary_metrics, subject_sort_key
from .runner import _asset_roots, _assert_local_modules, _source_raw
from .severity import (
    attach_subject_labels,
    evaluate_city_r4,
    fit_source_head,
    score_outer,
    subject_r4_features,
)


FOLD_IDS = tuple(f"fold_{index}" for index in range(1, 6))


def _checkpoint_paths(checkpoint_root: Path) -> dict[str, Path]:
    paths = {
        fold_id: Path(checkpoint_root) / fold_id / "checkpoint.pt" for fold_id in FOLD_IDS
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Trained evaluation requires fold_1..fold_5 checkpoints: " + ", ".join(missing)
        )
    return paths


def _assert_empty_output(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Trained evaluation refuses a non-empty output directory: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)


def _validate_checkpoint_metadata(
    payload: dict, dataset: str, fold_id: str, protocol: FoldProtocol
) -> None:
    expected = {
        "checkpoint_schema": "trained_fold_refit_v1",
        "dataset": dataset,
        "fold_id": fold_id,
        "training_seed": 1001,
        "severity_weight": 0.3,
        "severity_windows_per_session": 5,
        "encoder_frozen": True,
        "refit_protocol": "source_validation_selection_then_source_refit",
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"{fold_id} checkpoint metadata mismatch: {mismatches}")
    if tuple(payload.get("source_subjects", ())) != protocol.source_subjects:
        raise RuntimeError(f"{fold_id} checkpoint source subjects changed")
    if tuple(payload.get("test_subjects", ())) != protocol.test_subjects:
        raise RuntimeError(f"{fold_id} checkpoint test subjects changed")
    if not isinstance(payload.get("model_state_dict"), dict):
        raise RuntimeError(f"{fold_id} checkpoint has no model_state_dict")
    if not isinstance(payload.get("severity_head_state_dict"), dict):
        raise RuntimeError(f"{fold_id} checkpoint has no severity_head_state_dict")
    best_epoch = payload.get("best_epoch")
    if not isinstance(best_epoch, int) or best_epoch <= 0:
        raise RuntimeError(f"{fold_id} checkpoint has no positive best_epoch")


def _load_trained_a1(
    checkpoint_path: Path,
    dataset: str,
    fold_id: str,
    protocol: FoldProtocol,
    device: torch.device,
) -> tuple[DirectionalMambaKAN, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _validate_checkpoint_metadata(payload, dataset, fold_id, protocol)
    model = DirectionalMambaKAN.from_model_spec(payload.get("model_spec")).to(device)
    load_checkpoint_state_dict(
        model, payload["model_state_dict"], source=str(checkpoint_path)
    )
    model.eval()
    return model, payload


def _mark_target_anchors(
    rows: list[dict], subjects: tuple[str, ...], anchor_session: str
) -> None:
    selected = set(subjects)
    for subject in subjects:
        indices = [
            index
            for index, row in enumerate(rows)
            if row["subject_id"] == subject and row["subject_id"] in selected
        ]
        mask = uniform_anchor_mask([rows[index] for index in indices], anchor_session)
        for index, is_anchor in zip(indices, mask):
            rows[index]["calibration_anchor"] = int(is_anchor)


def _aligned_moments(rows: list[dict], moments: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack(
        [moments[row["subject_id"]][int(row["local_index"])] for row in rows]
    ).astype(np.float64)


def _trained_deep_views(
    model: DirectionalMambaKAN,
    bank,
    protocol: FoldProtocol,
    fold_id: str,
    dataset: str,
    device: torch.device,
) -> tuple[list[dict], list[dict], dict[str, np.ndarray]]:
    subjects = list(protocol.source_subjects + protocol.test_subjects)
    if dataset == "monifeixing":
        embeddings, moments = monifeixing.extract_views(model, bank, subjects, device, 128)
    else:
        embeddings, moments = vrq.extract_views(model, bank, subjects, device, 128)
    fit_prototypes = (
        monifeixing.fit_prototypes if dataset == "monifeixing" else vrq.fit_prototypes
    )
    prototypes = fit_prototypes(embeddings, bank, list(protocol.calibration_train_subjects))
    calibration_rows = vrq.deep_rows(
        model,
        embeddings,
        prototypes,
        bank,
        protocol.calibration_val_subjects,
        fold_id,
        "trained_source_calibration",
    )
    target_rows = vrq.deep_rows(
        model,
        embeddings,
        prototypes,
        bank,
        protocol.test_subjects,
        fold_id,
        "trained_outer_test",
    )
    return calibration_rows, target_rows, moments


def _save_fold_cache(
    root: Path,
    target_rows: list[dict],
    margins: np.ndarray,
    target_moments: np.ndarray,
    source_rows: list[dict],
    source_moments: np.ndarray,
    checkpoint_path: Path,
    threshold: float,
) -> None:
    write_csv(root / "target_rows.csv", target_rows)
    write_npz(root / "target_arrays.npz", margins=margins, moments=target_moments)
    write_csv(root / "source_rows.csv", source_rows)
    write_npz(root / "source_arrays.npz", moments=source_moments)
    write_json(
        root / "manifest.json",
        {
            "status": "complete",
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "source_only_a1_threshold": float(threshold),
            "severity_head_loaded": False,
        },
    )


def _monifeixing_fold_inputs(
    protocol_root: Path,
    fold_id: str,
    target_deep_rows: list[dict],
    threshold: float,
    moments: dict[str, np.ndarray],
    protocol: FoldProtocol,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    outer = protocol_root / "outer_inputs" / fold_id
    rows = read_csv(outer / "outer_rows.csv")
    with np.load(outer / "outer_arrays.npz", allow_pickle=False) as payload:
        fixed_margins = payload["fixed_margins"].astype(np.float64)
    by_sample = {int(row["sample_index"]): row for row in target_deep_rows}
    if set(by_sample) != {int(row["sample_index"]) for row in rows}:
        raise RuntimeError(f"{fold_id} trained A1 rows do not align with fixed R1/R2 rows")
    for row in rows:
        row["local_index"] = int(by_sample[int(row["sample_index"])]["local_index"])
    deep_scores = np.asarray(
        [float(by_sample[int(row["sample_index"])]["mambakan_score"]) for row in rows],
        dtype=np.float64,
    )
    for row, score in zip(rows, deep_scores):
        for stale in ("score", "threshold", "y_pred", "correct"):
            row.pop(stale, None)
        row["mambakan_score"] = float(score)
    margins = np.column_stack(
        [fixed_margins, vrq.logit(deep_scores) - float(vrq.logit(threshold))]
    )
    _mark_target_anchors(rows, protocol.test_subjects, "rest1")
    return rows, margins, _aligned_moments(rows, moments)


def _standard_fold_inputs(
    bank,
    fixed_features: dict,
    fixed_configs: dict,
    target_deep_rows: list[dict],
    moments: dict[str, np.ndarray],
    threshold: float,
    protocol: FoldProtocol,
    seed: int,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    fixed_predictions = {}
    for family in vrq.FEATURE_FAMILIES:
        prediction, _ = vrq.fit_fixed_view(
            fixed_features[family],
            bank,
            list(protocol.source_subjects),
            list(protocol.test_subjects),
            fixed_configs[family],
            seed,
        )
        fixed_predictions[family] = prediction
    target_deep_rows.sort(key=lambda row: int(row["sample_index"]))
    margins = np.column_stack(
        [
            vrq.logit(fixed_predictions[family]["scores"])
            - float(vrq.logit(fixed_predictions[family]["threshold"]))
            for family in vrq.FEATURE_FAMILIES
        ]
        + [
            vrq.logit(np.asarray([row["mambakan_score"] for row in target_deep_rows]))
            - float(vrq.logit(threshold))
        ]
    ).astype(np.float64)
    _mark_target_anchors(target_deep_rows, protocol.test_subjects, "rest01")
    return target_deep_rows, margins, _aligned_moments(target_deep_rows, moments)


def _finalize(
    dataset: str,
    output_root: Path,
    state: list[dict],
    severity: list[dict],
    folds: list[dict],
) -> dict:
    state.sort(key=lambda row: int(row["sample_index"]))
    severity.sort(
        key=lambda row: (
            subject_sort_key(str(row["subject_id"])),
            int(row.get("route_order", 0)),
        )
    )
    scored_state = [row for row in state if not int(row["calibration_anchor"])]
    report = {
        "status": "complete",
        "evaluation": "trained_a1_locked_protocol",
        "dataset": dataset,
        "protocol": {
            "state_fusion": "(R1 + R2 + 2R4) / 4",
            "severity": "R4-only source-fit LogisticRegression",
            "pair_severity_head_used_for_prediction": False,
            "reference_checkpoint_used": False,
        },
        "state_metrics": binary_metrics(scored_state),
        "r4_severity_metrics": binary_metrics(severity),
        "folds": folds,
        "module_audit": _assert_local_modules(),
    }
    write_csv(output_root / "state_predictions.csv", state)
    write_csv(output_root / "r4_predictions.csv", severity)
    write_json(output_root / "aggregate_report.json", report)
    return report


def run_trained_evaluation(
    config: dict,
    dataset: str,
    checkpoint_root: Path,
    output_root: Path,
    device_name: str,
) -> dict:
    if dataset not in {"monifeixing", "vrq", "city"}:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoints = _checkpoint_paths(Path(checkpoint_root))
    _assert_empty_output(Path(output_root))
    device = torch.device(device_name)
    dataset_data = load_training_dataset(config, dataset, device)
    roots = _asset_roots(config)

    if dataset == "monifeixing":
        encoder = TemporalEncoder().to(device)
        encoder.load_state_dict(dataset_data.bank.encoder_state, strict=True)
        monifeixing.refresh_tokens(
            dataset_data.bank,
            encoder,
            sorted(dataset_data.bank.records),
            device,
            64,
        )
        del encoder
        protocol_root = roots["monifeixing"]
        labels = {
            str(row["subject_id"]): {
                "vrsq_post_total": float(row["vrsq_post_total"]),
                "y_true": int(row["y_true"]),
            }
            for row in read_csv(protocol_root / "severity_predictions.csv")
        }
        task_sessions = {subject: "rest2" for subject in dataset_data.bank.records}
        fixed_features = None
        city_audit = None
        seed = 1001
    elif dataset == "vrq":
        protocol_root = roots["vrq"]
        manifest = read_json(protocol_root / "main" / "full" / "audit_manifest.json")
        labels = {
            subject: {"ssq_score": float(row["ssq_score"]), "y_true": int(row["ssq_label"])}
            for subject, row in manifest["audit"]["subjects"].items()
        }
        protocols = [vrq.SubjectProtocol(**row) for row in manifest["subject_protocols"]]
        task_sessions = {row.subject_id: row.final_task for row in protocols}
        fixed_features = vrq.fixed_feature_bank(dataset_data.bank)
        city_audit = None
        seed = int(manifest["run_fingerprint_payload"]["training"]["seed"])
    else:
        protocol_root = roots["city"]
        manifest = read_json(protocol_root / "audit" / "audit_manifest.json")
        city_audit = copy.deepcopy(manifest["audit"])
        data_root = Path(config["paths"]["city_data_root"])
        for metadata in city_audit["subjects"].values():
            if metadata.get("included"):
                metadata["mat_path"] = str(data_root / Path(metadata["mat_path"]).name)
        labels = None
        task_sessions = None
        fixed_features = vrq.fixed_feature_bank(dataset_data.bank)
        seed = 1001

    all_state = []
    all_severity = []
    fold_reports = []
    for fold_id in FOLD_IDS:
        protocol = dataset_data.folds[fold_id]
        model, checkpoint_payload = _load_trained_a1(
            checkpoints[fold_id], dataset, fold_id, protocol, device
        )
        calibration_deep, target_deep, moments = _trained_deep_views(
            model, dataset_data.bank, protocol, fold_id, dataset, device
        )
        threshold, threshold_metrics = vrq.choose_score_threshold(calibration_deep)

        if dataset == "monifeixing":
            target_rows, margins, target_moments = _monifeixing_fold_inputs(
                protocol_root,
                fold_id,
                target_deep,
                threshold,
                moments,
                protocol,
            )
            resmooth = True
        elif dataset == "vrq":
            metrics = read_json(protocol_root / "main" / "full" / "folds" / fold_id / "metrics.json")
            target_rows, margins, target_moments = _standard_fold_inputs(
                dataset_data.bank,
                fixed_features,
                metrics["fixed_view_configs"],
                target_deep,
                moments,
                threshold,
                protocol,
                seed,
            )
            resmooth = False
        else:
            fold_report = read_json(protocol_root / "lambda_0p3" / fold_id / "outer" / "report.json")
            target_rows, margins, target_moments = _standard_fold_inputs(
                dataset_data.bank,
                fixed_features,
                fold_report["fixed_views"],
                target_deep,
                moments,
                threshold,
                protocol,
                seed,
            )
            resmooth = False

        source_rows, source_moments = _source_raw(
            dataset_data.bank,
            {subject: moments[subject] for subject in protocol.source_subjects},
            list(protocol.source_subjects),
            "rest1" if dataset == "monifeixing" else "rest01",
        )
        _save_fold_cache(
            Path(output_root) / "raw_cache" / fold_id,
            target_rows,
            margins,
            target_moments,
            source_rows,
            source_moments,
            checkpoints[fold_id],
            threshold,
        )
        outer_state = state_rows(
            target_rows, margins, target_moments, resmooth_refinement=resmooth
        )
        source_state = source_r4_rows(source_rows, source_moments)

        if dataset == "city":
            predictions, coefficient = evaluate_city_r4(source_state, outer_state, city_audit)
        else:
            source_features = subject_r4_features(
                source_state,
                {subject: task_sessions[subject] for subject in protocol.source_subjects},
                preserve_window_order=dataset == "monifeixing",
            )
            outer_features = subject_r4_features(
                outer_state,
                {subject: task_sessions[subject] for subject in protocol.test_subjects},
                preserve_window_order=dataset == "monifeixing",
            )
            head, coefficient = fit_source_head(
                [{**row, **labels[row["subject_id"]]} for row in source_features]
            )
            predictions = attach_subject_labels(
                score_outer(head, outer_features),
                {subject: labels[subject] for subject in protocol.test_subjects},
                "vrsq_post_total" if dataset == "monifeixing" else "ssq_score",
            )
        for row in predictions:
            row["fold_id"] = fold_id
        all_state.extend(outer_state)
        all_severity.extend(predictions)
        fold_reports.append(
            {
                "fold_id": fold_id,
                "checkpoint_sha256": sha256_file(checkpoints[fold_id]),
                "source_subjects": list(protocol.source_subjects),
                "test_subjects": list(protocol.test_subjects),
                "calibration_train_subjects": list(
                    protocol.calibration_train_subjects
                ),
                "calibration_val_subjects": list(protocol.calibration_val_subjects),
                "source_only_a1_threshold": threshold,
                "source_only_a1_threshold_metrics": threshold_metrics,
                "source_r4_logistic_coefficient": coefficient,
                "severity_head_loaded": False,
                "best_epoch": int(checkpoint_payload["best_epoch"]),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return _finalize(dataset, Path(output_root), all_state, all_severity, fold_reports)
