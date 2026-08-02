from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ..data import city, monifeixing, vrq
from ..model.a1 import DirectionalMambaKAN
from ..protocol import PROTOCOL
from .fusion import source_r4_rows, state_rows, uniform_anchor_mask
from .io import (
    array_sha256,
    comparison,
    read_csv,
    read_json,
    rows_sha256,
    sha256_file,
    write_csv,
    write_json,
    write_npz,
)
from .metrics import binary_metrics, subject_sort_key
from .severity import (
    attach_subject_labels,
    evaluate_city_r4,
    fit_source_head,
    score_outer,
    subject_r4_features,
)


DATASETS = ("monifeixing", "vrq", "city")


def _asset_roots(config: dict) -> dict[str, Path]:
    asset_root = Path(config["paths"]["asset_root"]).resolve()
    vr_root = asset_root / "vr_ssq_regression"
    baseline = vr_root / "fair_joint_lambda0p3_no_inner_seed42" / "artifacts"
    return {
        "asset": asset_root,
        "vr": vr_root,
        "baseline": baseline,
        "reference_state": baseline / "r_fusion_drop",
        "monifeixing": vr_root
        / "artifacts_fair_joint_lambda0p3"
        / "monifeixing"
        / "lambda0p3"
        / "seed42"
        / "full",
        "vrq": vr_root / "artifacts_fair_joint_lambda0p3" / "vrq" / "seed_42",
        "city": vr_root / "artifacts_city_a3_lambda_sweep_strict",
    }


def _resolve_asset(asset_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else asset_root / path


def _assert_empty_output(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Reproduction refuses a non-empty output directory (no resume): {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)


def _assert_local_modules() -> dict:
    package_root = Path(__file__).resolve().parents[1]
    offenders = []
    for name, module in sys.modules.items():
        if not name.startswith("biofoundation_v1") or not getattr(module, "__file__", None):
            continue
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(package_root) or ("_" + "engine") in path.parts:
            offenders.append(f"{name}: {path}")
    if offenders:
        raise RuntimeError("Reproduction imported a non-release project module:\n" + "\n".join(offenders))
    return {"package_root": str(package_root), "checked_modules": len(sys.modules)}


def _load_checkpoint_model(path: Path, device: torch.device) -> tuple[DirectionalMambaKAN, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = DirectionalMambaKAN.from_model_spec(payload.get("model_spec")).to(device)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint has no model_state_dict: {path}")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, payload


def _source_raw(
    bank, moments: dict[str, np.ndarray], subjects: list[str], anchor_session: str
) -> tuple[list[dict], np.ndarray]:
    selected = set(subjects)
    rows = [
        {
            "sample_index": int(sample.sample_index),
            "subject_id": sample.subject_id,
            "session": sample.session,
            "window_index": int(sample.window_index),
            "local_index": int(sample.local_index),
        }
        for sample in sorted(bank.samples, key=lambda sample: int(sample.sample_index))
        if sample.subject_id in selected
    ]
    for subject in sorted(selected, key=subject_sort_key):
        indices = [index for index, row in enumerate(rows) if row["subject_id"] == subject]
        mask = uniform_anchor_mask([rows[index] for index in indices], anchor_session)
        for index, is_anchor in zip(indices, mask):
            rows[index]["calibration_anchor"] = int(is_anchor)
    values = np.stack(
        [moments[row["subject_id"]][int(row["local_index"])] for row in rows]
    ).astype(np.float64)
    return rows, values


def _save_raw_cache(
    root: Path,
    target_rows: list[dict],
    margins: np.ndarray,
    target_moments: np.ndarray,
    source_rows: list[dict],
    source_moments: np.ndarray,
    checkpoint: dict,
) -> dict:
    hashes = {
        "target_rows": rows_sha256(target_rows),
        "margins": array_sha256(margins),
        "target_moments": array_sha256(target_moments),
        "source_rows": rows_sha256(source_rows),
        "source_moments": array_sha256(source_moments),
    }
    write_csv(root / "target_rows.csv", target_rows)
    write_npz(root / "target_arrays.npz", margins=margins, moments=target_moments)
    write_csv(root / "source_rows.csv", source_rows)
    write_npz(root / "source_arrays.npz", moments=source_moments)
    manifest = {"status": "complete", "hashes": hashes, "checkpoint": checkpoint}
    write_json(root / "manifest.json", manifest)
    return manifest


def _reference_predictions(roots: dict[str, Path], dataset: str) -> tuple[list[dict], list[dict]]:
    state = read_csv(roots["reference_state"] / dataset / "full_no_r3" / "predictions.csv")
    severity = read_csv(roots["baseline"] / dataset / "r4_predictions.csv")
    return state, severity


def _finalize_dataset(
    dataset: str,
    output_root: Path,
    roots: dict[str, Path],
    state: list[dict],
    severity: list[dict],
    folds: list[dict],
    raw_manifests: dict[str, dict],
) -> dict:
    state.sort(key=lambda row: int(row["sample_index"]))
    severity.sort(
        key=lambda row: (
            subject_sort_key(str(row["subject_id"])),
            int(row.get("route_order", 0)),
        )
    )
    scored_state = [row for row in state if not int(row["calibration_anchor"])]
    reference_state, reference_severity = _reference_predictions(roots, dataset)
    state_reproduction = comparison(reference_state, scored_state)
    severity_reproduction = comparison(reference_severity, severity)
    report = {
        "dataset": dataset,
        "state_metrics": binary_metrics(scored_state),
        "r4_severity_metrics": binary_metrics(severity),
        "state_reproduction": state_reproduction,
        "severity_reproduction": severity_reproduction,
        "folds": folds,
        "raw_manifests": raw_manifests,
    }
    root = output_root / dataset
    write_csv(root / "state_predictions.csv", state)
    write_csv(root / "r4_predictions.csv", severity)
    write_json(root / "report.json", report)
    return report


def _monifeixing_target_inputs(root: Path, fold_id: str) -> tuple[list[dict], np.ndarray, np.ndarray]:
    outer = root / "outer_inputs" / fold_id
    rows = read_csv(outer / "outer_rows.csv")
    with np.load(outer / "outer_arrays.npz", allow_pickle=False) as payload:
        margins = np.column_stack([payload["fixed_margins"], payload["deep_margin"]]).astype(
            np.float64
        )
        moments = monifeixing.select_summary(payload["summaries"].astype(np.float64))
    anchors = {
        int(row["sample_index"]): int(row["calibration_anchor"])
        for row in read_csv(root / "state_predictions.csv")
    }
    for row in rows:
        row["calibration_anchor"] = anchors[int(row["sample_index"])]
    return rows, margins, moments


def run_monifeixing(config: dict, output_root: Path, device: torch.device, roots: dict) -> dict:
    protocol_root = roots["monifeixing"]
    report = read_json(protocol_root / "report.json")
    bank = monifeixing.build_raw_bank(
        Path(config["paths"]["monifeixing_data_root"]),
        Path(config["paths"]["monifeixing_initial_femba"]),
        device,
    )
    labels = {
        str(row["subject_id"]): {
            "vrsq_post_total": float(row["vrsq_post_total"]),
            "y_true": int(row["y_true"]),
        }
        for row in read_csv(protocol_root / "severity_predictions.csv")
    }
    initializations = {
        row["initialization_id"]: row
        for row in report["initialization_reports"]
        if row["kind"] == "outer"
    }
    all_state, all_severity, fold_reports = [], [], []
    raw_manifests = {}
    for number in range(1, 6):
        fold_id = f"fold_{number}"
        split = report["identity_audit"]["folds"][fold_id]
        source_subjects = list(split["source_outer_train_subjects"])
        test_subjects = list(split["test_subjects"])
        initialization = initializations[fold_id]
        encoder_path = _resolve_asset(roots["asset"], initialization["encoder"]["checkpoint"])
        a1_path = _resolve_asset(roots["asset"], initialization["a1_checkpoint"])
        monifeixing.apply_selected_encoder(bank, encoder_path, device, 64)
        model, _ = monifeixing.load_a1_checkpoint(a1_path, device)
        source_moments = {}
        for subject in source_subjects:
            _, summaries = monifeixing.extract_subject_views(
                model, bank, subject, device, SimpleNamespace(eval_batch_size=128)
            )
            source_moments[subject] = monifeixing.select_summary(summaries)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        target_rows, margins, target_moments = _monifeixing_target_inputs(
            protocol_root, fold_id
        )
        source_rows, source_values = _source_raw(
            bank, source_moments, source_subjects, "rest1"
        )
        raw_manifests[fold_id] = _save_raw_cache(
            output_root / "monifeixing" / "raw_cache" / fold_id,
            target_rows,
            margins,
            target_moments,
            source_rows,
            source_values,
            {
                "encoder": str(encoder_path),
                "encoder_sha256": sha256_file(encoder_path),
                "a1": str(a1_path),
                "a1_sha256": sha256_file(a1_path),
            },
        )
        outer_state = state_rows(
            target_rows, margins, target_moments, resmooth_refinement=True
        )
        source_state = source_r4_rows(source_rows, source_values)
        source_features = subject_r4_features(
            source_state,
            {subject: "rest2" for subject in source_subjects},
            preserve_window_order=True,
        )
        outer_features = subject_r4_features(
            outer_state,
            {subject: "rest2" for subject in test_subjects},
            preserve_window_order=True,
        )
        head, coefficient = fit_source_head(
            [{**row, **labels[row["subject_id"]]} for row in source_features]
        )
        predictions = attach_subject_labels(
            score_outer(head, outer_features),
            {subject: labels[subject] for subject in test_subjects},
            "vrsq_post_total",
        )
        for row in predictions:
            row["fold_id"] = fold_id
        all_state.extend(outer_state)
        all_severity.extend(predictions)
        fold_reports.append(
            {
                "fold_id": fold_id,
                "source_subjects": source_subjects,
                "test_subjects": test_subjects,
                "source_coefficient": coefficient,
            }
        )
    return _finalize_dataset(
        "monifeixing",
        output_root,
        roots,
        all_state,
        all_severity,
        fold_reports,
        raw_manifests,
    )


def _vrq_args(config: dict, manifest: dict) -> SimpleNamespace:
    training = manifest["run_fingerprint_payload"]["training"]
    return SimpleNamespace(
        pretrain_ckpt=str(config["paths"]["pretrain_checkpoint"]),
        data_root=str(config["paths"]["vrq_data_root"]),
        ssq_path=str(config["paths"]["vrq_ssq_path"]),
        mat_key=manifest["run_fingerprint_payload"]["mat_key"],
        encoder_backend="native",
        ea_mode=training["ea_mode"],
        encode_batch_size=int(training["encode_batch_size"]),
        eval_batch_size=int(training["eval_batch_size"]),
        record_storage_dtype=np.float16,
        dropout=float(training["dropout"]),
        seed=int(training["seed"]),
    )


def _fold_raw(
    *,
    bank,
    features: dict,
    source_subjects: list[str],
    target_subjects: list[str],
    checkpoint_path: Path,
    fixed_configs: dict,
    deep_threshold: float,
    fold_id: str,
    split_name: str,
    device: torch.device,
    batch_size: int,
    seed: int,
    baseline_state_path: Path,
) -> tuple[list[dict], np.ndarray, np.ndarray, list[dict], np.ndarray, dict]:
    model, checkpoint = _load_checkpoint_model(checkpoint_path, device)
    if list(checkpoint.get("source_subjects", [])) != list(source_subjects):
        raise RuntimeError(f"{fold_id} checkpoint source subjects changed")
    fixed_predictions = {}
    for family in vrq.FEATURE_FAMILIES:
        prediction, _ = vrq.fit_fixed_view(
            features[family], bank, source_subjects, target_subjects, fixed_configs[family], seed
        )
        fixed_predictions[family] = prediction
    embeddings, moments = vrq.extract_views(
        model, bank, source_subjects + target_subjects, device, batch_size
    )
    prototypes = vrq.fit_prototypes(embeddings, bank, source_subjects)
    rows = vrq.deep_rows(
        model, embeddings, prototypes, bank, target_subjects, fold_id, split_name
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    margins = np.column_stack(
        [
            vrq.logit(fixed_predictions[family]["scores"])
            - float(vrq.logit(fixed_predictions[family]["threshold"]))
            for family in vrq.FEATURE_FAMILIES
        ]
        + [
            vrq.logit(np.asarray([row["mambakan_score"] for row in rows]))
            - float(vrq.logit(deep_threshold))
        ]
    ).astype(np.float64)
    anchors = {
        int(row["sample_index"]): int(row["calibration_anchor"])
        for row in read_csv(baseline_state_path)
    }
    for row in rows:
        row["calibration_anchor"] = anchors[int(row["sample_index"])]
    target_moments = np.stack(
        [moments[row["subject_id"]][int(row["local_index"])] for row in rows]
    ).astype(np.float64)
    source_rows, source_moments = _source_raw(
        bank, {subject: moments[subject] for subject in source_subjects}, source_subjects, "rest01"
    )
    return rows, margins, target_moments, source_rows, source_moments, checkpoint


def run_vrq(config: dict, output_root: Path, device: torch.device, roots: dict) -> dict:
    protocol_root = roots["vrq"]
    manifest = read_json(protocol_root / "main" / "full" / "audit_manifest.json")
    args = _vrq_args(config, manifest)
    protocols = [vrq.SubjectProtocol(**row) for row in manifest["subject_protocols"]]
    bank = vrq.build_feature_bank(args, device, manifest["audit"], protocols)
    features = vrq.fixed_feature_bank(bank)
    labels = {
        subject: {"ssq_score": float(row["ssq_score"]), "y_true": int(row["ssq_label"])}
        for subject, row in manifest["audit"]["subjects"].items()
    }
    task_sessions = {row.subject_id: row.final_task for row in protocols}
    all_state, all_severity, fold_reports = [], [], []
    raw_manifests = {}
    for fold_id, fold in manifest["folds"].items():
        fold_root = protocol_root / "main" / "full" / "folds" / fold_id
        metrics = read_json(fold_root / "metrics.json")
        checkpoint_path = fold_root / "refit.pt"
        source_subjects = sorted(
            fold["train_subjects"] + fold["val_subjects"], key=subject_sort_key
        )
        test_subjects = list(fold["test_subjects"])
        target_rows, margins, target_moments, source_rows, source_moments, checkpoint = _fold_raw(
            bank=bank,
            features=features,
            source_subjects=source_subjects,
            target_subjects=test_subjects,
            checkpoint_path=checkpoint_path,
            fixed_configs=metrics["fixed_view_configs"],
            deep_threshold=float(metrics["selection"]["mambakan_threshold"]),
            fold_id=fold_id,
            split_name="outer_test",
            device=device,
            batch_size=args.eval_batch_size,
            seed=args.seed,
            baseline_state_path=fold_root / "state_predictions.csv",
        )
        raw_manifests[fold_id] = _save_raw_cache(
            output_root / "vrq" / "raw_cache" / fold_id,
            target_rows,
            margins,
            target_moments,
            source_rows,
            source_moments,
            {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "fixed_epochs": int(checkpoint["fixed_epochs"]),
            },
        )
        outer_state = state_rows(target_rows, margins, target_moments, resmooth_refinement=False)
        source_state = source_r4_rows(source_rows, source_moments)
        source_features = subject_r4_features(
            source_state, {subject: task_sessions[subject] for subject in source_subjects}
        )
        outer_features = subject_r4_features(
            outer_state, {subject: task_sessions[subject] for subject in test_subjects}
        )
        head, coefficient = fit_source_head(
            [{**row, **labels[row["subject_id"]]} for row in source_features]
        )
        predictions = attach_subject_labels(
            score_outer(head, outer_features),
            {subject: labels[subject] for subject in test_subjects},
            "ssq_score",
        )
        for row in predictions:
            row["fold_id"] = fold_id
        all_state.extend(outer_state)
        all_severity.extend(predictions)
        fold_reports.append(
            {
                "fold_id": fold_id,
                "source_subjects": source_subjects,
                "test_subjects": test_subjects,
                "source_coefficient": coefficient,
            }
        )
    return _finalize_dataset(
        "vrq", output_root, roots, all_state, all_severity, fold_reports, raw_manifests
    )


def _city_args(config: dict) -> SimpleNamespace:
    return SimpleNamespace(
        pretrain_ckpt=str(config["paths"]["pretrain_checkpoint"]),
        data_root=str(config["paths"]["city_data_root"]),
        record_workbook=str(config["paths"]["city_record_workbook"]),
        ssq_workbook=str(config["paths"]["city_ssq_workbook"]),
        acq26_scores=str(config["paths"]["city_acq26_scores"]),
        source_vrsq_workbook=str(config["paths"]["city_source_vrsq_workbook"]),
        mat_key="data256",
        encoder_backend="native",
        ea_mode="subject_unlabeled",
        encode_batch_size=64,
        eval_batch_size=128,
        record_storage_dtype=np.float16,
        dropout=0.25,
        seed=1001,
    )


def _rebase_city_audit(audit: dict, data_root: Path) -> dict:
    audit = copy.deepcopy(audit)
    audit["data_root"] = str(data_root)
    for metadata in audit["subjects"].values():
        if metadata.get("included"):
            metadata["mat_path"] = str(data_root / Path(metadata["mat_path"]).name)
    return audit


def run_city(config: dict, output_root: Path, device: torch.device, roots: dict) -> dict:
    protocol_root = roots["city"]
    manifest = read_json(protocol_root / "audit" / "audit_manifest.json")
    args = _city_args(config)
    audit = _rebase_city_audit(manifest["audit"], Path(config["paths"]["city_data_root"]))
    bank = city.build_feature_bank(args, device, audit)
    features = vrq.fixed_feature_bank(bank)
    all_state, all_severity, fold_reports = [], [], []
    raw_manifests = {}
    for fold_id, fold in manifest["fold_manifest"]["folds"].items():
        fold_root = protocol_root / "lambda_0p3" / fold_id / "outer"
        fold_report = read_json(fold_root / "report.json")
        checkpoint_path = fold_root / "refit.pt"
        source_subjects = sorted(
            fold["train_subjects"] + fold["val_subjects"], key=subject_sort_key
        )
        test_subjects = list(fold["test_subjects"])
        target_rows, margins, target_moments, source_rows, source_moments, checkpoint = _fold_raw(
            bank=bank,
            features=features,
            source_subjects=source_subjects,
            target_subjects=test_subjects,
            checkpoint_path=checkpoint_path,
            fixed_configs=fold_report["fixed_views"],
            deep_threshold=float(fold_report["selection"]["threshold"]),
            fold_id=fold_id,
            split_name="test",
            device=device,
            batch_size=args.eval_batch_size,
            seed=args.seed,
            baseline_state_path=fold_root / "state_predictions.csv",
        )
        raw_manifests[fold_id] = _save_raw_cache(
            output_root / "city" / "raw_cache" / fold_id,
            target_rows,
            margins,
            target_moments,
            source_rows,
            source_moments,
            {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "fixed_epochs": int(checkpoint["fixed_epochs"]),
            },
        )
        outer_state = state_rows(target_rows, margins, target_moments, resmooth_refinement=False)
        source_state = source_r4_rows(source_rows, source_moments)
        predictions, coefficient = evaluate_city_r4(source_state, outer_state, audit)
        for row in predictions:
            row["fold_id"] = fold_id
        all_state.extend(outer_state)
        all_severity.extend(predictions)
        fold_reports.append(
            {
                "fold_id": fold_id,
                "source_subjects": source_subjects,
                "test_subjects": test_subjects,
                "source_coefficient": coefficient,
            }
        )
    return _finalize_dataset(
        "city", output_root, roots, all_state, all_severity, fold_reports, raw_manifests
    )


def run_reproduction(
    config: dict, datasets: list[str], device_name: str, output_root: Path
) -> dict:
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise ValueError(f"Unsupported datasets: {unknown}")
    _assert_empty_output(output_root)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    roots = _asset_roots(config)
    reports = {}
    runners = {
        "monifeixing": run_monifeixing,
        "vrq": run_vrq,
        "city": run_city,
    }
    for dataset in datasets:
        reports[dataset] = runners[dataset](config, output_root, device, roots)
        write_json(
            output_root / "progress.json",
            {"status": "running", "completed_datasets": list(reports)},
        )
    module_audit = _assert_local_modules()
    aggregate = {
        "status": "complete",
        "protocol": PROTOCOL,
        "datasets": reports,
        "module_audit": module_audit,
    }
    write_json(output_root / "aggregate_report.json", aggregate)
    (output_root / "progress.json").unlink(missing_ok=True)
    return aggregate
