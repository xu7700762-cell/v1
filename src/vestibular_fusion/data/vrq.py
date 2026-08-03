from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from .types import AuditMetadata, FeatureBank, SubjectRecord


WINDOW_SIZE = 1280
STRIDE = 1280
CHANNEL_INDICES = tuple(range(22)) + tuple(range(24, 32))
FEATURE_FAMILIES = ("spectral_topography", "covariance_tangent")


@dataclass(frozen=True)
class SubjectProtocol:
    subject_id: str
    final_task: str
    post_rest: Optional[str]
    expected_post_rest: str


@dataclass(frozen=True)
class StateSample:
    sample_index: int
    subject_id: str
    session: str
    label: int
    window_index: int
    local_index: int
    mat_path: str


def subject_sort_key(subject: str) -> tuple[str, int | str]:
    text = str(subject)
    suffix = "".join(character for character in text if character.isdigit())
    return text.rstrip("0123456789"), int(suffix) if suffix else text


def training_dependencies() -> dict[str, Any]:
    from ..model.encoder import build_encoder, load_pretrained_checkpoint
    from .features import (
        covariance_tangent_features,
        fit_and_apply_subject_ea,
        robust_subject_normalize,
        spectral_hjorth_features,
        spectral_topography_features,
    )

    return locals()


def load_windows(path: Path, mat_key: str) -> np.ndarray:
    payload = loadmat(path, variable_names=[mat_key])
    if mat_key not in payload:
        raise KeyError(f"{path} has no {mat_key!r} array")
    raw = np.asarray(payload[mat_key])
    if raw.ndim != 2 or raw.shape[0] <= max(CHANNEL_INDICES):
        raise ValueError(f"Invalid EEG shape in {path}: {raw.shape}")
    if not np.issubdtype(raw.dtype, np.number) or not np.isfinite(raw).all():
        raise ValueError(f"Non-numeric or non-finite EEG in {path}")
    selected = raw[np.asarray(CHANNEL_INDICES)].astype(np.float32)
    starts = range(0, selected.shape[1] - WINDOW_SIZE + 1, STRIDE)
    windows = np.stack([selected[:, start : start + WINDOW_SIZE] for start in starts])
    if not len(windows):
        raise ValueError(f"{path} produced no complete five-second windows")
    return windows


def session_order(protocol: SubjectProtocol) -> list[str]:
    sessions = ["rest01", "rest02", protocol.final_task]
    if protocol.post_rest:
        sessions.append(protocol.post_rest)
    return sessions


def build_feature_bank(
    args, device: torch.device, audit: dict, protocols: list[SubjectProtocol]
) -> FeatureBank:
    checkpoint = Path(args.pretrain_ckpt)
    dependencies = training_dependencies()
    encoder = dependencies["build_encoder"](device=device, backend="native", num_blocks=4)
    load_info = dependencies["load_pretrained_checkpoint"](encoder, checkpoint)
    if int(load_info["loaded_keys"]) != 83 or any(
        load_info[name] for name in ("missing_keys", "unexpected_keys", "skipped_keys")
    ):
        raise RuntimeError(f"Incomplete 4-block Temporal Encoder checkpoint: {load_info}")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()
    fit_ea = dependencies["fit_and_apply_subject_ea"]
    records, samples, feature_manifest = {}, [], {}
    root = Path(args.data_root)
    for subject_number, protocol in enumerate(protocols, start=1):
        raw_parts, labels, sessions, window_indices, paths, session_counts = [], [], [], [], [], {}
        for session in session_order(protocol):
            path = root / f"{protocol.subject_id}_{session}.mat"
            windows = load_windows(path, args.mat_key)
            label = 0 if session in {"rest01", "rest02"} else 1
            session_counts[session] = len(windows)
            raw_parts.append(windows)
            labels.extend([label] * len(windows))
            sessions.extend([session] * len(windows))
            window_indices.extend(range(len(windows)))
            paths.extend([str(path)] * len(windows))
        combined = np.concatenate(raw_parts).astype(np.float32)
        centered = combined.astype(np.float64) - combined.mean(axis=-1, keepdims=True)
        centered /= np.maximum(centered.std(axis=(1, 2), keepdims=True), 1e-8)
        permutation = np.random.RandomState(9000 + subject_number).permutation(len(combined))
        _, ea_matrix, ea_diagnostics = fit_ea(combined[permutation])
        aligned = np.einsum("cd,ndt->nct", ea_matrix, centered, optimize=True).astype(np.float32)
        token_parts = []
        with torch.no_grad():
            for start in range(0, len(aligned), int(args.encode_batch_size)):
                batch = torch.as_tensor(
                    aligned[start : start + int(args.encode_batch_size)],
                    dtype=torch.float32,
                    device=device,
                )
                token_parts.append(
                    encoder.forward_tokens(batch).detach().cpu().to(torch.float16).numpy()
                )
        tokens = np.concatenate(token_parts)
        label_array = np.asarray(labels, dtype=np.int64)
        for local_index, (session, label, window_index, path) in enumerate(
            zip(sessions, labels, window_indices, paths)
        ):
            samples.append(
                StateSample(
                    sample_index=len(samples),
                    subject_id=protocol.subject_id,
                    session=session,
                    label=int(label),
                    window_index=int(window_index),
                    local_index=int(local_index),
                    mat_path=path,
                )
            )
        records[protocol.subject_id] = SubjectRecord(
            windows=aligned.astype(getattr(args, "record_storage_dtype", np.float16)),
            tokens=tokens,
            labels=label_array,
            sessions=list(sessions),
        )
        feature_manifest[protocol.subject_id] = {
            "sessions": session_counts,
            "num_windows": int(len(aligned)),
            "token_shape": list(tokens.shape),
            "ea": ea_diagnostics,
        }
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not all(record.tokens.shape[1:] == (80, 525) for record in records.values()):
        raise AssertionError("Temporal Encoder token shape changed from [windows,80,525]")
    return FeatureBank(
        records=records,
        samples=samples,
        encoder_state={},
        audit=AuditMetadata(
            encoder_load_info=load_info,
            encoder_mode="frozen",
            manifest={
                "num_subjects": len(records),
                "num_allowed_windows": len(samples),
                "offline_transductive_subject_EA": True,
                "subjects": feature_manifest,
                "read_only_qc": audit.get("read_only_qc", True),
            },
        ),
    )


@torch.no_grad()
def extract_subject_views(
    model, bank: FeatureBank, subject: str, device: torch.device, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    tokens = bank.records[subject].tokens
    sequences = []
    for start in range(0, len(tokens), int(batch_size)):
        tensor = torch.as_tensor(
            tokens[start : start + int(batch_size)], dtype=torch.float32, device=device
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            sequences.append(model.encode_sequence(tensor).float())
    sequence = torch.cat(sequences)
    normalized = model.normalize_subject(sequence, sequence)
    embeddings = []
    for start in range(0, len(normalized), int(batch_size)):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            embeddings.append(
                model.pool_embedding(normalized[start : start + int(batch_size)]).float().cpu()
            )
    moments = torch.cat(
        [
            normalized.mean(dim=1),
            normalized.std(dim=1, unbiased=False),
            normalized.max(dim=1).values,
            normalized.min(dim=1).values,
        ],
        dim=1,
    )
    return torch.cat(embeddings).numpy(), moments.float().cpu().numpy()


def extract_views(
    model, bank: FeatureBank, subjects: Iterable[str], device: torch.device, batch_size: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    embeddings, moments = {}, {}
    for subject in sorted(subjects, key=subject_sort_key):
        embeddings[subject], moments[subject] = extract_subject_views(
            model, bank, subject, device, batch_size
        )
    return embeddings, moments


def fit_prototypes(
    embeddings: dict[str, np.ndarray], bank: FeatureBank, source_subjects: list[str]
) -> np.ndarray:
    centers = []
    for subject in source_subjects:
        labels = bank.records[subject].labels
        current = np.stack(
            [embeddings[subject][labels == label].mean(axis=0) for label in (0, 1)]
        )
        current /= np.maximum(np.linalg.norm(current, axis=1, keepdims=True), 1e-8)
        centers.append(current)
    prototypes = np.asarray(centers).mean(axis=0)
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-8)
    return prototypes.astype(np.float32)


def deep_rows(
    model,
    embeddings: dict[str, np.ndarray],
    prototypes: np.ndarray,
    bank: FeatureBank,
    subjects: Iterable[str],
    fold_id: str,
    split: str,
) -> list[dict]:
    temperature = float(model.temperature().detach().cpu())
    sample_lookup = {
        (sample.subject_id, sample.local_index): sample for sample in bank.samples
    }
    rows = []
    for subject in sorted(subjects, key=subject_sort_key):
        logits = temperature * (
            embeddings[subject] @ prototypes[1] - embeddings[subject] @ prototypes[0]
        )
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        for local_index, score in enumerate(scores):
            sample = sample_lookup[(subject, local_index)]
            rows.append(
                {
                    "fold_id": fold_id,
                    "split": split,
                    "sample_index": int(sample.sample_index),
                    "subject_id": subject,
                    "session": sample.session,
                    "window_index": int(sample.window_index),
                    "local_index": int(sample.local_index),
                    "y_true": int(sample.label),
                    "mambakan_score": float(score),
                    "mat_path": sample.mat_path,
                }
            )
    rows.sort(key=lambda row: row["sample_index"])
    return rows


def _threshold_metrics(rows: list[dict], threshold: float) -> dict:
    labels = np.asarray([row["y_true"] for row in rows], dtype=np.int64)
    scores = np.asarray([row["mambakan_score"] for row in rows], dtype=np.float64)
    predictions = (scores >= float(threshold)).astype(np.int64)
    subjects = np.asarray([row["subject_id"] for row in rows], dtype=object)
    subject_scores = []
    for subject in sorted(set(subjects.tolist()), key=subject_sort_key):
        mask = subjects == subject
        subject_scores.append(float(balanced_accuracy_score(labels[mask], predictions[mask])))
    return {
        "threshold": float(threshold),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "subject_macro_balanced_accuracy": float(np.mean(subject_scores)),
    }


def choose_score_threshold(rows: list[dict]) -> tuple[float, dict]:
    if not rows:
        raise ValueError("Source-only threshold selection requires non-empty rows")
    best_threshold = 0.5
    best_metrics = _threshold_metrics(rows, best_threshold)
    best_key = (-np.inf, -np.inf, -np.inf)
    for threshold in np.round(np.arange(0.10, 0.9001, 0.01), 2):
        metrics = _threshold_metrics(rows, float(threshold))
        key = (
            metrics["subject_macro_balanced_accuracy"],
            metrics["balanced_accuracy"],
            -abs(float(threshold) - 0.5),
        )
        if key > best_key:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_key = key
    return best_threshold, best_metrics


def fixed_feature_bank(bank: FeatureBank) -> dict[str, dict[str, np.ndarray]]:
    dependencies = training_dependencies()
    result = {family: {} for family in FEATURE_FAMILIES}
    for subject, record in bank.records.items():
        windows = record.windows.astype(np.float32)
        spectral = dependencies["spectral_hjorth_features"](windows)
        result["spectral_topography"][subject] = dependencies[
            "spectral_topography_features"
        ](spectral)
        result["covariance_tangent"][subject] = dependencies[
            "covariance_tangent_features"
        ](windows)
    return result


def stack_features(
    features: dict[str, np.ndarray],
    bank: FeatureBank,
    subjects: Iterable[str],
    subject_normalization: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normalize = training_dependencies()["robust_subject_normalize"]
    values, labels, groups, indices = [], [], [], []
    lookup = {
        (sample.subject_id, sample.local_index): sample.sample_index for sample in bank.samples
    }
    for subject in sorted(subjects, key=subject_sort_key):
        current = features[subject]
        if subject_normalization:
            current = normalize(current)
        values.append(current)
        labels.append(bank.records[subject].labels)
        groups.extend([subject] * len(current))
        indices.extend(lookup[(subject, local_index)] for local_index in range(len(current)))
    return (
        np.concatenate(values).astype(np.float64),
        np.concatenate(labels).astype(np.int64),
        np.asarray(groups, dtype=object),
        np.asarray(indices, dtype=np.int64),
    )


def fit_fixed_view(
    features: dict[str, np.ndarray],
    bank: FeatureBank,
    source_subjects: list[str],
    target_subjects: list[str],
    config: dict,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict]:
    source = stack_features(
        features, bank, source_subjects, bool(config["subject_normalization"])
    )
    target = stack_features(
        features, bank, target_subjects, bool(config["subject_normalization"])
    )
    scaler = StandardScaler().fit(source[0])
    source_scaled, target_scaled = scaler.transform(source[0]), scaler.transform(target[0])
    effective = min(int(config["pca_dim"]), len(source_scaled) - 1, source_scaled.shape[1])
    pca = PCA(n_components=effective, random_state=int(seed)).fit(source_scaled)
    classifier = LogisticRegression(
        C=float(config["C"]),
        class_weight="balanced",
        max_iter=3000,
        solver="lbfgs",
        random_state=int(seed),
    ).fit(pca.transform(source_scaled), source[1])
    scores = classifier.predict_proba(pca.transform(target_scaled))[:, 1]
    order = np.argsort(target[3])
    prediction = {
        "labels": target[1][order],
        "scores": scores[order],
        "subjects": target[2][order],
        "indices": target[3][order],
        "threshold": float(config["threshold"]),
    }
    return prediction, {
        "config": config,
        "scaler": scaler,
        "pca": pca,
        "classifier": classifier,
        "source_subjects": source_subjects,
    }


def logit(values: np.ndarray | float) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values))
