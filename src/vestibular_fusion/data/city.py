from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from . import vrq as body


WINDOW_SIZE = 1280
STRIDE = 1280
CHANNEL_INDICES = tuple(range(22)) + tuple(range(24, 32))
SUBJECTS = tuple(f"acq{index:02d}" for index in range(1, 27) if index != 22)


def subject_sort_key(subject: str) -> tuple[str, int | str]:
    return body.subject_sort_key(subject)


def load_subject_mat(path: Path, mat_key: str = "data256") -> np.ndarray:
    payload = loadmat(path, variable_names=[mat_key])
    if mat_key not in payload:
        raise KeyError(f"{path} has no {mat_key!r} array")
    raw = np.asarray(payload[mat_key])
    if raw.ndim != 2 or raw.shape[0] != 37 or not np.isfinite(raw).all():
        raise ValueError(f"Invalid city-cruise MAT data: {path} {raw.shape}")
    return raw


def session_alias(segment: dict, anchor_session: str) -> str:
    if segment["session"] == anchor_session:
        return "rest01"
    return f"{segment['state']}_seg_{int(segment['segment_index']):02d}"


def build_feature_bank(args, device: torch.device, audit: dict) -> body.FeatureBank:
    dependencies = body.training_dependencies()
    encoder = dependencies["build_encoder"](device=device, backend="native", num_blocks=4)
    load_info = dependencies["load_pretrained_checkpoint"](encoder, Path(args.pretrain_ckpt))
    if int(load_info["loaded_keys"]) != 83 or any(
        load_info[name] for name in ("missing_keys", "unexpected_keys", "skipped_keys")
    ):
        raise RuntimeError(f"Incomplete frozen Temporal Encoder checkpoint load: {load_info}")
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()
    fit_ea = dependencies["fit_and_apply_subject_ea"]
    records, samples, subject_manifest = {}, [], {}
    for subject_position, subject in enumerate(SUBJECTS, start=1):
        metadata = audit["subjects"][subject]
        raw = load_subject_mat(Path(metadata["mat_path"]), args.mat_key)
        parts, labels, sessions, window_indices = [], [], [], []
        segment_counts = {}
        for segment in metadata["segments"]:
            alias = session_alias(segment, metadata["anchor_session"])
            selected = raw[
                np.asarray(CHANNEL_INDICES),
                int(segment["start_sample"]) : int(segment["end_sample"]),
            ]
            starts = list(range(0, selected.shape[1] - WINDOW_SIZE + 1, STRIDE))
            if len(starts) < 3:
                raise AssertionError(f"{subject}/{alias} cannot provide a 15-second context")
            windows = np.stack(
                [selected[:, start : start + WINDOW_SIZE] for start in starts]
            ).astype(np.float32)
            parts.append(windows)
            labels.extend([int(segment["state"] == "task")] * len(windows))
            sessions.extend([alias] * len(windows))
            window_indices.extend(range(len(windows)))
            segment_counts[alias] = len(windows)
        combined = np.concatenate(parts).astype(np.float32)
        centered = combined.astype(np.float64) - combined.mean(axis=-1, keepdims=True)
        centered /= np.maximum(centered.std(axis=(1, 2), keepdims=True), 1e-8)
        permutation = np.random.RandomState(9000 + subject_position).permutation(len(combined))
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
        for local_index, (session, label, window_index) in enumerate(
            zip(sessions, labels, window_indices)
        ):
            samples.append(
                body.StateSample(
                    sample_index=len(samples),
                    subject_id=subject,
                    session=session,
                    label=int(label),
                    window_index=int(window_index),
                    local_index=local_index,
                    mat_path=str(metadata["mat_path"]),
                )
            )
        records[subject] = body.SubjectRecord(
            windows=aligned.astype(getattr(args, "record_storage_dtype", np.float16)),
            tokens=tokens,
            labels=np.asarray(labels, dtype=np.int64),
            sessions=list(sessions),
        )
        subject_manifest[subject] = {
            "segments": segment_counts,
            "num_windows": len(aligned),
            "token_shape": list(tokens.shape),
            "ea": ea_diagnostics,
        }
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not all(record.tokens.shape[1:] == (80, 525) for record in records.values()):
        raise AssertionError("Frozen Temporal Encoder token shape changed from [N,80,525]")
    return body.FeatureBank(
        records=records,
        samples=samples,
        encoder_state={},
        audit=body.AuditMetadata(
            encoder_load_info=load_info,
            encoder_mode="frozen",
            manifest={
                "num_subjects": len(records),
                "num_windows": len(samples),
                "offline_transductive_subject_EA": True,
                "subjects": subject_manifest,
            },
        ),
    )
