from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.io import loadmat

from ..evaluation.geometry import geometry_distance, stable_rank
from ..model.a1 import DirectionalMambaKAN, load_checkpoint_state_dict
from ..model.encoder import TemporalEncoder, build_encoder, load_pretrained_checkpoint
from .features import cpu_state_dict, fit_and_apply_subject_ea
from .types import AuditMetadata, FeatureBank, SubjectRecord


SUBJECTS = tuple(f"sub{index}" for index in range(1, 19))
SESSIONS = (("rest1", 0), ("rest2", 1))
CHANNEL_INDICES = tuple(range(22)) + tuple(range(24, 32))
SUMMARY_SLICES = {"moments": (0, 1, 2, 3)}


@dataclass(frozen=True)
class StateSample:
    sample_index: int
    subject_id: str
    session: str
    label: int
    window_index: int
    local_index: int
    mat_path: str


def _load_windows(path: Path, mat_key: str) -> np.ndarray:
    payload = loadmat(path)
    if mat_key not in payload:
        raise KeyError(f"{path} does not contain {mat_key!r}")
    signal = np.asarray(payload[mat_key], dtype=np.float32)
    signal = signal[np.asarray(CHANNEL_INDICES)]
    windows = [signal[:, start : start + 1280] for start in range(0, signal.shape[1] - 1279, 1280)]
    if not windows:
        raise ValueError(f"{path} produced no five-second windows")
    return np.stack(windows).astype(np.float32, copy=False)


def build_raw_bank(
    source_dir: Path,
    initial_checkpoint: Path,
    device: torch.device,
    *,
    mat_key: str = "data256",
    encode_batch_size: int = 64,
) -> FeatureBank:
    encoder = build_encoder(device=device, backend="native", num_blocks=4)
    load_info = load_pretrained_checkpoint(encoder, initial_checkpoint)
    if int(load_info["loaded_keys"]) != 83 or any(
        load_info[name] for name in ("missing_keys", "unexpected_keys", "skipped_keys")
    ):
        raise RuntimeError(f"Incomplete Temporal Encoder initialization: {load_info}")
    encoder_state = cpu_state_dict(encoder)
    records: dict[str, SubjectRecord] = {}
    samples: list[StateSample] = []
    for subject_number, subject in enumerate(SUBJECTS, start=1):
        session_windows = {
            session: _load_windows(Path(source_dir) / f"{subject}_{session}_q.mat", mat_key)
            for session, _ in SESSIONS
        }
        combined = np.concatenate([session_windows[session] for session, _ in SESSIONS])
        centered = combined.astype(np.float64) - combined.mean(axis=-1, keepdims=True)
        centered /= np.maximum(centered.std(axis=(1, 2), keepdims=True), 1e-8)
        permutation = np.random.RandomState(9000 + subject_number).permutation(len(combined))
        _, ea_matrix, _ = fit_and_apply_subject_ea(combined[permutation])
        aligned = np.einsum("cd,ndt->nct", ea_matrix, centered, optimize=True).astype(np.float32)
        labels, sessions, window_indices = [], [], []
        offset = 0
        for session, label in SESSIONS:
            path = Path(source_dir) / f"{subject}_{session}_q.mat"
            for window_index in range(len(session_windows[session])):
                labels.append(label)
                sessions.append(session)
                window_indices.append(window_index)
                samples.append(
                    StateSample(
                        len(samples), subject, session, label, window_index, offset + window_index, str(path)
                    )
                )
            offset += len(session_windows[session])
        records[subject] = SubjectRecord(
            windows=aligned.astype(np.float16),
            tokens=aligned.astype(np.float16),
            labels=np.asarray(labels, dtype=np.int64),
            sessions=sessions,
        )
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if len(samples) != 2159:
        raise AssertionError(f"Expected 2,159 monifeixing windows, got {len(samples)}")
    return FeatureBank(
        records=records,
        samples=samples,
        encoder_state=encoder_state,
        audit=AuditMetadata(
            encoder_load_info=load_info,
            encoder_mode="frozen",
            manifest={"frozen_prefix_blocks": 0},
        ),
    )


@torch.no_grad()
def refresh_tokens(
    bank: FeatureBank,
    encoder: TemporalEncoder,
    subjects: list[str],
    device: torch.device,
    batch_size: int,
) -> None:
    encoder.eval()
    for subject in subjects:
        record = bank.records[subject]
        parts = []
        for start in range(0, len(record.windows), int(batch_size)):
            batch = torch.as_tensor(
                np.asarray(record.windows[start : start + int(batch_size)], dtype=np.float32),
                device=device,
            )
            parts.append(encoder.forward_tokens(batch).detach().cpu().to(torch.float16).numpy())
        record.tokens = np.concatenate(parts)


def apply_selected_encoder(
    bank: FeatureBank, checkpoint: Path, device: torch.device, batch_size: int
) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    encoder = TemporalEncoder().to(device)
    encoder.load_state_dict(payload["state_dict"], strict=True)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    refresh_tokens(bank, encoder, sorted(bank.records), device, batch_size)
    info = {
        "checkpoint": str(checkpoint),
        "best_epoch": int(payload["best_epoch"]),
        "model_spec": payload["model_spec"],
    }
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return info


def load_a1_checkpoint(path: Path, device: torch.device) -> tuple[DirectionalMambaKAN, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = DirectionalMambaKAN.from_model_spec(payload.get("model_spec")).to(device)
    load_checkpoint_state_dict(model, payload["model_state_dict"], source=str(path))
    return model, payload


@torch.no_grad()
def extract_subject_views(
    model: DirectionalMambaKAN,
    bank: FeatureBank,
    subject: str,
    device: torch.device,
    args: SimpleNamespace,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    record = bank.records[subject]
    sequences = []
    for start in range(0, len(record.tokens), int(args.eval_batch_size)):
        tokens = torch.as_tensor(
            record.tokens[start : start + int(args.eval_batch_size)], dtype=torch.float32, device=device
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            sequences.append(model.encode_sequence(tokens).float())
    sequence = torch.cat(sequences)
    normalized = model.normalize_subject(sequence, sequence)
    embeddings = []
    for start in range(0, len(normalized), int(args.eval_batch_size)):
        embeddings.append(model.pool_embedding(normalized[start : start + int(args.eval_batch_size)]).cpu())
    difference = normalized[:, 1:] - normalized[:, :-1]
    summaries = torch.cat(
        [
            normalized.mean(dim=1),
            normalized.std(dim=1, unbiased=False),
            normalized.max(dim=1).values,
            normalized.min(dim=1).values,
            difference.abs().mean(dim=1),
            difference.std(dim=1, unbiased=False),
            normalized[:, -1] - normalized[:, 0],
        ],
        dim=1,
    )
    return torch.cat(embeddings).numpy(), summaries.cpu().numpy()


def extract_views(
    model: DirectionalMambaKAN,
    bank: FeatureBank,
    subjects: list[str],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    args = SimpleNamespace(eval_batch_size=int(batch_size))
    embeddings = {}
    moments = {}
    for subject in sorted(subjects, key=lambda value: int(str(value)[3:])):
        embedding, summaries = extract_subject_views(model, bank, subject, device, args)
        embeddings[subject] = embedding
        moments[subject] = select_summary(summaries)
    return embeddings, moments


def select_summary(summaries: np.ndarray, name: str = "moments") -> np.ndarray:
    blocks = summaries.reshape(len(summaries), 7, 96)
    return np.concatenate([blocks[:, index] for index in SUMMARY_SLICES[name]], axis=1)


def fit_prototypes(
    embeddings: dict[str, np.ndarray], bank: FeatureBank, source_subjects: list[str]
) -> np.ndarray:
    subject_centers = []
    for subject in source_subjects:
        labels = bank.records[subject].labels
        subject_centers.append(
            np.stack(
                [embeddings[subject][labels == label].mean(axis=0) for label in (0, 1)],
                axis=0,
            )
        )
    subject_centers = np.asarray(subject_centers, dtype=np.float64)
    subject_centers /= np.maximum(
        np.linalg.norm(subject_centers, axis=-1, keepdims=True), 1e-8
    )
    prototypes = subject_centers.mean(axis=0)
    prototypes /= np.maximum(np.linalg.norm(prototypes, axis=-1, keepdims=True), 1e-8)
    return prototypes.astype(np.float32)


def r4_rank(values: np.ndarray, anchor_mask: np.ndarray) -> np.ndarray:
    return stable_rank(geometry_distance(values, anchor_mask, 2))
