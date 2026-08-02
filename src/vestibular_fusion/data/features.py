from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


WINDOW_SIZE = 1280
SAMPLING_RATE = 256
EA_SHRINKAGE = 0.0
BANDS = ((1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45), (55, 80), (80, 100))
REGIONS = (tuple(range(0, 14)), tuple(range(14, 22)), tuple(range(22, 27)), tuple(range(27, 30)))
SYMMETRIC_PAIRS = (
    (0, 1), (2, 8), (3, 7), (4, 6), (9, 13), (10, 12),
    (14, 18), (15, 17), (19, 21), (22, 26), (23, 25), (27, 29),
)


def spectral_hjorth_features(windows: np.ndarray) -> np.ndarray:
    centered = windows - windows.mean(axis=-1, keepdims=True)
    taper = np.hanning(windows.shape[-1]).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(centered * taper[None, None, :], axis=-1)) ** 2
    frequencies = np.fft.rfftfreq(windows.shape[-1], 1.0 / SAMPLING_RATE)
    band_power = np.stack(
        [spectrum[:, :, (frequencies >= low) & (frequencies < high)].mean(axis=-1) for low, high in BANDS],
        axis=-1,
    )
    absolute_log = np.log(band_power + 1e-8)
    relative = band_power / (band_power.sum(axis=-1, keepdims=True) + 1e-8)
    diff1 = np.diff(centered, axis=-1)
    diff2 = np.diff(diff1, axis=-1)
    var0 = centered.var(axis=-1) + 1e-8
    var1 = diff1.var(axis=-1) + 1e-8
    var2 = diff2.var(axis=-1) + 1e-8
    mobility = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / (mobility + 1e-8)
    hjorth = np.stack([np.log(var0), mobility, complexity], axis=-1)
    return np.concatenate([absolute_log, relative, hjorth], axis=-1).reshape(len(windows), -1).astype(np.float32)


def spectral_topography_features(spectral: np.ndarray) -> np.ndarray:
    per_channel = spectral.reshape(len(spectral), 30, -1)
    region_means = np.concatenate(
        [per_channel[:, np.asarray(region)].mean(axis=1) for region in REGIONS], axis=1
    )
    asymmetry = np.concatenate(
        [per_channel[:, left] - per_channel[:, right] for left, right in SYMMETRIC_PAIRS], axis=1
    )
    return np.concatenate(
        [region_means, asymmetry, per_channel.mean(axis=1), per_channel.std(axis=1)], axis=1
    ).astype(np.float32)


def covariance_tangent_features(windows: np.ndarray, shrinkage: float = 0.10) -> np.ndarray:
    centered = windows.astype(np.float64) - windows.mean(axis=-1, keepdims=True)
    covariance = centered @ centered.transpose(0, 2, 1) / float(windows.shape[-1] - 1)
    trace = np.trace(covariance, axis1=1, axis2=2) / covariance.shape[1]
    covariance = (1.0 - shrinkage) * covariance + shrinkage * trace[:, None, None] * np.eye(
        covariance.shape[1], dtype=np.float64
    )[None]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    log_covariance = np.einsum(
        "nij,nj,nkj->nik",
        eigenvectors,
        np.log(np.maximum(eigenvalues, 1e-8)),
        eigenvectors,
        optimize=True,
    )
    upper = np.triu_indices(covariance.shape[1])
    features = log_covariance[:, upper[0], upper[1]]
    features[:, upper[0] != upper[1]] *= math.sqrt(2.0)
    return features.astype(np.float32)


def robust_subject_normalize(features: np.ndarray) -> np.ndarray:
    median = np.median(features, axis=0, keepdims=True)
    q25, q75 = np.quantile(features, [0.25, 0.75], axis=0)
    return ((features - median) / np.maximum(q75 - q25, 1e-5)[None]).astype(np.float32)


def cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def fit_and_apply_subject_ea(unlabeled_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    if not isinstance(unlabeled_windows, np.ndarray) or unlabeled_windows.ndim != 3:
        raise TypeError("EA expects [windows,channels,time]")
    if unlabeled_windows.shape[1:] != (30, WINDOW_SIZE):
        raise ValueError(f"Unexpected EA input shape: {unlabeled_windows.shape}")
    centered = unlabeled_windows.astype(np.float64) - unlabeled_windows.mean(axis=-1, keepdims=True)
    standardized = centered / np.maximum(centered.std(axis=(1, 2), keepdims=True), 1e-8)
    covariance = standardized @ standardized.transpose(0, 2, 1) / float(WINDOW_SIZE - 1)
    reference = covariance.mean(axis=0)
    isotropic = float(np.trace(reference)) / reference.shape[0]
    reference = (1.0 - EA_SHRINKAGE) * reference + EA_SHRINKAGE * isotropic * np.eye(
        reference.shape[0], dtype=np.float64
    )
    ridge = max(float(np.trace(reference)) / reference.shape[0] * 1e-6, 1e-8)
    reference += np.eye(reference.shape[0], dtype=np.float64) * ridge
    eigenvalues, eigenvectors = np.linalg.eigh(reference)
    inverse_sqrt = (eigenvectors * np.maximum(eigenvalues, 1e-10) ** -0.5) @ eigenvectors.T
    aligned = np.einsum("cd,ndt->nct", inverse_sqrt, standardized, optimize=True).astype(np.float32)
    return aligned, inverse_sqrt.astype(np.float32), {
        "num_unlabeled_windows": int(len(unlabeled_windows)),
        "shrinkage": EA_SHRINKAGE,
        "reference_condition_number": float(np.linalg.cond(reference)),
        "aligned_mean_abs": float(np.abs(aligned.mean()).item()),
        "aligned_std": float(aligned.std().item()),
    }


def domain_batches(bank, indices: list[int], seed: int, domains_per_batch: int, trials_per_class: int):
    by_subject: dict[str, dict[int, list[int]]] = {}
    for sample_index in indices:
        sample = bank.samples[int(sample_index)]
        by_subject.setdefault(sample.subject_id, {}).setdefault(sample.label, []).append(sample.local_index)
    subjects = sorted(by_subject)
    rng = random.Random(int(seed))
    num_batches = int(math.ceil(len(indices) / (domains_per_batch * trials_per_class * 2)))
    for _ in range(num_batches):
        chosen = rng.sample(subjects, min(domains_per_batch, len(subjects)))
        while len(chosen) < domains_per_batch:
            chosen.append(rng.choice(subjects))
        local_indices = []
        for subject in chosen:
            current = rng.sample(by_subject[subject][0], trials_per_class) + rng.sample(
                by_subject[subject][1], trials_per_class
            )
            rng.shuffle(current)
            local_indices.append(current)
        yield chosen, local_indices


def assemble_domain_batch(bank, subjects: list[str], local_indices: list[list[int]]):
    tokens, windows, labels = [], [], []
    for subject, indices in zip(subjects, local_indices):
        record = bank.records[subject]
        tokens.append(record.tokens[indices].astype(np.float32))
        windows.append(record.windows[indices].astype(np.float32))
        labels.append(record.labels[indices].astype(np.float32))
    return (
        torch.from_numpy(np.stack(tokens)),
        torch.from_numpy(np.stack(windows)),
        torch.from_numpy(np.stack(labels)),
    )


def subject_balanced_centers(embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    centers = []
    for domain_index in range(embeddings.shape[0]):
        centers.append(
            torch.stack(
                [
                    F.normalize(
                        embeddings[domain_index, labels[domain_index].long() == label].mean(dim=0), dim=0
                    )
                    for label in (0, 1)
                ]
            )
        )
    return torch.stack(centers)


def leave_one_subject_out_logits(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    centers = subject_balanced_centers(embeddings, labels)
    if centers.shape[0] < 2:
        raise ValueError("Directional prototype training requires at least two subjects")
    directions = F.normalize(centers[:, 1] - centers[:, 0], dim=-1)
    logits, direction_losses, ranking_losses = [], [], []
    for domain_index in range(len(centers)):
        other = torch.arange(len(centers), device=centers.device) != domain_index
        prototype = F.normalize(centers[other].mean(dim=0), dim=-1)
        current = temperature * (
            embeddings[domain_index] @ prototype[1] - embeddings[domain_index] @ prototype[0]
        )
        logits.append(current)
        global_direction = F.normalize(directions[other].mean(dim=0), dim=0)
        direction_losses.append(1.0 - torch.dot(directions[domain_index], global_direction))
        ranking_losses.append(
            F.softplus(
                current.new_tensor(0.20)
                - (current[labels[domain_index] > 0.5].mean() - current[labels[domain_index] <= 0.5].mean())
            )
        )
    logits_tensor = torch.stack(logits)
    return logits_tensor, torch.stack(direction_losses).mean(), torch.stack(ranking_losses).mean(), {}
