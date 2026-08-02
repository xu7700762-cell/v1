from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class SubjectRecord:
    """Data consumed by training and evaluation for one subject."""

    windows: np.ndarray
    tokens: np.ndarray
    labels: np.ndarray
    sessions: list[str]


@dataclass(frozen=True)
class AuditMetadata:
    """Provenance retained for audits but not used as model input."""

    encoder_load_info: dict[str, Any]
    encoder_mode: str
    manifest: dict[str, Any]


@dataclass
class FeatureBank:
    records: dict[str, SubjectRecord]
    samples: list[Any]
    encoder_state: dict[str, torch.Tensor]
    audit: AuditMetadata
