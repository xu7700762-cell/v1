from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolynomialKANLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, degree: int = 2) -> None:
        super().__init__()
        self.degree = int(degree)
        scale = 1.0 / math.sqrt(max(1, int(input_dim) * self.degree))
        self.poly_weight = nn.Parameter(
            torch.randn(int(output_dim), int(input_dim) * self.degree) * scale
        )
        self.bias = nn.Parameter(torch.zeros(int(output_dim)))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        limited = torch.tanh(value)
        terms = [limited]
        for _ in range(1, self.degree):
            terms.append(terms[-1] * limited)
        return F.linear(
            torch.cat(terms, dim=-1),
            self.poly_weight.to(dtype=value.dtype),
            self.bias.to(dtype=value.dtype),
        )


CHECKPOINT_COMPAT_PREFIXES = (
    "mamba_scale",
    "support_raw_scale",
    "gated_kan.",
    "gated_post_norm.",
)


def load_checkpoint_state_dict(
    model: nn.Module, state: Mapping[str, torch.Tensor], *, source: str = "checkpoint"
) -> tuple[str, ...]:
    """Load an A1 checkpoint while migrating keys from removed inactive branches.

    The locked checkpoints predate the release cleanup and contain parameters for
    branches that were never read by the forward path. Those keys are discarded
    explicitly; all active model keys still use strict loading.
    """
    filtered = dict(state)
    ignored = tuple(
        sorted(
            key
            for key in filtered
            if any(key == prefix or key.startswith(prefix) for prefix in CHECKPOINT_COMPAT_PREFIXES)
        )
    )
    for key in ignored:
        del filtered[key]
    try:
        model.load_state_dict(filtered, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"Invalid {source} after legacy checkpoint migration") from exc
    return ignored


class DirectionalMambaKAN(nn.Module):
    """Locked A1 directional state head used by the v1 checkpoints."""

    def __init__(self, dropout: float = 0.25) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba2
        except Exception as exc:  # pragma: no cover - checked in WSL preflight
            raise ImportError("DirectionalMambaKAN requires mamba_ssm.Mamba2.") from exc
        self.dropout = float(dropout)
        self.candidate = "A1_directional"
        self.kan_mode = "polykan"
        self.mixer_mode = "mamba"
        self.domain_norm_mode = "subject"
        self.input_norm = nn.LayerNorm(525)
        self.input_proj = nn.Sequential(nn.Linear(525, 96), nn.SiLU(), nn.Dropout(self.dropout))
        self.mamba = Mamba2(d_model=96, d_state=64, d_conv=4, expand=2)
        self.mamba_norm = nn.LayerNorm(96)
        self.legacy_domain_scale = nn.Parameter(torch.ones(96))
        self.domain_bias = nn.Parameter(torch.zeros(96))
        self.attention = nn.Sequential(
            nn.LayerNorm(96), nn.Linear(96, 96), nn.SiLU(), nn.Linear(96, 1, bias=False)
        )
        self.pool_norm = nn.LayerNorm(384)
        self.pool_proj = nn.Linear(384, 96)
        self.legacy_kan = PolynomialKANLayer(96, 160, degree=2)
        self.legacy_post_norm = nn.LayerNorm(160)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(math.expm1(9.0))))

    @property
    def embedding_dim(self) -> int:
        return 160

    def temperature(self) -> torch.Tensor:
        return F.softplus(self.log_temperature) + 1.0

    def encode_sequence(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-2:] != (80, 525):
            raise ValueError(f"Expected tokens ending in [80,525], got {tuple(tokens.shape)}")
        prefix = tokens.shape[:-2]
        flat = tokens.reshape(-1, 80, 525)
        sequence = self.input_proj(self.input_norm(flat))
        mixed = self.mamba(sequence)
        sequence = self.mamba_norm(
            sequence + F.dropout(mixed, p=self.dropout, training=self.training)
        )
        return sequence.reshape(*prefix, 80, 96)

    def normalize_domains(self, sequence: torch.Tensor, statistics: torch.Tensor) -> torch.Tensor:
        stats = statistics.float().detach()
        mean = stats.mean(dim=(1, 2), keepdim=True)
        variance = stats.var(dim=(1, 2), unbiased=False, keepdim=True)
        normalized = (sequence.float() - mean) / torch.sqrt(variance + 1e-4)
        return normalized * self.legacy_domain_scale[None, None, None] + self.domain_bias[None, None, None]

    def normalize_subject(self, sequence: torch.Tensor, statistics: torch.Tensor) -> torch.Tensor:
        stats = statistics.float()
        mean = stats.mean(dim=(0, 1), keepdim=True)
        variance = stats.var(dim=(0, 1), unbiased=False, keepdim=True)
        normalized = (sequence.float() - mean) / torch.sqrt(variance + 1e-4)
        return normalized * self.legacy_domain_scale[None, None] + self.domain_bias[None, None]

    def pool_embedding(self, sequence: torch.Tensor) -> torch.Tensor:
        prefix = sequence.shape[:-2]
        flat = sequence.reshape(-1, 80, 96)
        attention = torch.softmax(self.attention(flat).squeeze(-1), dim=1)
        pooled = torch.cat(
            [
                torch.sum(flat * attention.unsqueeze(-1), dim=1),
                flat.mean(dim=1),
                flat.std(dim=1, unbiased=False),
                flat.max(dim=1).values,
            ],
            dim=-1,
        )
        latent = F.dropout(
            F.silu(self.pool_proj(self.pool_norm(pooled))),
            p=self.dropout,
            training=self.training,
        )
        embedding = self.legacy_post_norm(self.legacy_kan(latent))
        return F.normalize(embedding, dim=-1).reshape(*prefix, self.embedding_dim)

    def forward_domain_batch(self, anchor_tokens: torch.Tensor) -> torch.Tensor:
        sequence = self.encode_sequence(anchor_tokens)
        return self.pool_embedding(self.normalize_domains(sequence, sequence))

    def model_spec(self) -> dict:
        return {
            "candidate": self.candidate,
            "dropout": self.dropout,
            "kan_mode": self.kan_mode,
            "mixer_mode": self.mixer_mode,
            "domain_norm_mode": self.domain_norm_mode,
        }

    @classmethod
    def from_model_spec(cls, spec: Optional[dict], candidate: str = "A1_directional", dropout: float = 0.25):
        values = dict(spec or {})
        if values.get("candidate", candidate) != "A1_directional":
            raise ValueError("v1 supports only the locked A1_directional head")
        return cls(float(values.get("dropout", dropout)))
