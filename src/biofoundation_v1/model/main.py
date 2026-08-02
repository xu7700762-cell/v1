from __future__ import annotations

import torch
import torch.nn as nn

from .a1 import DirectionalMambaKAN
from .femba import FEMBAEncoder
from .severity import PairSeverityHead


class BioFoundationV1(nn.Module):
    """Locked FEMBA + A1 + pair-severity training model."""

    def __init__(
        self,
        encoder_state: dict[str, torch.Tensor],
        *,
        a1_seed: int = 1001,
        dropout: float = 0.25,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = FEMBAEncoder()
        self.encoder.load_state_dict(encoder_state, strict=True)
        self.encoder_frozen = bool(freeze_encoder)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(not self.encoder_frozen)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(a1_seed))
            self.a1 = DirectionalMambaKAN(dropout)
        self.severity_head = PairSeverityHead(self.a1.embedding_dim, dropout, int(a1_seed) + 17)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def forward_domain_batch(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 4 or windows.shape[-2:] != (30, 1280):
            raise ValueError(f"Expected [domains,trials,30,1280], got {tuple(windows.shape)}")
        domains, trials = windows.shape[:2]
        if self.encoder_frozen:
            with torch.no_grad():
                tokens = self.encoder.forward_tokens(windows.reshape(-1, 30, 1280).float())
        else:
            tokens = self.encoder.forward_tokens(windows.reshape(-1, 30, 1280).float())
        return self.a1.forward_domain_batch(tokens.reshape(domains, trials, 80, 525))

    def severity_logits(
        self, reference_embeddings: list[torch.Tensor], task_embeddings: list[torch.Tensor]
    ) -> torch.Tensor:
        features = [
            self.severity_head.features(reference, task)
            for reference, task in zip(reference_embeddings, task_embeddings)
        ]
        return self.severity_head(torch.stack(features))

    def parameter_deltas(self, before: dict[str, torch.Tensor]) -> dict[str, float]:
        def delta(prefix: str) -> float:
            values = [
                (parameter.detach().cpu() - before[name]).reshape(-1)
                for name, parameter in self.named_parameters()
                if name.startswith(prefix)
            ]
            return float(torch.linalg.vector_norm(torch.cat(values))) if values else 0.0

        return {
            "encoder_delta": delta("encoder."),
            "a1_delta": delta("a1."),
            "severity_head_delta": delta("severity_head."),
        }
