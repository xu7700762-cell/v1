from __future__ import annotations

import torch
import torch.nn as nn


class PairSeverityHead(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float, seed: int = 1018) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.network = nn.Sequential(
                nn.LayerNorm(int(embedding_dim) * 4),
                nn.Linear(int(embedding_dim) * 4, 96),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(96, 1),
            )

    @staticmethod
    def features(reference: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        if reference.ndim != 2 or task.ndim != 2:
            raise ValueError("Severity features require [windows,embedding] tensors")
        if reference.shape[1] != task.shape[1] or min(len(reference), len(task)) < 2:
            raise ValueError("Severity reference/task embeddings are incompatible")
        delta = task.mean(dim=0) - reference.mean(dim=0)
        return torch.cat(
            [
                delta,
                delta.abs(),
                reference.std(dim=0, unbiased=False),
                task.std(dim=0, unbiased=False),
            ]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)
