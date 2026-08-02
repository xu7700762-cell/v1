from __future__ import annotations

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def minkowski_inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    product = left * right
    return -product[..., 0] + product[..., 1:].sum(dim=-1)


def project_lorentz(spatial: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norm = spatial.norm(dim=-1, keepdim=True)
    safe = torch.where(norm > eps, spatial, spatial + eps)
    safe_norm = safe.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    limited = safe_norm.clamp_max(10.0)
    time = torch.cosh(limited)
    space = torch.sinh(limited) * safe / safe_norm
    return torch.cat([time, space], dim=-1)


def lorentz_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    argument = (-minkowski_inner(left, right)).double()
    root = torch.sqrt(torch.clamp_min(argument.pow(2) - 1.0, 1e-15))
    return torch.log(argument + root).to(left.dtype)


def lorentz_min_anchor_distance(
    normalized_features: np.ndarray, anchor_mask: np.ndarray, tangent_scale: float
) -> tuple[np.ndarray, float]:
    spatial = torch.as_tensor(
        np.asarray(normalized_features, dtype=np.float64) * float(tangent_scale),
        dtype=torch.float64,
    )
    mask = torch.as_tensor(np.asarray(anchor_mask, dtype=bool), dtype=torch.bool)
    if spatial.ndim != 2 or mask.shape != (len(spatial),) or not bool(mask.any()):
        raise ValueError("Lorentz distance requires a feature matrix and at least one anchor")
    with torch.no_grad():
        points = project_lorentz(spatial)
        anchors = points[mask]
        distances = lorentz_distance(points[:, None, :], anchors[None, :, :])
        minimum = distances.min(dim=1).values.cpu().numpy()
        constraint = float(torch.max(torch.abs(minkowski_inner(points, points) + 1.0)).cpu())
    return minimum, constraint


def geometry_distance(features: np.ndarray, anchor_mask: np.ndarray, pca_dim: int) -> np.ndarray:
    normalized = StandardScaler().fit_transform(np.asarray(features, dtype=np.float64))
    effective = min(int(pca_dim), len(normalized) - 1, normalized.shape[1])
    projected = PCA(n_components=effective, random_state=1001).fit_transform(normalized)
    projected = StandardScaler().fit_transform(projected)
    distance, _ = lorentz_min_anchor_distance(projected, anchor_mask, 0.10)
    return distance


def stable_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("stable_rank expects one non-empty vector")
    order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
    return order.astype(np.float64) / max(1, len(values) - 1)
