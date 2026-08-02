from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .context import smooth_current_future
from .geometry import geometry_distance, lorentz_min_anchor_distance, stable_rank


THRESHOLD = 0.5


def fuse_state_evidence(evidence: np.ndarray) -> np.ndarray:
    values = np.asarray(evidence, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("State evidence must contain exactly R1/R2/R4")
    return (values[:, 0] + values[:, 1] + 2.0 * values[:, 2]) / 4.0


def uniform_anchor_mask(rows: list[dict], session: str) -> np.ndarray:
    mask = np.zeros(len(rows), dtype=bool)
    candidates = sorted(
        [index for index, row in enumerate(rows) if str(row["session"]) == str(session)],
        key=lambda index: int(rows[index]["window_index"]),
    )
    positions = np.rint(np.linspace(0, len(candidates) - 1, 8)).astype(np.int64)
    if len(np.unique(positions)) != 8:
        raise AssertionError(f"{session} cannot provide eight unique anchor slots")
    selected = [candidates[int(positions[position])] for position in (2, 3, 4, 5)]
    mask[np.asarray(selected, dtype=np.int64)] = True
    if int(mask.sum()) != 4:
        raise AssertionError("Expected exactly four U3-U6 anchors")
    return mask


def source_r4_rows(rows: list[dict], moments: np.ndarray, *, pca_dim: int = 2) -> list[dict]:
    moments = np.asarray(moments, dtype=np.float64)
    if len(rows) != len(moments):
        raise ValueError("Source rows and moments are not aligned")
    output = []
    for subject in sorted({str(row["subject_id"]) for row in rows}):
        indices = np.asarray(
            [index for index, row in enumerate(rows) if str(row["subject_id"]) == subject],
            dtype=np.int64,
        )
        current_rows = [rows[int(index)] for index in indices]
        anchor_mask = np.asarray(
            [bool(int(row["calibration_anchor"])) for row in current_rows], dtype=bool
        )
        if int(anchor_mask.sum()) != 4:
            raise AssertionError(f"{subject} must retain exactly four source anchors")
        smoothed = smooth_current_future(current_rows, moments[indices], 3)
        r4 = stable_rank(geometry_distance(smoothed, anchor_mask, int(pca_dim)))
        output.extend(
            {**row, "mamba_moments_evidence": float(value)}
            for row, value in zip(current_rows, r4)
        )
    output.sort(key=lambda row: int(row["sample_index"]))
    return output


def _logit(values: np.ndarray | float) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values))


def _target_baseline(feature_matrix: np.ndarray, anchor_margin: np.ndarray) -> np.ndarray:
    values = StandardScaler().fit_transform(np.asarray(feature_matrix, dtype=np.float64))
    clustering = KMeans(n_clusters=2, n_init=50, random_state=1001).fit(values)
    cluster_anchor = np.asarray(
        [anchor_margin[clustering.labels_ == cluster].mean() for cluster in (0, 1)]
    )
    positive = int(np.argmax(cluster_anchor))
    distances = clustering.transform(values)
    decision = distances[:, 1 - positive] - distances[:, positive]
    scale = max(float(decision.std()), 1e-6)
    return 1.0 / (1.0 + np.exp(-np.clip(decision / scale, -30, 30)))


def _prepared_rank(query: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    combined = np.vstack([query, anchors])
    mask = np.zeros(len(combined), dtype=bool)
    mask[len(query) :] = True
    distance, _ = lorentz_min_anchor_distance(combined, mask, 0.10)
    return stable_rank(distance[: len(query)])


def _cross_lorentz_rank(query: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    scaler = StandardScaler().fit(query)
    return _prepared_rank(scaler.transform(query), scaler.transform(anchors))


def _cross_geometry_rank(query: np.ndarray, anchors: np.ndarray, pca_dim: int) -> np.ndarray:
    scaler = StandardScaler().fit(query)
    query_scaled = scaler.transform(query)
    anchor_scaled = scaler.transform(anchors)
    from sklearn.decomposition import PCA

    effective = min(int(pca_dim), len(query_scaled) - 1, query_scaled.shape[1])
    pca = PCA(n_components=effective, random_state=1001).fit(query_scaled)
    projected_scaler = StandardScaler().fit(pca.transform(query_scaled))
    query_projected = projected_scaler.transform(pca.transform(query_scaled))
    anchor_projected = projected_scaler.transform(pca.transform(anchor_scaled))
    return _prepared_rank(query_projected, anchor_projected)


def state_rows(
    rows: list[dict], margins: np.ndarray, moments: np.ndarray, *, resmooth_refinement: bool
) -> list[dict]:
    rows = [dict(row) for row in rows]
    margins = np.asarray(margins, dtype=np.float64)
    moments = np.asarray(moments, dtype=np.float64)
    if margins.shape != (len(rows), 3) or moments.shape[0] != len(rows):
        raise ValueError("State rows and evidence tensors are not aligned")
    query_margins = smooth_current_future(rows, margins, 3)
    query_moments = smooth_current_future(rows, moments, 3)
    anchor_mask = np.asarray([bool(int(row["calibration_anchor"])) for row in rows])
    output = [dict(row) for row in rows]
    for subject in sorted({str(row["subject_id"]) for row in rows}):
        indices = np.asarray(
            [index for index, row in enumerate(rows) if str(row["subject_id"]) == subject],
            dtype=np.int64,
        )
        current_anchor = anchor_mask[indices]
        if int(current_anchor.sum()) != 4:
            raise AssertionError(f"{subject} must retain exactly four anchors")
        current = query_margins[indices]
        baseline = _target_baseline(current, current[:, :2].mean(axis=1))
        oriented = 1.0 - baseline if baseline[current_anchor].mean() >= 0.5 else baseline
        if resmooth_refinement:
            raw = margins[indices]
            refinement = smooth_current_future(
                [rows[index] for index in indices],
                np.column_stack([raw[:, :2], _logit(baseline)]),
                3,
            )
        else:
            refinement = np.column_stack([current[:, :2], _logit(baseline)])
        r1 = _cross_lorentz_rank(refinement, refinement[current_anchor])
        r2 = stable_rank(oriented)
        r4 = _cross_geometry_rank(query_moments[indices], query_moments[indices][current_anchor], 2)
        score = fuse_state_evidence(np.column_stack([r1, r2, r4]))
        prediction = (score >= THRESHOLD).astype(np.int64)
        prediction[current_anchor] = 0
        score[current_anchor] = np.minimum(score[current_anchor], np.nextafter(THRESHOLD, 0.0))
        for local_index, global_index in enumerate(indices):
            row = output[int(global_index)]
            row.update(
                multiview_evidence=float(r1[local_index]),
                oriented_baseline_evidence=float(r2[local_index]),
                mamba_moments_evidence=float(r4[local_index]),
                score=float(score[local_index]),
                threshold=THRESHOLD,
                y_pred=int(prediction[local_index]),
            )
            row["correct"] = int(row["y_pred"] == int(row["y_true"]))
    return output
