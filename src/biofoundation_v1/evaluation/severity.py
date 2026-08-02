from __future__ import annotations

import json
from collections.abc import Iterable, Mapping

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import subject_sort_key


FORBIDDEN_OUTER_LABEL_FIELDS = frozenset(
    {"y_true", "ssq_score", "vrsq_post_total", "path_score", "delta_ssq", "label", "target"}
)


def winsorized_std(values: Iterable[float]) -> float:
    ordered = np.sort(np.asarray(list(values), dtype=np.float64).reshape(-1))
    if len(ordered) < 4 or not np.isfinite(ordered).all():
        raise ValueError("Winsorized population std requires at least four finite values")
    ordered = ordered.copy()
    ordered[0] = ordered[1]
    ordered[-1] = ordered[-2]
    return float(ordered.std(ddof=0))


def winsorized_std_preserve_order(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if len(values) < 4 or not np.isfinite(values).all():
        raise ValueError("Winsorized population std requires at least four finite values")
    ordered = np.sort(values)
    return float(np.clip(values, ordered[1], ordered[-2]).std(ddof=0))


def subject_r4_features(
    r4_rows: list[dict],
    task_session_by_subject: Mapping[str, str],
    *,
    preserve_window_order: bool = False,
) -> list[dict]:
    """Build one R4-only feature from 11 uniformly spaced task windows per subject."""

    output = []
    for subject in sorted(task_session_by_subject, key=subject_sort_key):
        session = str(task_session_by_subject[subject])
        task = sorted(
            (
                row
                for row in r4_rows
                if str(row["subject_id"]) == subject and str(row["session"]) == session
            ),
            key=lambda row: int(row["window_index"]),
        )
        if len(task) < 11:
            raise ValueError(f"{subject}/{session} has only {len(task)} task windows")
        positions = np.rint(np.linspace(0, len(task) - 1, 11)).astype(np.int64)
        if len(np.unique(positions)) != 11:
            raise AssertionError(f"{subject}/{session} cannot provide 11 unique task windows")
        selected = [task[int(position)] for position in positions]
        values = np.asarray(
            [float(row["mamba_moments_evidence"]) for row in selected], dtype=np.float64
        )
        dispersion = (
            winsorized_std_preserve_order(values)
            if preserve_window_order
            else winsorized_std(values)
        )
        output.append(
            {
                "subject_id": subject,
                "task_session": session,
                "R4_winsorized_std": dispersion,
                "severity_feature": -dispersion,
                "num_task_windows": 11,
                "selected_task_window_indices": json.dumps(
                    [int(row["window_index"]) for row in selected], separators=(",", ":")
                ),
            }
        )
    return output


def fit_source_head(rows: list[dict], *, seed: int = 1001) -> tuple[Pipeline, float]:
    features = np.asarray([[float(row["severity_feature"])] for row in rows], dtype=np.float64)
    labels = np.asarray([int(row["y_true"]) for row in rows], dtype=np.int64)
    if features.shape != (len(rows), 1) or not np.isfinite(features).all():
        raise ValueError("Severity head requires one finite source feature")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("The source fold must contain both severity classes")
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=0.1,
                    penalty="l2",
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=3000,
                    random_state=int(seed),
                ),
            ),
        ]
    ).fit(features, labels)
    return model, float(model.named_steps["classifier"].coef_[0, 0])


def score_outer(model: Pipeline, rows: list[dict]) -> list[dict]:
    if any(FORBIDDEN_OUTER_LABEL_FIELDS & set(row) for row in rows):
        raise ValueError("Outer scoring rows contain forbidden severity labels")
    features = np.asarray([[float(row["severity_feature"])] for row in rows], dtype=np.float64)
    if features.shape != (len(rows), 1) or not np.isfinite(features).all():
        raise ValueError("Outer severity features are invalid")
    probabilities = model.predict_proba(features)[:, 1]
    return [
        {**row, "score": float(score), "threshold": 0.5, "y_pred": int(score >= 0.5)}
        for row, score in zip(rows, probabilities)
    ]


def attach_subject_labels(
    scored_rows: list[dict], labels: Mapping[str, Mapping[str, float | int]], score_field: str
) -> list[dict]:
    expected = {str(row["subject_id"]) for row in scored_rows}
    if expected != set(labels):
        raise ValueError("Outer severity labels do not match the already scored subjects")
    output = []
    for row in scored_rows:
        label = labels[str(row["subject_id"])]
        current = {
            **row,
            score_field: float(label[score_field]),
            "y_true": int(label["y_true"]),
        }
        current["correct"] = int(current["y_pred"] == current["y_true"])
        output.append(current)
    return output


def _city_segment_lookup(audit: dict) -> dict[tuple[str, str], dict]:
    lookup = {}
    for subject, metadata in audit["subjects"].items():
        for segment in metadata.get("segments", []):
            alias = (
                "rest01"
                if segment["session"] == metadata["anchor_session"]
                else f"{segment['state']}_seg_{int(segment['segment_index']):02d}"
            )
            lookup[(subject, alias)] = segment
    return lookup


def city_path_r4_rows(state_rows: list[dict], audit: dict) -> list[dict]:
    lookup = _city_segment_lookup(audit)
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in state_rows:
        key = (str(row["subject_id"]), str(row["session"]))
        segment = lookup.get(key)
        if segment is not None and segment.get("path_score") is not None:
            grouped.setdefault(key, []).append(row)
    output = []
    for key, rows in sorted(grouped.items(), key=lambda item: (subject_sort_key(item[0][0]), item[0][1])):
        rows.sort(key=lambda row: int(row["window_index"]))
        positions = np.rint(np.linspace(0, len(rows) - 1, 11)).astype(np.int64)
        if len(np.unique(positions)) != 11:
            raise ValueError(f"{key} cannot provide 11 unique severity windows")
        selected = [rows[int(position)] for position in positions]
        segment = lookup[key]
        output.append(
            {
                "subject_id": key[0],
                "session": key[1],
                "route_order": int(segment["route_order"]),
                "route_id": int(segment["route_id"]),
                "r4_values": [float(row["mamba_moments_evidence"]) for row in selected],
            }
        )
    return output


def city_cumulative_features(path_rows: list[dict]) -> list[dict]:
    output = []
    for subject in sorted({str(row["subject_id"]) for row in path_rows}, key=subject_sort_key):
        rows = sorted(
            (row for row in path_rows if str(row["subject_id"]) == subject),
            key=lambda row: int(row["route_order"]),
        )
        history: list[dict] = []
        for row in rows:
            history.append(row)
            values = np.concatenate(
                [np.asarray(item["r4_values"], dtype=np.float64) for item in history]
            )
            dispersion = winsorized_std(values)
            output.append(
                {
                    "subject_id": subject,
                    "session": row["session"],
                    "route_order": int(row["route_order"]),
                    "route_id": int(row["route_id"]),
                    "num_source_paths": len(history),
                    "R4_history_winsorized_std": dispersion,
                    "severity_feature": -dispersion,
                }
            )
    return output


def contextualize_city(rows: list[dict]) -> list[dict]:
    output = []
    for subject in sorted({str(row["subject_id"]) for row in rows}, key=subject_sort_key):
        current = [row for row in rows if str(row["subject_id"]) == subject]
        raw = np.asarray([float(row["severity_feature"]) for row in current], dtype=np.float64)
        if len(current) == 1:
            values = raw
            mode = "single_record_raw"
        else:
            values = (raw - raw.mean()) / max(float(raw.std(ddof=0)), 1e-6)
            mode = "repeated_record_subject_zscore"
        output.extend(
            {
                **row,
                "raw_severity_feature": float(raw[index]),
                "contextual_feature": float(values[index]),
                "context_mode": mode,
                "subject_record_count": int(len(current)),
            }
            for index, row in enumerate(current)
        )
    return output


def _city_sample_weights(rows: list[dict]) -> np.ndarray:
    labels = np.asarray([int(row["y_true"]) for row in rows], dtype=np.int64)
    subjects = np.asarray([str(row["subject_id"]) for row in rows], dtype=object)
    counts = {subject: int(np.sum(subjects == subject)) for subject in set(subjects)}
    weights = np.asarray([1.0 / counts[subject] for subject in subjects], dtype=np.float64)
    for label in (0, 1):
        mask = labels == label
        if not mask.any():
            raise ValueError("City source fold requires both severity classes")
        weights[mask] *= 0.5 / weights[mask].sum()
    return weights * len(weights) / weights.sum()


def attach_city_labels(rows: list[dict], audit: dict) -> list[dict]:
    labels = {
        (str(row["subject_id"]), int(row["route_order"])): row for row in audit["path_labels"]
    }
    output = []
    for row in rows:
        label = labels[(str(row["subject_id"]), int(row["route_order"]))]
        current = {
            **row,
            "path_score": float(label["path_score"]),
            "y_true": int(label["path_label"]),
        }
        if "y_pred" in current:
            current["correct"] = int(current["y_pred"] == current["y_true"])
        output.append(current)
    return output


def fit_city_head(rows: list[dict]) -> Pipeline:
    features = np.asarray([[float(row["contextual_feature"])] for row in rows], dtype=np.float64)
    labels = np.asarray([int(row["y_true"]) for row in rows], dtype=np.int64)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=0.1,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=3000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(features, labels, classifier__sample_weight=_city_sample_weights(rows))
    return model


def score_city_outer(model: Pipeline, rows: list[dict]) -> list[dict]:
    if any(FORBIDDEN_OUTER_LABEL_FIELDS & set(row) for row in rows):
        raise ValueError("City outer rows contain severity labels before scoring")
    features = np.asarray([[float(row["contextual_feature"])] for row in rows], dtype=np.float64)
    probabilities = model.predict_proba(features)[:, 1]
    return [
        {
            **row,
            "endpoint": "city_path",
            "score": float(score),
            "threshold": 0.5,
            "y_pred": int(score >= 0.5),
        }
        for row, score in zip(rows, probabilities)
    ]


def evaluate_city_r4(
    source_state: list[dict], outer_state: list[dict], audit: dict
) -> tuple[list[dict], float]:
    source = contextualize_city(city_cumulative_features(city_path_r4_rows(source_state, audit)))
    outer = contextualize_city(city_cumulative_features(city_path_r4_rows(outer_state, audit)))
    labeled_source = attach_city_labels(source, audit)
    model = fit_city_head(labeled_source)
    coefficient = float(model.named_steps["classifier"].coef_[0, 0])
    return attach_city_labels(score_city_outer(model, outer), audit), coefficient
