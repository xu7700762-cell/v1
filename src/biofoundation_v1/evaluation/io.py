from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


INTEGER_FIELDS = {
    "sample_index",
    "window_index",
    "local_index",
    "y_true",
    "y_pred",
    "calibration_anchor",
    "correct",
    "route_order",
    "route_id",
    "num_task_windows",
    "num_source_paths",
    "subject_record_count",
}
FLOAT_FIELDS = {
    "score",
    "threshold",
    "mambakan_score",
    "ssq_score",
    "vrsq_post_total",
    "path_score",
    "multiview_evidence",
    "oriented_baseline_evidence",
    "mamba_moments_evidence",
    "severity_feature",
    "R4_winsorized_std",
    "R4_history_winsorized_std",
    "raw_severity_feature",
    "contextual_feature",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def rows_sha256(rows: list[dict]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if value in (None, ""):
                continue
            if key in INTEGER_FIELDS:
                row[key] = int(value)
            elif key in FLOAT_FIELDS:
                row[key] = float(value)
    return rows


def write_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".json", delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    if any(set(row) != set(fieldnames) for row in rows):
        raise ValueError(f"CSV rows have inconsistent fields: {path}")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8-sig", newline="", dir=path.parent, suffix=".csv", delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def comparison(reference: list[dict], current: list[dict]) -> dict:
    def key(row: dict):
        if "sample_index" in row:
            return str(row["subject_id"]), str(row.get("fold_id", "")), int(row["sample_index"])
        route = int(row["route_order"]) if "route_order" in row else None
        return str(row["subject_id"]), str(row.get("fold_id", "")), route

    left = {key(row): row for row in reference}
    right = {key(row): row for row in current}
    if left.keys() != right.keys():
        raise AssertionError("Reference/current prediction rows are not aligned")
    return {
        "max_abs_score_error": max(
            abs(float(left[item]["score"]) - float(right[item]["score"])) for item in left
        ),
        "prediction_mismatches": sum(
            int(left[item]["y_pred"]) != int(right[item]["y_pred"]) for item in left
        ),
        "label_mismatches": sum(
            int(left[item]["y_true"]) != int(right[item]["y_true"]) for item in left
        ),
    }
