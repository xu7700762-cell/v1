from __future__ import annotations

import json
from pathlib import Path

from .protocol import EXPECTED, MAX_SCORE_ERROR


def verify_report(actual_path: str | Path) -> dict:
    with Path(actual_path).open("r", encoding="utf-8") as handle:
        actual = json.load(handle)
    failures = []
    if actual.get("status") != "complete":
        failures.append("aggregate report is not complete")
    for dataset, expected in EXPECTED.items():
        report = actual.get("datasets", {}).get(dataset)
        if report is None:
            failures.append(f"missing dataset report: {dataset}")
            continue
        state = float(report["state_metrics"]["accuracy"])
        severity = float(report["r4_severity_metrics"]["accuracy"])
        if abs(state - expected["state_accuracy"]) > 1e-12:
            failures.append(f"{dataset} state accuracy mismatch: {state}")
        if abs(severity - expected["severity_accuracy"]) > 1e-12:
            failures.append(f"{dataset} severity accuracy mismatch: {severity}")
        for task, reproduction in (
            ("state", report.get("state_reproduction", {})),
            ("severity", report.get("severity_reproduction", {})),
        ):
            if reproduction.get("prediction_mismatches") != 0:
                failures.append(f"{dataset} {task} prediction mismatch")
            if reproduction.get("label_mismatches") != 0:
                failures.append(f"{dataset} {task} label mismatch")
            observed = reproduction.get("max_abs_score_error")
            if observed is None:
                failures.append(f"{dataset} {task} score error is missing")
            elif float(observed) > MAX_SCORE_ERROR[dataset][task]:
                failures.append(
                    f"{dataset} {task} score error {observed} > {MAX_SCORE_ERROR[dataset][task]}"
                )
    result = {"status": "passed" if not failures else "failed", "failures": failures}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise AssertionError("\n".join(failures))
    return result
