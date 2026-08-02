from __future__ import annotations

PROTOCOL = {
    "name": "non_nested_subject_fivefold_no_inner",
    "split_seed": 42,
    "training_seed": 1001,
    "severity_weight": 0.3,
    "outer_folds": 5,
    "inner_folds": 0,
    "forward_context": "[t, t+1, t+2] with right-edge replication",
    "context_crosses_session": False,
    "state_fusion": "(R1 + R2 + 2R4) / 4",
    "severity": "R4-only",
    "checkpoint_retraining": False,
    "interpretation": "optimistic_non_nested_severity_diagnostic",
}

EXPECTED = {
    "monifeixing": {"state_accuracy": 0.8840440824149497, "severity_accuracy": 0.8333333333333334},
    "vrq": {"state_accuracy": 0.8078602620087336, "severity_accuracy": 0.6956521739130435},
    "city": {"state_accuracy": 0.7783367226800814, "severity_accuracy": 0.6883116883116883},
}

MAX_SCORE_ERROR = {
    "monifeixing": {"state": 0.0, "severity": 0.0},
    "vrq": {"state": 0.0, "severity": 1.2e-4},
    "city": {"state": 8.6e-4, "severity": 1.3e-5},
}
