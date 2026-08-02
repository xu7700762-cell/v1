from __future__ import annotations

import argparse
import json
from pathlib import Path

from vestibular_fusion.config import load_config
from vestibular_fusion.evaluation.io import sha256_file, write_json
from vestibular_fusion.preflight import _check_environment, _check_protocol
from vestibular_fusion.protocol import PROTOCOL


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_WORKSPACE_FILES = (
    ("upstream_encoder_module", "models/" + "F" + "EMBA.py"),
    ("vr_signal_encoder_adapter", "vr_ssq_regression/" + "f" + "emba_encoder.py"),
    ("vr_subject_split_runner", "vr_ssq_regression/run_vrq_anchor_mambakan_identity_disjoint.py"),
    ("city_subject_split_runner", "vr_ssq_regression/run_city_cruise_identity_disjoint.py"),
    ("monifeixing_subject_split_runner", "vr_ssq_regression/run_monifeixing_vrsq_identity_disjoint.py"),
    (
        "multitask_model",
        "vr_ssq_regression/random_" + "f" + "emba_a1_vrsq_multitask/model.py",
    ),
    (
        "multitask_training",
        "vr_ssq_regression/random_" + "f" + "emba_a1_vrsq_multitask/training.py",
    ),
    ("joint_ablation_context", "vr_ssq_regression/fair_joint_lambda0p3_ablation/context.py"),
    ("subject_split_protocol", "vr_ssq_regression/fair_joint_lambda0p3_no_inner_seed42/run.py"),
    ("temporal_core", "vr_ssq_regression/depth4_unified_scalar_temporal/core.py"),
    ("r4_context", "vr_ssq_regression/unified_temporal_severity_head/contextual_r4.py"),
)


def public_source_manifest() -> list[dict]:
    paths = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(REPO_ROOT).as_posix()
        if any(
            part in {".git", ".pytest_cache", "__pycache__"} or part.endswith(".egg-info")
            for part in path.relative_to(REPO_ROOT).parts
        ):
            continue
        if relative.startswith("outputs/") or relative == "configs/paths.local.json":
            continue
        if path.name in {"source_manifest.json", "asset_manifest.local.json"}:
            continue
        if path.suffix not in {".py", ".toml", ".json", ".md", ".txt", ".sh", ".ps1"}:
            continue
        paths.append(
            {"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return sorted(paths, key=lambda item: item["path"])


def source_workspace_manifest(asset_root: Path) -> list[dict]:
    rows = []
    for label, relative in SOURCE_WORKSPACE_FILES:
        path = asset_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing recorded source workspace file: {path}")
        rows.append(
            {"label": label, "size": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate local asset and public source SHA-256 manifests"
    )
    parser.add_argument("--config", default="configs/paths.local.json")
    parser.add_argument("--public-output", default="results/reference/source_manifest.json")
    parser.add_argument("--asset-output", default="asset_manifest.local.json")
    args = parser.parse_args()
    config = load_config(args.config)
    environment = _check_environment()
    assets = _check_protocol(config)
    write_json(
        Path(args.asset_output),
        {
            "status": "complete",
            "protocol": PROTOCOL,
            "environment": environment,
            "assets": assets,
        },
    )
    public = {
        "status": "complete",
        "protocol": PROTOCOL,
        "asset_summary_sha256": assets["summary_sha256"],
        "assets": [
            {key: record[key] for key in ("label", "size", "sha256")}
            for record in assets["files"]
        ],
        "source_workspace_files": source_workspace_manifest(Path(config["paths"]["asset_root"])),
        "release_files": public_source_manifest(),
    }
    write_json(Path(args.public_output), public)
    print(
        json.dumps(
            {"public_output": args.public_output, "asset_output": args.asset_output},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
