from __future__ import annotations

import argparse
import json
from pathlib import Path

from biofoundation_v1.config import load_config
from biofoundation_v1.evaluation.io import sha256_file, write_json
from biofoundation_v1.preflight import _check_environment, _check_protocol
from biofoundation_v1.protocol import PROTOCOL


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_WORKSPACE_FILES = (
    "models/FEMBA.py",
    "vr_ssq_regression/femba_encoder.py",
    "vr_ssq_regression/run_vrq_anchor_mambakan_identity_disjoint.py",
    "vr_ssq_regression/run_city_cruise_identity_disjoint.py",
    "vr_ssq_regression/run_monifeixing_vrsq_identity_disjoint.py",
    "vr_ssq_regression/random_femba_a1_vrsq_multitask/model.py",
    "vr_ssq_regression/random_femba_a1_vrsq_multitask/training.py",
    "vr_ssq_regression/fair_joint_lambda0p3_ablation/context.py",
    "vr_ssq_regression/fair_joint_lambda0p3_no_inner_seed42/run.py",
    "vr_ssq_regression/depth4_unified_scalar_temporal/core.py",
    "vr_ssq_regression/unified_temporal_severity_head/contextual_r4.py",
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
    for relative in SOURCE_WORKSPACE_FILES:
        path = asset_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing recorded source workspace file: {path}")
        rows.append(
            {"path": relative, "size": path.stat().st_size, "sha256": sha256_file(path)}
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
