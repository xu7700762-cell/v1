from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from .evaluation.io import read_json, sha256_file
from .protocol import PROTOCOL


EXPECTED_PACKAGES = {
    "numpy": "2.2.6",
    "scipy": "1.15.3",
    "scikit-learn": "1.7.2",
    "openpyxl": "3.1.5",
    "joblib": "1.5.3",
    "torch": "2.11.0+cu128",
    "mamba-ssm": "2.3.1",
}


def _require(path: Path, label: str) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return {
        "label": label,
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _require_hash(path: Path, expected: str, label: str) -> dict:
    result = _require(path, label)
    if result["sha256"] != str(expected):
        raise RuntimeError(
            f"SHA-256 mismatch for {label}: expected {expected}, found {result['sha256']}"
        )
    return result


def _check_environment() -> dict:
    if tuple(sys.version_info[:2]) != (3, 10):
        raise RuntimeError(f"Python 3.10 is required; found {platform.python_version()}")
    if platform.system() != "Linux":
        raise RuntimeError("The locked reproduction environment must run under WSL2 Linux")
    release = Path("/etc/os-release").read_text(encoding="utf-8")
    if 'VERSION_ID="22.04"' not in release:
        raise RuntimeError("Ubuntu 22.04 is required by the locked reproduction protocol")
    versions = {}
    for distribution, expected in EXPECTED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Missing required package {distribution}=={expected}") from exc
        if actual != expected:
            raise RuntimeError(f"{distribution}=={expected} is required; found {actual}")
        versions[distribution] = actual
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the locked reproduction protocol")
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"CUDA 12.8 is required; found torch CUDA {torch.version.cuda}")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "torch_cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0),
    }


def _test_subject_key(fold: dict) -> str:
    for key in ("test_subjects", "vrq_test_subjects"):
        if key in fold:
            return key
    raise RuntimeError("A fold has no test subject list")


def _check_fivefold_subject_split(manifest: dict, label: str, *, fold_key: str = "folds") -> None:
    folds = manifest.get(fold_key)
    if not isinstance(folds, dict) or set(folds) != {f"fold_{index}" for index in range(1, 6)}:
        raise RuntimeError(f"{label} must contain exactly fold_1..fold_5")
    test_sets = []
    for fold_id, fold in sorted(folds.items()):
        if fold.get("status") not in (None, "complete"):
            raise RuntimeError(f"{label}/{fold_id} is not complete")
        test_key = _test_subject_key(fold)
        test = set(fold[test_key])
        source_groups = []
        if "source_outer_train_subjects" in fold:
            source_groups.append(set(fold["source_outer_train_subjects"]))
        else:
            source_groups.extend(
                set(fold[key]) for key in ("train_subjects", "val_subjects") if key in fold
            )
        if any(test & source for source in source_groups):
            raise RuntimeError(f"{label}/{fold_id} has source/test subject overlap")
        assertions = fold.get("identity_disjoint_assertions", {})
        if any(value not in (True, [], {}) for value in assertions.values()):
            raise RuntimeError(f"{label}/{fold_id} contains a failed identity assertion")
        test_sets.append((fold_id, test))
    for index, (left_id, left) in enumerate(test_sets):
        for right_id, right in test_sets[index + 1 :]:
            overlap = left & right
            if overlap:
                raise RuntimeError(
                    f"{label} test subjects overlap between {left_id} and {right_id}: {sorted(overlap)}"
                )


def _summary_digest(files: list[dict]) -> str:
    payload = "\n".join(
        f"{item['label']}\0{item['size']}\0{item['sha256']}" for item in sorted(files, key=lambda x: x["label"])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _check_source_tree() -> dict:
    package_root = Path(__file__).resolve().parent
    project_root = package_root.parents[1]
    forbidden_paths = [
        package_root / ("_" + "engine"),
        project_root / "src" / "models",
        project_root / "src" / "vr_vrsq_ea_encoder_hyperbolic",
    ]
    existing = [str(path) for path in forbidden_paths if path.exists()]
    if existing:
        raise RuntimeError(f"Historical source trees remain in v1: {existing}")
    forbidden_patterns = (
        "sys.path" + ".insert",
        "from " + "models import",
        "importlib" + ".import_module",
    )
    violations = []
    for path in package_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            if pattern in text:
                violations.append(f"{path.relative_to(project_root)}: {pattern}")
    if violations:
        raise RuntimeError("Forbidden source dependencies remain:\n" + "\n".join(violations))
    return {
        "package_root": str(package_root),
        "python_files": len(list(package_root.rglob("*.py"))),
        "forbidden_dependencies": 0,
    }


def _check_protocol(config: dict, *, reference_assets: bool) -> dict:
    paths = config["paths"]
    asset_root = Path(paths["asset_root"])
    vr_root = asset_root / "vr_ssq_regression"
    mono = vr_root / "artifacts_fair_joint_lambda0p3" / "monifeixing" / "lambda0p3" / "seed42" / "full"
    mono_outer = (
        vr_root
        / "artifacts_monifeixing_crossval"
        / "identity_disjoint_vrq_aligned_a1_init_seed1001_full"
        / "identity_disjoint_manifest.json"
    )
    vrq_root = vr_root / "artifacts_fair_joint_lambda0p3" / "vrq" / "seed_42"
    city_root = vr_root / "artifacts_city_a3_lambda_sweep_strict"
    baseline = vr_root / "fair_joint_lambda0p3_no_inner_seed42" / "artifacts"
    files = [
        _require(mono / "config.json", "monifeixing protocol config"),
        _require(mono / "report.json", "monifeixing report"),
        _require(mono / "severity_predictions.csv", "monifeixing severity labels"),
        _require(mono_outer, "monifeixing outer identity manifest"),
        _require(vrq_root / "main" / "full" / "audit_manifest.json", "VRQ audit manifest"),
        _require(vrq_root / "main" / "full" / "fold_manifest.json", "VRQ fold manifest"),
        _require(city_root / "audit" / "audit_manifest.json", "city audit manifest"),
        _require(city_root / "audit" / "fold_manifest.json", "city fold manifest"),
    ]
    mono_config = read_json(mono / "config.json")
    expected_mono = {
        "split_seed": 42,
        "training_seed": 1001,
        "severity_weight": 0.3,
        "encoder_frozen": True,
        "stage": "full",
    }
    if any(mono_config.get(key) != value for key, value in expected_mono.items()):
        raise RuntimeError("monifeixing config does not match the locked protocol")
    mono_report = read_json(mono / "report.json")
    _check_fivefold_subject_split(mono_report["identity_audit"], "monifeixing report")
    mono_manifest = read_json(mono_outer)
    if mono_manifest.get("status") != "complete":
        raise RuntimeError("monifeixing outer identity manifest is incomplete")
    _check_fivefold_subject_split(mono_manifest, "monifeixing outer identity manifest")
    vrq_manifest = read_json(vrq_root / "main" / "full" / "audit_manifest.json")
    if vrq_manifest.get("subject_split_seed") != 42 or vrq_manifest.get("training_seed") != 1001:
        raise RuntimeError("VRQ manifest does not match split_seed=42/training_seed=1001")
    _check_fivefold_subject_split(vrq_manifest, "VRQ audit manifest")
    city_manifest = read_json(city_root / "audit" / "audit_manifest.json")
    if city_manifest.get("split_seed") != 42:
        raise RuntimeError("city manifest does not match split_seed=42")
    if city_manifest["audit"].get("identity_overlap_count") != 0:
        raise RuntimeError("city manifest reports source/target identity overlap")
    _check_fivefold_subject_split(city_manifest["fold_manifest"], "city fold manifest")

    initial_hash = next(
        value
        for key, value in mono_config.items()
        if key.startswith("initial_") and key.endswith("_sha256")
    )
    files.append(
        _require_hash(
            Path(paths["monifeixing_initial_encoder"]),
            initial_hash,
            "monifeixing initial Temporal Encoder",
        )
    )
    pretrain_expected = vrq_manifest["run_fingerprint_payload"]["inputs"]["checkpoint_sha256"]
    files.append(
        _require_hash(Path(paths["pretrain_checkpoint"]), pretrain_expected, "pretrained Temporal Encoder")
    )
    for name, entry in mono_manifest["inputs"]["source_mat_files"].items():
        files.append(
            _require_hash(Path(paths["monifeixing_data_root"]) / name, entry["sha256"], f"monifeixing data {name}")
        )
    files.append(_require(Path(paths["monifeixing_workbook"]), "monifeixing questionnaire"))

    vrq_inputs = vrq_manifest["run_fingerprint_payload"]["inputs"]
    for name, expected in vrq_inputs["mat_sha256"].items():
        files.append(_require_hash(Path(paths["vrq_data_root"]) / name, expected, f"VRQ data {name}"))
    files.append(
        _require_hash(
            Path(paths["vrq_ssq_path"]), vrq_inputs["ssq_workbook_sha256"], "VRQ questionnaire"
        )
    )

    for subject, metadata in city_manifest["audit"]["subjects"].items():
        name = Path(metadata["mat_path"]).name
        files.append(
            _require_hash(
                Path(paths["city_data_root"]) / name,
                metadata["mat_sha256"],
                f"city data {subject}",
            )
        )
    city_inputs = city_manifest["audit"]["inputs"]
    files.extend(
        [
            _require_hash(
                Path(paths["city_record_workbook"]),
                city_inputs["record_workbook"]["sha256"],
                "city record workbook",
            ),
            _require_hash(
                Path(paths["city_ssq_workbook"]),
                city_inputs["ssq_workbook"]["sha256"],
                "city SSQ workbook",
            ),
            _require_hash(
                Path(paths["city_acq26_scores"]),
                city_inputs["acq26_scores"]["sha256"],
                "city acq26 scores",
            ),
            _require(Path(paths["city_source_vrsq_workbook"]), "city source identity workbook"),
        ]
    )

    if not reference_assets:
        return {"file_count": len(files), "summary_sha256": _summary_digest(files), "files": files}

    initialization = {
        row["initialization_id"]: row
        for row in mono_report["initialization_reports"]
        if row["kind"] == "outer"
    }
    for number in range(1, 6):
        fold_id = f"fold_{number}"
        item = initialization[fold_id]
        files.extend(
            [
                _require_hash(
                    asset_root / item["encoder"]["checkpoint"],
                    item["encoder"]["checkpoint_sha256"],
                    f"monifeixing {fold_id} encoder",
                ),
                _require_hash(
                    asset_root / item["a1_checkpoint"],
                    item["a1_checkpoint_sha256"],
                    f"monifeixing {fold_id} A1",
                ),
            ]
        )
        outer = mono / "outer_inputs" / fold_id
        files.extend(
            [
                _require(outer / "outer_rows.csv", f"monifeixing {fold_id} outer rows"),
                _require(outer / "outer_arrays.npz", f"monifeixing {fold_id} outer arrays"),
            ]
        )

        vrq_fold = vrq_root / "main" / "full" / "folds" / fold_id
        vrq_complete = read_json(vrq_fold / "complete.json")
        files.extend(
            [
                _require_hash(
                    vrq_fold / "refit.pt",
                    vrq_complete["artifact_sha256"]["refit.pt"],
                    f"VRQ {fold_id} checkpoint",
                ),
                _require_hash(
                    vrq_fold / "metrics.json",
                    vrq_complete["artifact_sha256"]["metrics.json"],
                    f"VRQ {fold_id} metrics",
                ),
                _require_hash(
                    vrq_fold / "state_predictions.csv",
                    vrq_complete["artifact_sha256"]["state_predictions.csv"],
                    f"VRQ {fold_id} anchor protocol",
                ),
            ]
        )

        city_fold = city_root / "lambda_0p3" / fold_id / "outer"
        city_complete = read_json(city_fold / "complete.json")
        files.extend(
            [
                _require_hash(
                    city_fold / "refit.pt",
                    city_complete["sha256"]["refit.pt"],
                    f"city {fold_id} checkpoint",
                ),
                _require_hash(
                    city_fold / "report.json",
                    city_complete["sha256"]["report.json"],
                    f"city {fold_id} report",
                ),
                _require_hash(
                    city_fold / "state_predictions.csv",
                    city_complete["sha256"]["state_predictions.csv"],
                    f"city {fold_id} anchor protocol",
                ),
            ]
        )
    for dataset in ("monifeixing", "vrq", "city"):
        files.extend(
            [
                _require(
                    baseline / "r_fusion_drop" / dataset / "full_no_r3" / "predictions.csv",
                    f"{dataset} authoritative state predictions",
                ),
                _require(
                    baseline / dataset / "r4_predictions.csv",
                    f"{dataset} authoritative severity predictions",
                ),
            ]
        )
    return {"file_count": len(files), "summary_sha256": _summary_digest(files), "files": files}


def run_preflight(config: dict, *, reference_assets: bool = True) -> dict:
    result = {
        "status": "passed",
        "protocol": PROTOCOL,
        "environment": _check_environment(),
        "source": _check_source_tree(),
        "assets": _check_protocol(config, reference_assets=reference_assets),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
