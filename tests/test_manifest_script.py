import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SPEC = spec_from_file_location(
    "release_generate_manifest", Path(__file__).resolve().parents[1] / "scripts" / "generate_manifest.py"
)
assert SPEC is not None and SPEC.loader is not None
generate_manifest = module_from_spec(SPEC)
SPEC.loader.exec_module(generate_manifest)


def test_manifest_generation_requests_reference_asset_checks(monkeypatch, tmp_path):
    calls = []

    def check_protocol(config, *, reference_assets):
        calls.append((config, reference_assets))
        return {"summary_sha256": "sha256", "files": []}

    monkeypatch.setattr(generate_manifest, "load_config", lambda path: {"paths": {"asset_root": path}})
    monkeypatch.setattr(generate_manifest, "_check_environment", lambda: {})
    monkeypatch.setattr(generate_manifest, "_check_protocol", check_protocol)
    monkeypatch.setattr(generate_manifest, "source_workspace_manifest", lambda path: [])
    monkeypatch.setattr(generate_manifest, "public_source_manifest", lambda: [])
    monkeypatch.setattr(generate_manifest, "write_json", lambda path, payload: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_manifest.py",
            "--config",
            "config.json",
            "--public-output",
            str(tmp_path / "public.json"),
            "--asset-output",
            str(tmp_path / "asset.json"),
        ],
    )

    assert generate_manifest.main() == 0
    assert calls == [({"paths": {"asset_root": "config.json"}}, True)]
