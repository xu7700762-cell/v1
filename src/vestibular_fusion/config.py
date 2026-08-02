from __future__ import annotations

import json
import os
from pathlib import Path


REQUIRED_PATH_KEYS = (
    "asset_root",
    "monifeixing_data_root",
    "monifeixing_workbook",
    "vrq_data_root",
    "vrq_ssq_path",
    "city_data_root",
    "city_record_workbook",
    "city_ssq_workbook",
    "city_acq26_scores",
    "city_source_vrsq_workbook",
    "pretrain_checkpoint",
    "monifeixing_initial_encoder",
)


def native_path(value: str | Path) -> Path:
    """Resolve Windows paths when running from the WSL reproduction environment."""

    text = os.fspath(value)
    if len(text) >= 3 and text[1:3] == ":\\":
        drive = text[0].lower()
        rest = text[3:].replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")
    if len(text) >= 3 and text[1:3] == ":/":
        drive = text[0].lower()
        return Path(f"/mnt/{drive}/{text[3:]}")
    return Path(text).expanduser()


def load_config(path: str | Path) -> dict:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("paths"), dict):
        raise ValueError("Config must contain a top-level 'paths' object.")
    missing = [key for key in REQUIRED_PATH_KEYS if key not in config["paths"]]
    if missing:
        raise ValueError(f"Config is missing required paths: {', '.join(missing)}")
    config["_config_path"] = str(config_path)
    config["paths"] = {key: native_path(value) for key, value in config["paths"].items()}
    config["output_root"] = native_path(config.get("output_root", "outputs/seed42"))
    return config


def path(config: dict, name: str) -> Path:
    return config["paths"][name]
