from __future__ import annotations

from pathlib import Path

from .evaluation.runner import run_reproduction


def run_reproduce(config: dict, datasets: list[str], device: str, output_root: Path) -> dict:
    return run_reproduction(config, datasets, device, Path(output_root))
