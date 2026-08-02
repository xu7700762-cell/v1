from __future__ import annotations

from pathlib import Path

from .training.runner import run_training


def run_train(
    config: dict,
    dataset: str,
    fold: int | None,
    device: str,
    smoke: bool,
    output_root: Path | None = None,
) -> int:
    root = Path(output_root or config["output_root"] / "training" / dataset / f"fold_{fold or 'all'}")
    result = run_training(config, dataset, fold, device, smoke, root)
    print(result)
    return 0
