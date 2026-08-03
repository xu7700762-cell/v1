from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .evaluate import run_evaluate
from .preflight import run_preflight
from .reproduce import run_reproduce
from .train import run_train
from .verify import verify_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vestibular_fusion")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("preflight", "reproduce", "train", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
    reproduce = sub.choices["reproduce"]
    reproduce.add_argument("--datasets", nargs="+", choices=("monifeixing", "vrq", "city"), default=["monifeixing", "vrq", "city"])
    reproduce.add_argument("--device", default="cuda")
    reproduce.add_argument("--output-root", type=Path)
    train = sub.choices["train"]
    train.add_argument("--dataset", choices=("monifeixing", "vrq", "city"), required=True)
    train.add_argument("--fold", type=int)
    train.add_argument("--device", default="cuda")
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--output-root", type=Path)
    evaluate = sub.choices["evaluate"]
    evaluate.add_argument("--dataset", choices=("monifeixing", "vrq", "city"), required=True)
    evaluate.add_argument("--checkpoint-root", type=Path)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--output-root", type=Path)

    verify = sub.add_parser("verify")
    verify.add_argument("--actual", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        verify_report(args.actual)
        return 0
    config = load_config(args.config)
    if args.command == "preflight":
        run_preflight(config)
        return 0
    if args.command == "reproduce":
        output_root = args.output_root or config["output_root"]
        run_preflight(config)
        run_reproduce(config, args.datasets, args.device, output_root)
        if set(args.datasets) == {"monifeixing", "vrq", "city"}:
            verify_report(output_root / "aggregate_report.json")
        return 0
    if args.command == "evaluate":
        run_preflight(config, reference_assets=False)
        checkpoint_root = args.checkpoint_root or (
            config["output_root"] / "training" / args.dataset
        )
        output_root = args.output_root or (
            config["output_root"] / "trained_evaluation" / args.dataset
        )
        report = run_evaluate(config, args.dataset, checkpoint_root, output_root, args.device)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "train":
        run_preflight(config, reference_assets=False)
        return int(
            run_train(
                config,
                args.dataset,
                args.fold,
                args.device,
                args.smoke,
                args.output_root.resolve() if args.output_root is not None else None,
            )
            or 0
        )
    raise AssertionError(args.command)
