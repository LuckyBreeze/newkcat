"""Standalone CLI: python -m newkcat.readout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def build_parser():
    parser = argparse.ArgumentParser(description="Complete-reaction readout ablations")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect-model", help="Inspect architecture without reading data")
    inspect.add_argument("--config", type=Path)
    for name in ("validate", "train", "evaluate", "predict"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, required=True)
        sub.add_argument("--samples", type=Path, help="Existing prepared sample table; never copied")
        sub.add_argument("--features", type=Path, help="Existing ID-aligned protein cache")
        sub.add_argument("--seed", type=int)
        sub.add_argument("--device")
        sub.add_argument("--output", type=Path, required=name in {"evaluate", "predict"})
        if name != "validate":
            sub.add_argument("--overwrite", action="store_true")
        if name in {"evaluate", "predict"}:
            sub.add_argument("--checkpoint", type=Path, required=True)
        if name != "train":
            sub.add_argument("--split", choices=("train", "val", "test"),
                             default="test" if name == "evaluate" else None)
    audit = commands.add_parser("audit", help="Read saved predictions; never train or select weights")
    audit.add_argument("--samples", type=Path, required=True)
    audit.add_argument("--prediction", action="append", default=[], metavar="MODEL=CSV")
    audit.add_argument("--split", choices=("train", "val", "test"), default="test")
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--group-column", default="group_id")
    audit.add_argument("--bootstrap-repetitions", type=int, default=0)
    audit.add_argument("--bootstrap-seed", type=int, default=668)
    audit.add_argument("--sample-target-column", default="log10_kcat")
    audit.add_argument("--prediction-target-column")
    audit.add_argument("--prediction-column")
    return parser


def dispatch(args) -> dict:
    if args.command == "audit":
        from .audit import audit_predictions

        predictions = {}
        for item in args.prediction:
            name, separator, filename = item.partition("=")
            if not separator or not name.strip() or not filename.strip():
                raise ValueError("Each --prediction must be MODEL=CSV")
            if name in predictions:
                raise ValueError(f"Duplicate prediction model: {name}")
            predictions[name] = Path(filename).expanduser().resolve()
        return audit_predictions(
            args.samples, predictions, args.split, args.output, group_column=args.group_column,
            bootstrap_repetitions=args.bootstrap_repetitions, bootstrap_seed=args.bootstrap_seed,
            sample_target_column=args.sample_target_column,
            prediction_target_column=args.prediction_target_column, prediction_column=args.prediction_column,
        )
    from . import runner

    if args.command == "inspect-model":
        return runner.inspect_model(runner.load_config(args.config) if args.config else None)
    config = runner.with_overrides(
        runner.load_config(args.config), samples=args.samples, features=args.features,
        seed=args.seed, device=args.device, output=args.output,
    )
    if args.command == "validate":
        result = runner.validate(config, split=args.split)
        if args.output:
            output = args.output.expanduser().resolve()
            runner._protect_output(output, (config.samples, config.features, config.config_path))
            runner._prepare_output(output, False)
            runner._json_write(output / "validation.json", result)
        return result
    if args.command == "train":
        return runner.fit(config, overwrite=args.overwrite)
    return runner.infer(config, args.checkpoint, output_dir=args.output, split=args.split,
                        evaluate=args.command == "evaluate", overwrite=args.overwrite)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = dispatch(args)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, ImportError) as error:
        print(f"newkcat readout: {error}", file=sys.stderr)
        return 2
