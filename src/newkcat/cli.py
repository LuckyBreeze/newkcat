"""Command-line interface. Heavy encoders are deliberately not auto-downloaded."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="newkcat", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "train", "evaluate", "predict", "audit-centers"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", required=True, type=Path)
        if command in {"validate", "train"}:
            sub.add_argument("--mode", choices=[
                "baseline", "global", "center", "shell", "conditioned", "learned"
            ], help="Override only model.mode for a controlled ablation")
        if command != "validate":
            sub.add_argument("--overwrite", action="store_true")
        if command == "train":
            sub.add_argument("--init-catapro", type=Path,
                             help="Official CataPro state_dict; newly added input columns start at zero")
            sub.add_argument("--output", type=Path, help="Override output_dir for this run")
        if command in {"evaluate", "predict", "audit-centers"}:
            sub.add_argument("--checkpoint", type=Path, required=True)
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--allow-reference-centers", action="store_true",
                             help="Explicitly declare reference/oracle reaction-center information")
            default = "test" if command == "evaluate" else "val" if command == "audit-centers" else None
            sub.add_argument("--split", choices=["train", "val", "test"], default=default,
                             help="predict defaults to all records; evaluate defaults to test")
    centers = subparsers.add_parser(
        "centers-from-reactions",
        help="Extract reference centers from complete atom-mapped reaction SMILES",
    )
    centers.add_argument("--samples", type=Path, required=True,
                         help="CSV with sample_id, smiles and reaction_smiles")
    centers.add_argument("--output", type=Path, required=True,
                         help="Destination reference-center JSONL file")
    centers.add_argument("--reaction-column", default="reaction_smiles")
    centers.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from dataclasses import replace

    from . import train

    try:
        if args.command == "centers-from-reactions":
            from .centers import reference_centers_from_csv

            count = reference_centers_from_csv(
                args.samples, args.output, reaction_column=args.reaction_column,
                overwrite=args.overwrite,
            )
            print(json.dumps({"n": count, "output": str(args.output.resolve()),
                              "kind": "reference", "information_regime": "reference_center_diagnostic"}))
            return 0
        config = train.load_config(args.config, mode=getattr(args, "mode", None))
        if args.command == "validate":
            result = train.validate(config)
        elif args.command == "train":
            if args.output:
                config = replace(config, output_dir=args.output.resolve())
            result = train.fit(config, overwrite=args.overwrite, init_catapro=args.init_catapro)
        else:
            operation = {
                "evaluate": train.evaluate,
                "predict": train.predict,
                "audit-centers": train.audit_centers,
            }[args.command]
            result = operation(
                config, args.checkpoint, split=args.split, output_dir=args.output,
                overwrite=args.overwrite, allow_reference=args.allow_reference_centers,
            )
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        print(f"newkcat: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
