"""Independent command line entry point; the original newkcat CLI is unchanged."""

from __future__ import annotations

import argparse
import json

from .runner import fit, infer, inspect_model, load_config, with_overrides


def _configuration_arguments(parser):
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples")
    parser.add_argument("--cache")
    parser.add_argument("--global-features")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    describe = commands.add_parser("describe", help="Show actual dimensions/parameters without data")
    describe.add_argument("--config")
    cache = commands.add_parser("cache", help="Extract complete frozen residue/token states")
    cache.add_argument("--samples", required=True)
    cache.add_argument("--output", required=True)
    cache.add_argument("--reaction-checkpoint", required=True,
                       help="Local RXNFP directory containing config, vocabulary and model weights")
    cache.add_argument("--protein-model", help="Local ProtT5 model directory or model ID")
    cache.add_argument("--protein-import-index", help="Existing complete per-residue NPZ index")
    cache.add_argument("--device", default="cpu")
    cache.add_argument("--reaction-overlap", type=int, default=128)
    cache.add_argument("--protein-window", type=int, default=1000)
    cache.add_argument("--protein-overlap", type=int, default=128)
    cache.add_argument("--allow-model-download", action="store_true",
                       help="Allow a missing protein model to be downloaded from its model hub")
    _configuration_arguments(commands.add_parser("train", help="Fit adapters/fusion using train+val"))
    for name, help_text in (("evaluate", "Evaluate a completed checkpoint on original val/test"),
                            ("predict", "Predict with a completed checkpoint without reading targets")):
        command = commands.add_parser(name, help=help_text)
        _configuration_arguments(command)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "describe":
            result = inspect_model(load_config(args.config) if args.config else None)
        elif args.command == "cache":
            from .data import create_cache
            from .encoders import ProteinResidueEncoder, ReactionTokenEncoder

            if not args.protein_model and not args.protein_import_index:
                raise ValueError("cache needs --protein-model or --protein-import-index")
            reaction_encoder = ReactionTokenEncoder(
                args.reaction_checkpoint, device=args.device, overlap=args.reaction_overlap,
            )
            protein_encoder = (
                ProteinResidueEncoder(
                    args.protein_model, device=args.device, window_size=args.protein_window,
                    overlap=args.protein_overlap, local_files_only=not args.allow_model_download,
                ) if args.protein_model else None
            )
            index = create_cache(
                args.samples, args.output, protein_encoder, reaction_encoder,
                protein_import_index=args.protein_import_index,
            )
            result = {"cache": args.output, "rows": len(index["samples"]),
                      "dimensions": index.get("dimensions"), "rows_dropped": 0,
                      "encoders": index.get("encoders")}
        else:
            config = with_overrides(
                load_config(args.config), samples=args.samples, cache=args.cache,
                global_features=args.global_features, output=args.output,
                seed=args.seed, device=args.device,
            )
            if args.command == "train":
                result = fit(config)
            else:
                result = infer(
                    config, args.checkpoint,
                    split=None if args.split == "all" else args.split,
                    evaluate=args.command == "evaluate",
                )
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
