"""Explicit command group for sequence replacement experiments."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

from .runner import ENCODERS


def configure(parser):
    commands = parser.add_subparsers(dest="sequence_command", required=True)
    importer = commands.add_parser("import-splits", help="Preserve existing split tables and labels")
    for split in ("train", "val", "test"):
        importer.add_argument(f"--{split}", type=Path, required=True)
    importer.add_argument("--output", type=Path, required=True)
    importer.add_argument("--id-column", default="row_uid")
    importer.add_argument("--smiles-column", default="substrate")
    importer.add_argument("--target-column", default="log10kcat_max")
    importer.add_argument("--sequence-column", default="sequence")
    importer.add_argument("--reaction-column", default="reaction_smiles")
    importer.add_argument("--group-column", default="cluster_id")
    importer.add_argument("--overwrite", action="store_true")
    prepare = commands.add_parser("prepare", help="Paired R/C conversion, roundtrip and coverage audit")
    prepare.add_argument("--samples", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--mapping-jsonl", type=Path)
    prepare.add_argument("--mapped-column", default="mapped_rxn")
    prepare.add_argument("--max-length", type=int, default=2048)
    prepare.add_argument("--on-error", choices=("error", "substrate"), default="error")
    prepare.add_argument("--overwrite", action="store_true")
    for name in ("validate", "train", "evaluate", "predict"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, required=True)
        if name in {"validate", "train"}:
            sub.add_argument("--encoder", choices=ENCODERS)
            sub.add_argument("--pooling", choices=("mean", "edit_conditioned"))
        if name != "validate":
            sub.add_argument("--output", type=Path, required=name != "train")
            sub.add_argument("--overwrite", action="store_true")
        if name in {"evaluate", "predict"}:
            sub.add_argument("--checkpoint", type=Path, required=True)
        if name != "train":
            sub.add_argument("--split", choices=("train", "val", "test"),
                             default="test" if name == "evaluate" else None)
    pretrain = commands.add_parser("pretrain", help="Train an encoder with masked language modeling")
    pretrain.add_argument("--config", type=Path, required=True,
                          help="Same model/training configuration; no protein cache is read")
    pretrain.add_argument("--train-corpus", type=Path, required=True)
    pretrain.add_argument("--val-corpus", type=Path, required=True)
    pretrain.add_argument("--exclude-corpus", type=Path)
    pretrain.add_argument("--text-column")
    pretrain.add_argument("--encoder", choices=ENCODERS[1:])
    pretrain.add_argument("--mask-probability", type=float, default=.15)
    pretrain.add_argument("--output", type=Path, required=True)
    pretrain.add_argument("--overwrite", action="store_true")


def dispatch(args) -> int:
    from . import prepare, runner

    try:
        command = args.sequence_command
        if command == "import-splits":
            result = prepare.import_splits(
                args.train, args.val, args.test, output_dir=args.output,
                id_column=args.id_column, smiles_column=args.smiles_column,
                target_column=args.target_column, sequence_column=args.sequence_column,
                reaction_column=args.reaction_column, group_column=args.group_column,
                overwrite=args.overwrite,
            )
        elif command == "prepare":
            result = prepare.prepare_sequences(
                args.samples, output_dir=args.output, mapping_jsonl=args.mapping_jsonl,
                mapped_column=args.mapped_column, max_length=args.max_length,
                on_error=args.on_error, overwrite=args.overwrite,
            )
        else:
            config = runner.load_config(args.config, encoder=getattr(args, "encoder", None),
                                        pooling=getattr(args, "pooling", None))
            if command == "validate":
                result = runner.validate(config, split=args.split)
            elif command == "train":
                if args.output:
                    config = replace(config, output_dir=args.output.resolve())
                result = runner.fit(config, overwrite=args.overwrite)
            elif command in {"evaluate", "predict"}:
                result = runner.infer(config, args.checkpoint, output_dir=args.output,
                                      split=args.split, evaluate=command == "evaluate",
                                      overwrite=args.overwrite)
            elif command == "pretrain":
                from .pretrain import pretrain

                if config.chemical_encoder == "catapro_features":
                    raise ValueError("MLM requires a sequence chemical_encoder")
                if config.encoder_checkpoint or config.freeze_encoder:
                    raise ValueError("MLM trains a new encoder; remove encoder_checkpoint/freeze_encoder")
                runner._protect_output(args.output.resolve(), (args.config, args.train_corpus,
                                                               args.val_corpus, args.exclude_corpus))
                result = pretrain(
                    args.train_corpus, args.val_corpus, output_dir=args.output,
                    model_config=config.model, training=config.training,
                    chemical_encoder=config.chemical_encoder, text_column=args.text_column,
                    exclude_path=args.exclude_corpus, mask_probability=args.mask_probability,
                    overwrite=args.overwrite,
                )
            else:
                raise ValueError(f"Unknown sequence command: {command}")
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, ImportError) as error:
        print(f"newkcat sequence: {error}", file=sys.stderr)
        return 2
