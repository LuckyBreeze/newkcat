"""Controlled reaction readout ablations with locked, independently evaluated weights.

This runner reuses the audited reaction data contract. It never changes input
rows, labels, features, or splits, and fitting never materializes test targets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from itertools import islice
import json
import math
from pathlib import Path
import uuid

import torch
from torch.utils.data import DataLoader, Sampler
import yaml

from ..train import (
    TrainingConfig, _UniqueConfigLoader, _check_test_overlap, _git_provenance,
    _json_write, _mapping, _prepare_output, _record_identity, checked_mse,
    seed_everything, sha256_file,
)
from ..sequence.data import collate_sequences
from ..sequence.runner import (
    _dataset, _hashes, _load_initial_encoder, _loader, _protect_output,
    _safe_load, _verify_preparation, _write_predictions, collect_predictions,
)
from ..sequence.tokenizer import ReactionTokenizer
from .model import ReactionReadoutModel, ReadoutModelConfig


CHECKPOINT_KIND = "newkcat_reaction_readout_regressor"
CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class ReadoutTrainingConfig(TrainingConfig):
    """The shared training options, with singleton batches allowed for LayerNorm."""

    batch_size: int = 1
    grad_accum_steps: int = 1

    def __post_init__(self):
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("training.batch_size must be a positive integer")
        if type(self.grad_accum_steps) is not int or self.grad_accum_steps < 1:
            raise ValueError("training.grad_accum_steps must be a positive integer")
        # Preserve the legacy validator without modifying its BatchNorm contract.
        common = {field.name: getattr(self, field.name) for field in fields(TrainingConfig)}
        common["batch_size"] = max(2, self.batch_size)
        TrainingConfig(**common)
        if self.allow_reference_centers:
            raise ValueError("Readout experiments do not use reference-center flags")


@dataclass(frozen=True)
class ReadoutRunConfig:
    config_path: Path
    samples: Path
    features: Path
    model: ReadoutModelConfig
    training: ReadoutTrainingConfig
    output_dir: Path
    chemical_encoder: str = "reaction_smiles"
    missing_reaction: str = "error"
    encoder_checkpoint: Path | None = None
    freeze_encoder: bool = False

    def __post_init__(self):
        if self.chemical_encoder != "reaction_smiles":
            raise ValueError("Readout ablations require chemical_encoder: reaction_smiles")
        if self.missing_reaction != "error":
            raise ValueError("Readout ablations require complete reactions; missing_reaction: error")
        if type(self.freeze_encoder) is not bool:
            raise ValueError("freeze_encoder must be a YAML boolean")
        if self.freeze_encoder and self.encoder_checkpoint is None:
            raise ValueError("Do not freeze a random encoder: supply encoder_checkpoint")

    def snapshot(self) -> dict:
        return {
            "data": {"samples": str(self.samples), "features": str(self.features),
                     "missing_reaction": self.missing_reaction},
            "model": {"chemical_encoder": self.chemical_encoder, **asdict(self.model)},
            "training": {key: value for key, value in asdict(self.training).items()
                         if key != "allow_reference_centers"},
            "output_dir": str(self.output_dir),
            "encoder_checkpoint": str(self.encoder_checkpoint) if self.encoder_checkpoint else None,
            "freeze_encoder": self.freeze_encoder,
        }


def load_config(path: str | Path) -> ReadoutRunConfig:
    path = Path(path).expanduser().resolve()
    try:
        with path.open(encoding="utf-8") as handle:
            raw = _mapping(yaml.load(handle, Loader=_UniqueConfigLoader), "readout config")
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML: {error}") from error
    allowed = {"data", "model", "training", "output_dir", "encoder_checkpoint", "freeze_encoder"}
    if set(raw) - allowed:
        raise ValueError(f"Unknown readout config fields: {sorted(set(raw) - allowed)}")

    def resolve(value, label):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must be a nonempty path")
        value = Path(value).expanduser()
        return (path.parent / value).resolve() if not value.is_absolute() else value.resolve()

    data = _mapping(raw.get("data"), "data")
    if set(data) - {"samples", "features", "missing_reaction"}:
        raise ValueError("Readout data supports samples, features and missing_reaction")
    model_raw = dict(_mapping(raw.get("model", {}), "model"))
    representation = model_raw.pop("chemical_encoder", "reaction_smiles")
    if set(model_raw) - {field.name for field in fields(ReadoutModelConfig)}:
        raise ValueError("Unknown readout model fields")
    model = ReadoutModelConfig(**model_raw)
    options = dict(_mapping(raw.get("training", {}), "training"))
    training_fields = {field.name for field in fields(ReadoutTrainingConfig)} - {
        "allow_reference_centers"
    }
    if set(options) - training_fields:
        raise ValueError(f"Unknown training fields: {sorted(set(options) - training_fields)}")
    training = ReadoutTrainingConfig(**options)
    checkpoint = (resolve(raw["encoder_checkpoint"], "encoder_checkpoint")
                  if raw.get("encoder_checkpoint") is not None else None)
    return ReadoutRunConfig(
        config_path=path, samples=resolve(data.get("samples"), "data.samples"),
        features=resolve(data.get("features"), "data.features"), model=model, training=training,
        output_dir=resolve(raw.get("output_dir", "runs/readout"), "output_dir"),
        chemical_encoder=representation, missing_reaction=data.get("missing_reaction", "error"),
        encoder_checkpoint=checkpoint, freeze_encoder=raw.get("freeze_encoder", False),
    )


def with_overrides(config: ReadoutRunConfig, *, samples=None, features=None,
                   seed=None, device=None, output=None) -> ReadoutRunConfig:
    """Resolve explicit CLI paths against the current directory, without copying data."""
    changed = {}
    for field, value in (("samples", samples), ("features", features), ("output_dir", output)):
        if value is not None:
            changed[field] = Path(value).expanduser().resolve()
    training = {}
    if seed is not None:
        training["seed"] = seed
    if device is not None:
        training["device"] = device
    if training:
        changed["training"] = replace(config.training, **training)
    return replace(config, **changed)


class EpochBatchSampler(Sampler[list[int]]):
    """Use the existing epoch-seeded permutation, including singleton microbatches."""

    def __init__(self, size: int, batch_size: int, seed: int):
        self.size, self.batch_size, self.seed, self.epoch = size, batch_size, seed, 0
        if size < 1 or batch_size < 1:
            raise ValueError("Sampler size and batch size must be positive")

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.size, generator=generator).tolist()
        batches = [order[i:i + self.batch_size] for i in range(0, self.size, self.batch_size)]
        # Match the older sequence runner for batch sizes >= 2.
        if self.batch_size > 1 and len(batches) > 1 and len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        return iter(batches)

    def __len__(self):
        count = math.ceil(self.size / self.batch_size)
        return count - int(self.batch_size > 1 and count > 1 and self.size % self.batch_size == 1)


def inspect_model(config: ReadoutModelConfig | ReadoutRunConfig | None = None) -> dict:
    """Describe dimensions and parameter counts without opening dataset files."""
    model_config = config.model if isinstance(config, ReadoutRunConfig) else config
    model_config = model_config or ReadoutModelConfig()
    # Meta tensors permit safe inspection even for an explicitly requested large flatten head.
    with torch.device("meta"):
        model = ReactionReadoutModel(model_config)
    return {
        "model_config": asdict(model_config),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "parameters_by_module": {name: sum(p.numel() for p in module.parameters())
                                 for name, module in model.named_children()},
        "flatten_dense_parameters": model_config.flatten_dense_parameters(),
        "protein_cache_width": model_config.protein_dim,
        "tokenizer": ReactionTokenizer().metadata(),
        "data_read": False, "model_trained": False,
    }


def validate(config: ReadoutRunConfig, *, split: str | None = None) -> dict:
    dataset = _dataset(config, [split] if split else None)
    if not len(dataset):
        raise ValueError("No samples selected")
    info = inspect_model(config)
    return {
        **info, "data_read": True, "chemical_encoder": config.chemical_encoder,
        "n": len(dataset), "split": split or "all", "coverage": dataset.coverage,
        "source_hashes": _hashes(config), "preparation": _verify_preparation(config),
        "information_regime": "enzyme_known_reaction", "rows_dropped": 0,
    }


def _train_epoch(model, loader, optimizer, training, device):
    """Average gradients over samples, including unequal/final accumulation windows."""
    model.train()
    iterator = iter(loader)
    total, count, steps = 0.0, 0, 0
    while window := list(islice(iterator, training.grad_accum_steps)):
        window_count = sum(batch.batch_size for batch in window)
        optimizer.zero_grad(set_to_none=True)
        for batch in window:
            batch = batch.to(device)
            loss = checked_mse(model(batch), batch.target)
            (loss * (batch.batch_size / window_count)).backward()
            total += float(loss.detach().cpu()) * batch.batch_size
            count += batch.batch_size
        torch.nn.utils.clip_grad_norm_(model.parameters(), training.grad_clip,
                                      error_if_nonfinite=True)
        optimizer.step()
        steps += 1
    return total, count, steps


def fit(config: ReadoutRunConfig, *, overwrite: bool = False) -> dict:
    device = seed_everything(config.training)
    train, val = _dataset(config, ["train"]), _dataset(config, ["val"])
    if len(train) < 2 or len(val) < 1:
        raise ValueError("Need at least two training rows and one validation row")
    _protect_output(config.output_dir, (config.samples, config.features, config.config_path,
                                        config.encoder_checkpoint))
    _prepare_output(config.output_dir, overwrite)
    model = ReactionReadoutModel(config.model)
    initialization = _load_initial_encoder(model, config)
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config.training.lr, weight_decay=config.training.weight_decay)
    sampler = EpochBatchSampler(len(train), config.training.batch_size, config.training.seed)
    train_loader = DataLoader(train, batch_sampler=sampler, collate_fn=collate_sequences,
                              num_workers=config.training.num_workers)
    val_loader = _loader(val, config)
    splits = {split: [row["sample_id"] for row in train.all_records if row["split"] == split]
              for split in ("train", "val", "test")}
    metadata = {
        "kind": CHECKPOINT_KIND, "format_version": CHECKPOINT_VERSION,
        "chemical_encoder": config.chemical_encoder, "model_config": asdict(config.model),
        "tokenizer": ReactionTokenizer().metadata(), "missing_reaction": config.missing_reaction,
        "initialization": initialization, "source_hashes": _hashes(config),
        "train_identity": _record_identity(train.records), "val_identity": _record_identity(val.records),
        "split_ids": splits, "run_id": uuid.uuid4().hex, "git": _git_provenance(),
        "resolved_config": config.snapshot(), "target_definition": "log10(kcat / s^-1)",
        "information_regime": "enzyme_known_reaction", "representation_policy": "reaction_smiles",
        "selection": {"metric": "global_validation_rmse", "test_used": False},
        "coverage": {"train": train.coverage, "val": val.coverage},
        "preparation": _verify_preparation(config),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    _json_write(config.output_dir / "run.json", {**metadata, "status": "running"})
    with (config.output_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.snapshot(), handle, sort_keys=False)
    best, anchor, stale, best_epoch = float("inf"), float("inf"), 0, 0
    history = []
    destination = config.output_dir / "best.pt"
    try:
        for epoch in range(1, config.training.epochs + 1):
            sampler.set_epoch(epoch)
            total, count, steps = _train_epoch(model, train_loader, optimizer, config.training, device)
            if count != len(train):
                raise RuntimeError("Training did not visit every row exactly once")
            validation = collect_predictions(model, val_loader, device)
            score = validation["metrics"]["rmse"]
            entry = {"epoch": epoch, "train_rmse": (total / count)**.5,
                     "training_rows": count, "optimizer_steps": steps,
                     "val": validation["metrics"]}
            history.append(entry)
            with (config.output_dir / "history.jsonl").open("w" if epoch == 1 else "a") as handle:
                handle.write(json.dumps(entry, allow_nan=False) + "\n")
            if score < best:
                best, best_epoch = score, epoch
                checkpoint = {**metadata, "epoch": epoch, "best_val_rmse": score,
                              "training_completed": False,
                              "model_state": {key: value.detach().cpu().clone()
                                              for key, value in model.state_dict().items()}}
                temporary = destination.with_suffix(".pt.tmp")
                torch.save(checkpoint, temporary)
                temporary.replace(destination)
            if score < anchor - config.training.min_delta:
                anchor, stale = score, 0
            else:
                stale += 1
            print(json.dumps({"epoch": epoch, "val_rmse": score, "best_epoch": best_epoch}),
                  flush=True)
            if stale >= config.training.patience:
                break
        checkpoint = load_checkpoint(destination)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        locked = collect_predictions(model, val_loader, device)
        _write_predictions(config.output_dir / "validation_predictions.csv", locked)
        # A checkpoint produced mid-fit is not yet locked: a later validation
        # epoch could replace it. Finalize only after training and validation replay.
        checkpoint["training_completed"] = True
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(destination)
        result = {
            "status": "completed", "chemical_encoder": config.chemical_encoder,
            "model_config": asdict(config.model), "best_epoch": best_epoch,
            "epochs_completed": len(history), "best_validation": locked["metrics"],
            "train_count": len(train), "val_count": len(val), "test_evaluated": False,
            "checkpoint": str(destination), "checkpoint_sha256": sha256_file(destination),
            "parameter_count": metadata["parameter_count"],
            "trainable_parameter_count": metadata["trainable_parameter_count"],
            "initialization": initialization, "coverage": metadata["coverage"],
            "validation_by_input_status": locked["metrics_by_input_status"],
            "training": config.snapshot()["training"], "selection": metadata["selection"],
        }
        _json_write(config.output_dir / "summary.json", result)
        _json_write(config.output_dir / "run.json", {**metadata, "status": "completed"})
        return result
    except BaseException as error:
        _json_write(config.output_dir / "run.json", {**metadata, "status": "failed", "error": str(error)})
        raise


def load_checkpoint(path) -> dict:
    checkpoint = _safe_load(path)
    if (not isinstance(checkpoint, dict) or checkpoint.get("kind") != CHECKPOINT_KIND
            or type(checkpoint.get("format_version")) is not int
            or checkpoint["format_version"] != CHECKPOINT_VERSION):
        raise ValueError("Unsupported readout regressor checkpoint")
    required = {"chemical_encoder", "model_config", "tokenizer", "missing_reaction", "model_state",
                "source_hashes", "split_ids", "train_identity", "val_identity", "run_id",
                "epoch", "best_val_rmse", "selection", "training_completed"}
    if required - checkpoint.keys():
        raise ValueError(f"Checkpoint lacks provenance: {sorted(required - checkpoint.keys())}")
    if checkpoint["chemical_encoder"] != "reaction_smiles" or checkpoint["missing_reaction"] != "error":
        raise ValueError("Checkpoint must preserve complete reaction-SMILES inputs")
    if checkpoint["selection"] != {"metric": "global_validation_rmse", "test_used": False}:
        raise ValueError("Checkpoint does not declare validation-only selection")
    if type(checkpoint["training_completed"]) is not bool:
        raise ValueError("Checkpoint training_completed must be a boolean")
    if type(checkpoint["epoch"]) is not int or checkpoint["epoch"] < 1:
        raise ValueError("Checkpoint epoch must be a positive integer")
    best = checkpoint["best_val_rmse"]
    if isinstance(best, bool) or not isinstance(best, (int, float)) or not math.isfinite(best) or best < 0:
        raise ValueError("Checkpoint validation score must be finite and nonnegative")
    try:
        config = ReadoutModelConfig(**_mapping(checkpoint["model_config"], "model_config"))
    except TypeError as error:
        raise ValueError("Invalid checkpoint model_config") from error
    if checkpoint["tokenizer"] != ReactionTokenizer().metadata():
        raise ValueError("Checkpoint tokenizer metadata differs from this implementation")
    state = _mapping(checkpoint["model_state"], "model_state")
    if not state or any(not isinstance(value, torch.Tensor) or value.is_complex()
                        or not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("Checkpoint weights must be finite real tensors")
    with torch.device("meta"):
        expected = ReactionReadoutModel(config).state_dict()
    if set(state) != set(expected) or any(
            state[key].shape != expected[key].shape or state[key].dtype != expected[key].dtype
            for key in expected if key in state):
        raise ValueError("Checkpoint tensor keys, shapes or dtypes disagree with the stored architecture")

    def string_set(value):
        if (not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value)
                or len(set(value)) != len(value)):
            raise ValueError("Checkpoint identity lists must contain unique nonempty strings")
        return set(value)

    splits = _mapping(checkpoint["split_ids"], "split_ids")
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("Checkpoint splits must contain train, val, test")
    seen = set()
    for values in splits.values():
        current = string_set(values)
        if seen & current:
            raise ValueError("Checkpoint split sample IDs overlap")
        seen |= current
    for split in ("train", "val"):
        identity = _mapping(checkpoint[f"{split}_identity"], "identity")
        for field in ("sample_ids", "sequence_hashes", "group_ids"):
            string_set(identity.get(field))
        if set(identity["sample_ids"]) != set(splits[split]) or not identity["sequence_hashes"]:
            raise ValueError("Checkpoint training identities disagree with splits")
    for field in ("sequence_hashes", "group_ids"):
        if set(checkpoint["train_identity"][field]) & set(checkpoint["val_identity"][field]):
            raise ValueError("Checkpoint train/validation identities overlap")
    hashes = _mapping(checkpoint["source_hashes"], "source_hashes")
    for key in ("samples_sha256", "features_sha256"):
        value = hashes.get(key)
        if (not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)):
            raise ValueError("Invalid checkpoint source hash")
    return checkpoint


def infer(config: ReadoutRunConfig, checkpoint_path: str | Path, *, output_dir: str | Path,
          split: str | None = "test", evaluate: bool = True, overwrite: bool = False) -> dict:
    if evaluate and split not in {"train", "val", "test"}:
        raise ValueError("Evaluation requires one explicit split")
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint["training_completed"] is not True:
        raise ValueError("Checkpoint is not locked: wait for training to complete")
    if (asdict(config.model) != checkpoint["model_config"]
            or config.chemical_encoder != checkpoint["chemical_encoder"]
            or config.missing_reaction != checkpoint["missing_reaction"]):
        raise ValueError("Evaluation config disagrees with checkpoint architecture or input policy")
    device = seed_everything(config.training)
    dataset = _dataset(config, [split] if split else None, require_targets=evaluate)
    if evaluate and split == "test":
        _check_test_overlap(dataset, checkpoint)
    if evaluate:
        selected_ids = [row["sample_id"] for row in dataset.records]
        if selected_ids != checkpoint["split_ids"][split]:
            raise ValueError(
                "Benchmark evaluation requires exactly the locked split sample IDs and order; "
                "rows must not be filtered. Use predict for a new cohort instead."
            )
        if _hashes(config) != checkpoint["source_hashes"]:
            raise ValueError(
                "Benchmark evaluation input hashes differ from the locked training data; "
                "labels, features and prepared rows must remain byte-identical. Paths may be "
                "relocated. Use predict for new inputs instead of reporting a changed benchmark."
            )
    model = ReactionReadoutModel(config.model)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    output = Path(output_dir).expanduser().resolve()
    _protect_output(output, (config.samples, config.features, Path(checkpoint_path), config.config_path))
    _prepare_output(output, overwrite)
    collected = collect_predictions(model, _loader(dataset, config), device, need_targets=evaluate)
    _write_predictions(output / "predictions.csv", collected)
    result = {
        "chemical_encoder": config.chemical_encoder, "model_config": asdict(config.model),
        "split": split or "all", "n": len(dataset), "coverage": dataset.coverage,
        "checkpoint_sha256": sha256_file(checkpoint_path), "source_hashes": _hashes(config),
        "training_source_hashes": checkpoint["source_hashes"], "weights_changed": False,
        "preparation": _verify_preparation(config), "information_regime": "enzyme_known_reaction",
        "metrics_computed": evaluate, "checkpoint_run_id": checkpoint["run_id"],
        "note": "Architecture, tokenizer and input policy must match the locked checkpoint.",
    }
    if evaluate:
        result["metrics"] = collected["metrics"]
        result["metrics_by_input_status"] = collected["metrics_by_input_status"]
    _json_write(output / ("metrics.json" if evaluate else "prediction_metadata.json"), result)
    return result
