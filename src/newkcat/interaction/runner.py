"""Train the interaction model on frozen, complete position-wise feature caches.

The original manifest determines every row and split. Training reads targets
from train/validation only; test evaluation is a separate, locked-checkpoint
command. No graph or atom-mapping success criterion removes samples.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import json
import math
from pathlib import Path
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from ..data import read_manifest
from ..metrics import regression_metrics
from ..train import (
    TrainingConfig, _UniqueConfigLoader, _git_provenance, _json_write, _mapping,
    _prepare_output, _record_identity, _write_predictions, checked_mse,
    seed_everything, sha256_file,
)
from .data import InteractionDataset, collate_interactions
from .model import InteractionModelConfig, ReactionInteractionModel


CHECKPOINT_KIND = "newkcat_pretrained_reaction_interaction"
CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class InteractionTrainingConfig(TrainingConfig):
    batch_size: int = 1
    grad_accum_steps: int = 8
    epochs: int = 100
    patience: int = 15
    lr: float = 3e-4
    grad_clip: float = 1.0
    standardize_target: bool = True

    def __post_init__(self):
        for name in ("batch_size", "grad_accum_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"training.{name} must be a positive integer")
        common = {f.name: getattr(self, f.name) for f in fields(TrainingConfig)}
        common["batch_size"] = max(2, self.batch_size)
        TrainingConfig(**common)
        if self.allow_reference_centers:
            raise ValueError("This model does not require reaction-center annotations")
        if type(self.standardize_target) is not bool:
            raise ValueError("standardize_target must be a YAML boolean")


@dataclass(frozen=True)
class InteractionRunConfig:
    config_path: Path
    samples: Path
    cache: Path
    output_dir: Path
    model: InteractionModelConfig
    training: InteractionTrainingConfig
    global_features: Path | None = None

    def snapshot(self):
        return {
            "data": {"samples": str(self.samples), "cache": str(self.cache),
                     "global_features": str(self.global_features) if self.global_features else None},
            "model": asdict(self.model),
            "training": {k: v for k, v in asdict(self.training).items()
                         if k != "allow_reference_centers"},
            "output_dir": str(self.output_dir),
        }


def load_config(path: str | Path) -> InteractionRunConfig:
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        raw = _mapping(yaml.load(handle, Loader=_UniqueConfigLoader), "interaction config")
    if set(raw) - {"data", "model", "training", "output_dir"}:
        raise ValueError("Unknown interaction configuration keys")

    def resolve(value, label):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a nonempty path")
        value = Path(value).expanduser()
        return (path.parent / value).resolve() if not value.is_absolute() else value.resolve()

    data = _mapping(raw.get("data"), "data")
    if set(data) - {"samples", "cache", "global_features"}:
        raise ValueError("data supports samples, cache, and optional global_features")
    model = _mapping(raw.get("model", {}), "model")
    if set(model) - {f.name for f in fields(InteractionModelConfig)}:
        raise ValueError("Unknown interaction model options")
    train = _mapping(raw.get("training", {}), "training")
    if set(train) - ({f.name for f in fields(InteractionTrainingConfig)}
                     - {"allow_reference_centers"}):
        raise ValueError("Unknown interaction training options")
    return InteractionRunConfig(
        config_path=path, samples=resolve(data.get("samples"), "data.samples"),
        cache=resolve(data.get("cache"), "data.cache"),
        global_features=(resolve(data["global_features"], "data.global_features")
                         if data.get("global_features") else None),
        model=InteractionModelConfig(**model), training=InteractionTrainingConfig(**train),
        output_dir=resolve(raw.get("output_dir", "runs/interaction"), "output_dir"),
    )


def with_overrides(config, *, samples=None, cache=None, global_features=None,
                   output=None, seed=None, device=None):
    changes = {}
    for key, value in (("samples", samples), ("cache", cache),
                       ("global_features", global_features), ("output_dir", output)):
        if value is not None:
            changes[key] = Path(value).expanduser().resolve()
    train = {k: v for k, v in (("seed", seed), ("device", device)) if v is not None}
    if train:
        changes["training"] = replace(config.training, **train)
    return replace(config, **changes)


def _cache_index(path: Path) -> Path:
    return path / "index.json" if path.is_dir() else path


def _hashes(config):
    return {
        "samples_sha256": sha256_file(config.samples),
        "cache_index_sha256": sha256_file(_cache_index(config.cache)),
        "global_features_sha256": (sha256_file(config.global_features)
                                   if config.global_features else None),
    }


def _dataset(config, splits, *, require_targets=True):
    return InteractionDataset(
        samples_path=config.samples, cache_path=config.cache, splits=splits,
        global_features=config.global_features, require_targets=require_targets,
    )


def _check_dimensions(dataset, config):
    if not len(dataset):
        raise ValueError("Selected split is empty")
    item = dataset[0]
    if item["protein_states"].shape[-1] != config.protein_dim:
        raise ValueError("Protein residue cache dimension differs from model.protein_dim")
    if item["reaction_states"].shape[-1] != config.reaction_dim:
        raise ValueError("Reaction token cache dimension differs from model.reaction_dim")
    if item["protein_global"].shape != (config.protein_dim,):
        raise ValueError("Protein global cache dimension differs from model.protein_dim")


def _to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def _loader(dataset, config, *, epoch=None):
    generator = torch.Generator().manual_seed(config.seed + (epoch or 0))
    return DataLoader(
        dataset, batch_size=config.batch_size, shuffle=epoch is not None,
        generator=generator, drop_last=False, collate_fn=collate_interactions,
        num_workers=config.num_workers,
    )


def _protect_output(config, output):
    output = output.expanduser().resolve()
    sources = [config.samples, _cache_index(config.cache), config.config_path]
    if config.global_features:
        sources.append(config.global_features)
    if any(source == output or source.is_relative_to(output) for source in sources):
        raise ValueError("Output directory would contain source inputs; choose a new run directory")
    _prepare_output(output, overwrite=False)
    return output


def inspect_model(config=None):
    config = config.model if isinstance(config, InteractionRunConfig) else config
    config = config or InteractionModelConfig()
    with torch.device("meta"):
        model = ReactionInteractionModel(config)
    return {
        "model_config": asdict(config),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "parameters_by_module": {name: sum(p.numel() for p in module.parameters())
                                 for name, module in model.named_children()},
        "pretrained_encoder_parameters_in_optimizer": 0,
        "regressor_input_dim": config.protein_dim + 5 * config.d_model,
        "requires_residue_cache": True, "requires_reaction_token_cache": True,
        "requires_atom_mapping": False, "trained_on_kinetics": False,
    }


def collect_predictions(model, loader, device, *, mean=0.0, scale=1.0, need_targets=True):
    model.eval()
    result = {"sample_ids": [], "prediction": []}
    if need_targets:
        result["target"] = []
    with torch.inference_mode():
        for batch in loader:
            batch = _to_device(batch, device)
            prediction = model(batch) * scale + mean
            if prediction.shape != (len(batch["sample_ids"]),):
                raise ValueError("Predictions must be a vector with one value per sample")
            if not torch.isfinite(prediction).all():
                raise ValueError("Nonfinite predictions")
            result["sample_ids"].extend(batch["sample_ids"])
            result["prediction"].extend(prediction.double().cpu().tolist())
            if need_targets:
                targets = batch["targets"]
                if targets is None:
                    raise ValueError("Evaluation requires all selected targets")
                checked_mse(prediction, targets)
                result["target"].extend(targets.double().cpu().tolist())
    if not result["sample_ids"]:
        raise ValueError("No predictions produced")
    if need_targets:
        result["metrics"] = regression_metrics(result["target"], result["prediction"])
    return result


def _save_checkpoint(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def load_checkpoint(path, *, allow_unfinished=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("kind") != CHECKPOINT_KIND:
        raise ValueError("Not an interaction-model checkpoint")
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint version")
    if not allow_unfinished and checkpoint.get("training_completed") is not True:
        raise ValueError("Training is incomplete; this checkpoint is not locked for evaluation")
    transform = checkpoint.get("target_transform", {})
    for key in ("mean", "scale"):
        if not isinstance(transform.get(key), (int, float)) or not math.isfinite(transform[key]):
            raise ValueError("Invalid checkpoint target transform")
    if transform["scale"] <= 0:
        raise ValueError("Target transform scale must be positive")
    InteractionModelConfig(**checkpoint["model_config"])
    return checkpoint


def fit(config: InteractionRunConfig):
    device = seed_everything(config.training)
    train = _dataset(config, ["train"])
    val = _dataset(config, ["val"])
    _check_dimensions(train, config.model)
    _check_dimensions(val, config.model)
    if len(train) < 2:
        raise ValueError("At least two training rows are required")
    # Targets are parsed exclusively from train records, never a full-target array.
    targets = np.asarray([float(row["log10_kcat"]) for row in train.records], dtype=np.float64)
    mean = float(targets.mean()) if config.training.standardize_target else 0.0
    scale = float(targets.std()) if config.training.standardize_target else 1.0
    scale = scale if scale > 1e-8 else 1.0
    model = ReactionInteractionModel(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay,
    )
    output = _protect_output(config, config.output_dir)
    all_records = read_manifest(config.samples)
    split_ids = {split: [r["sample_id"] for r in all_records if r["split"] == split]
                 for split in ("train", "val", "test")}
    with _cache_index(config.cache).open(encoding="utf-8") as handle:
        cache_metadata = json.load(handle)
    provenance = {
        "kind": CHECKPOINT_KIND, "version": CHECKPOINT_VERSION, "run_id": uuid.uuid4().hex,
        "model_config": asdict(config.model), "resolved_config": config.snapshot(),
        "source_hashes": _hashes(config), "split_ids": split_ids,
        "train_identity": _record_identity(train.records),
        "val_identity": _record_identity(val.records),
        "encoders": cache_metadata.get("encoders", {}),
        "target_transform": {"mean": mean, "scale": scale, "fitted_on": "train"},
        "target_definition": "log10(kcat / s^-1)", "git": _git_provenance(),
        "protein_global_source": ("existing_sample_id_joined_cache" if config.global_features
                                  else "mean_of_all_cached_residues"),
        "pretrained_encoders_frozen": True, "training_completed": False,
    }
    _json_write(output / "run.json", {**provenance, "status": "running"})
    with (output / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.snapshot(), handle, sort_keys=False)
    val_loader = _loader(val, config.training)
    best, best_epoch, stale, anchor = float("inf"), 0, 0, float("inf")
    best_path = output / "best.pt"
    history = []
    try:
        for epoch in range(1, config.training.epochs + 1):
            model.train()
            loader = _loader(train, config.training, epoch=epoch)
            count, squared_error, updates = 0, 0.0, 0
            effective_batch = config.training.batch_size * config.training.grad_accum_steps
            optimizer.zero_grad(set_to_none=True)
            window_count = min(effective_batch, len(train))
            for step, batch in enumerate(loader, 1):
                batch = _to_device(batch, device)
                y = batch["targets"]
                if y is None:
                    raise ValueError("Training requires targets")
                pred = model(batch)
                loss = checked_mse(pred, (y - mean) / scale)
                n = len(batch["sample_ids"])
                (loss * n / window_count).backward()
                squared_error += float(loss.detach()) * n * scale**2
                count += n
                if step % config.training.grad_accum_steps == 0 or step == len(loader):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.training.grad_clip, error_if_nonfinite=True,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                    window_count = min(effective_batch, len(train) - count)
            if count != len(train):
                raise RuntimeError("Training did not visit every selected sample")
            validation = collect_predictions(model, val_loader, device, mean=mean, scale=scale)
            rmse = validation["metrics"]["rmse"]
            row = {"epoch": epoch, "train_rmse": math.sqrt(squared_error / count),
                   "val": validation["metrics"], "optimizer_updates": updates}
            history.append(row)
            with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            if rmse < best:
                best, best_epoch = rmse, epoch
                _save_checkpoint(best_path, {
                    **provenance, "epoch": epoch, "best_val_rmse": best,
                    "model_state": {k: v.detach().cpu().clone()
                                    for k, v in model.state_dict().items()},
                })
            if rmse < anchor - config.training.min_delta:
                anchor, stale = rmse, 0
            else:
                stale += 1
            print(json.dumps({"epoch": epoch, "train_rmse": row["train_rmse"],
                              "val_rmse": rmse, "best_epoch": best_epoch}), flush=True)
            if stale >= config.training.patience:
                break
        checkpoint = load_checkpoint(best_path, allow_unfinished=True)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        locked_val = collect_predictions(model, val_loader, device, mean=mean, scale=scale)
        _write_predictions(output / "validation_predictions.csv", locked_val)
        checkpoint["training_completed"] = True
        # Publish the completed checkpoint only after every final report succeeds.
        # If writing a report fails, the existing best.pt stays explicitly unfinished.
        completed_path = output / "best.completed.pt.tmp"
        torch.save(checkpoint, completed_path)
        result = {
            "status": "completed", "best_epoch": best_epoch, "epochs_completed": len(history),
            "best_validation": locked_val["metrics"], "test_evaluated": False,
            "train_count": len(train), "val_count": len(val), "rows_dropped": 0,
            "checkpoint": str(best_path), "checkpoint_sha256": sha256_file(completed_path),
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
        }
        _json_write(output / "summary.json", result)
        _json_write(output / "run.json", {**provenance, "training_completed": True,
                                          "status": "completed"})
        completed_path.replace(best_path)
        return result
    except BaseException as error:
        _json_write(output / "run.json", {**provenance, "status": "failed",
                                          "error": str(error), "error_type": type(error).__name__})
        raise


def infer(config, checkpoint_path, *, split="test", output=None, evaluate=True):
    checkpoint = load_checkpoint(checkpoint_path)
    if evaluate and checkpoint["source_hashes"] != _hashes(config):
        raise ValueError("Evaluation inputs differ from the checkpoint's original rows/cache")
    if evaluate and split not in {"val", "test"}:
        raise ValueError("Evaluation split must be val or test")
    global_source = ("existing_sample_id_joined_cache" if config.global_features
                     else "mean_of_all_cached_residues")
    if global_source != checkpoint.get("protein_global_source"):
        raise ValueError("Prediction must retain the trained protein global feature source")
    dataset = _dataset(config, [split] if split else None, require_targets=evaluate)
    model_config = InteractionModelConfig(**checkpoint["model_config"])
    _check_dimensions(dataset, model_config)
    if evaluate:
        expected = checkpoint["split_ids"][split]
        actual = [row["sample_id"] for row in dataset.records]
        if actual != expected:
            raise ValueError("Evaluation must preserve the original selected sample IDs and order")
    with _cache_index(config.cache).open(encoding="utf-8") as handle:
        encoders = json.load(handle).get("encoders", {})
    if encoders != checkpoint["encoders"]:
        raise ValueError("Use the same pretrained encoder settings as the training cache")
    device = seed_everything(config.training)
    model = ReactionInteractionModel(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    transform = checkpoint["target_transform"]
    collected = collect_predictions(
        model, _loader(dataset, config.training), device,
        mean=transform["mean"], scale=transform["scale"], need_targets=evaluate,
    )
    output = _protect_output(config, Path(output) if output else config.output_dir)
    _write_predictions(output / "predictions.csv", collected)
    result = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "n": len(dataset), "split": split, "rows_dropped": 0,
        "prediction_file": str(output / "predictions.csv"), "model_config": asdict(model_config),
        "metrics": collected.get("metrics"), "used_targets": evaluate,
    }
    _json_write(output / "summary.json", result)
    return result
