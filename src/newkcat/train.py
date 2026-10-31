"""Leakage-aware training, checkpointed inference, and center intervention audits.

Training does not instantiate a test dataset. Full-file hashes and split metadata
are recorded for provenance, but test targets/features never enter fitting,
checkpoint selection, or early stopping.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
import random
import subprocess
import uuid
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, Sampler

from . import __version__
from .data import KcatDataset, collate_graphs, ec2_token
from .metrics import regression_metrics
from .model import ModelConfig, NewKcat
from .types import GraphBatch


CHECKPOINT_VERSION = 1
CENTER_MODES = {"center", "shell", "conditioned"}
MODES = {"baseline", "global", "center", "shell", "conditioned", "learned"}
COVERAGE_FIELDS = ("n", "missing", "predicted", "reference", "active_nonzero",
                   "observed_all_zero", "ec_informed")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 668
    batch_size: int = 32
    epochs: int = 150
    patience: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    min_delta: float = 0.0
    grad_clip: float = 5.0
    num_workers: int = 0
    num_threads: int = 1
    device: str = "cpu"
    deterministic: bool = False
    allow_reference_centers: bool = False

    def __post_init__(self):
        positive_int = ("batch_size", "epochs", "patience", "num_threads")
        for name in positive_int:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"training.{name} must be a positive integer")
        if self.batch_size < 2:
            raise ValueError("training.batch_size must be >= 2 for CataPro BatchNorm")
        if type(self.num_workers) is not int or self.num_workers < 0:
            raise ValueError("training.num_workers must be a nonnegative integer")
        if type(self.seed) is not int or self.seed < 0 or self.seed >= 2**32:
            raise ValueError("training.seed must be an integer in [0, 2**32)")
        for name in ("lr", "weight_decay", "min_delta", "grad_clip"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError(f"training.{name} must be numeric")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"training.{name} must be finite and nonnegative")
        if self.lr == 0 or self.grad_clip == 0:
            raise ValueError("training.lr and training.grad_clip must be positive")
        for name in ("allow_reference_centers", "deterministic"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"training.{name} must be a YAML boolean")
        if not isinstance(self.device, str):
            raise ValueError("training.device must be cpu or cuda[:index]")


@dataclass(frozen=True)
class RunConfig:
    config_path: Path
    samples: Path
    features: Path
    centers: Path | None
    output_dir: Path
    model: dict[str, Any]
    training: TrainingConfig

    def snapshot(self) -> dict:
        return {
            "data": {
                "samples": str(self.samples),
                "features": str(self.features),
                "centers": str(self.centers) if self.centers is not None else None,
            },
            "model": self.model,
            "training": asdict(self.training),
            "output_dir": str(self.output_dir),
        }


class _UniqueConfigLoader(yaml.SafeLoader):
    """Safe YAML with explicit failure for accidentally overwritten settings."""


def _unique_yaml_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError("YAML configuration keys must be strings")
        if key in mapping:
            raise ValueError(f"Duplicate YAML configuration key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueConfigLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_yaml_mapping
)


def _mapping(value, label: str) -> dict:
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError(f"{label} must be a mapping with string keys")
    return value


def load_config(path: str | Path, *, mode: str | None = None) -> RunConfig:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        try:
            raw = _mapping(yaml.load(handle, Loader=_UniqueConfigLoader), "config")
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid YAML configuration: {error}") from error
    unknown = set(raw) - {"data", "model", "training", "output_dir"}
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    data = _mapping(raw.get("data"), "data")
    if set(data) - {"samples", "features", "centers"}:
        raise ValueError("data supports only samples, features, and centers")

    def resolve(value, label: str) -> Path:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must be a nonempty file path")
        result = Path(value).expanduser()
        return (path.parent / result).resolve() if not result.is_absolute() else result.resolve()

    model = dict(_mapping(raw.get("model", {}), "model"))
    allowed_model = {f.name for f in fields(ModelConfig)} - {
        "atom_dim", "bond_dim", "ec_vocab_size"
    }
    if set(model) - allowed_model:
        raise ValueError(f"Unknown or data-derived model fields: {sorted(set(model)-allowed_model)}")
    if mode is not None:
        if mode not in MODES:
            raise ValueError(f"Unknown model mode: {mode}")
        model["mode"] = mode
    options = dict(_mapping(raw.get("training", {}), "training"))
    unknown_training = set(options) - {f.name for f in fields(TrainingConfig)}
    if unknown_training:
        raise ValueError(f"Unknown training fields: {sorted(unknown_training)}")
    training = TrainingConfig(**options)
    return RunConfig(
        config_path=path,
        samples=resolve(data.get("samples"), "data.samples"),
        features=resolve(data.get("features"), "data.features"),
        centers=resolve(data["centers"], "data.centers") if data.get("centers") is not None else None,
        output_dir=resolve(raw.get("output_dir", "runs/default"), "output_dir"),
        model=model,
        training=training,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_hashes(config: RunConfig) -> dict:
    return {
        "samples_sha256": sha256_file(config.samples),
        "features_sha256": sha256_file(config.features),
        "centers_sha256": sha256_file(config.centers) if config.centers else None,
    }


def _json_write(path: Path, content: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temp.replace(path)


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"Output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {path}; use --overwrite explicitly")
    path.mkdir(parents=True, exist_ok=True)


def _git_provenance() -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True,
            text=True, check=True,
        ).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def seed_everything(config: TrainingConfig) -> torch.device:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.num_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(config.deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
    try:
        device = torch.device(config.device)
    except RuntimeError as error:
        raise ValueError("training.device must be cpu or cuda[:index]") from error
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only explicit CPU or CUDA devices are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; choose training.device: cpu")
    if device.type == "cuda" and device.index is not None:
        if device.index >= torch.cuda.device_count():
            raise ValueError(f"Requested CUDA device {device.index} does not exist")
    return device


class NonSingletonBatchSampler(Sampler[list[int]]):
    """Visit every record once; merge a final singleton for BatchNorm safety."""

    def __init__(self, size: int, batch_size: int, seed: int):
        if size < 2 or batch_size < 2:
            raise ValueError("Training requires at least two records and batch_size >= 2")
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.size, generator=generator).tolist()
        batches = [order[i:i + self.batch_size] for i in range(0, self.size, self.batch_size)]
        if len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        return iter(batches)

    def __len__(self) -> int:
        count = math.ceil(self.size / self.batch_size)
        return count - int(self.size % self.batch_size == 1)


def checked_mse(prediction: Tensor, target: Tensor) -> Tensor:
    if prediction.ndim != 1 or target.ndim != 1 or prediction.shape != target.shape:
        raise ValueError(
            f"Refusing broadcasted loss: prediction={tuple(prediction.shape)}, "
            f"target={tuple(target.shape)}; both must be [B]"
        )
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Nonfinite predictions or targets in loss")
    loss = torch.mean((prediction - target) ** 2)
    if not torch.isfinite(loss):
        raise ValueError("Squared-error loss overflowed; check targets and features")
    return loss


def _dataset(config: RunConfig, splits: list[str] | None, ec_vocab=None,
             require_targets: bool = True) -> KcatDataset:
    return KcatDataset(
        samples_path=config.samples, features_path=config.features, centers_path=config.centers,
        splits=splits, ec_vocab=ec_vocab, require_targets=require_targets,
    )


def _make_model_config(config: RunConfig, dataset: KcatDataset) -> ModelConfig:
    return ModelConfig(
        **config.model, atom_dim=dataset.atom_dim, bond_dim=dataset.bond_dim,
        ec_vocab_size=max(dataset.ec_vocab.values(), default=0) + 1,
    )


def _center_coverage(dataset: KcatDataset) -> dict[str, dict[str, int]]:
    """Summarize annotations only for selected rows, irrespective of model mode."""
    by_split = {}
    for record, item in zip(dataset.records, dataset.items, strict=True):
        counts = by_split.setdefault(record["split"], dict.fromkeys(COVERAGE_FIELDS, 0))
        counts["n"] += 1
        counts[item["center_kind"]] += 1
        if item["center_available"]:
            active = bool(np.any(item["center_scores"] > 0))
            counts["active_nonzero"] += int(active)
            counts["observed_all_zero"] += int(not active)
        counts["ec_informed"] += int(item["center_uses_ec"])
    return by_split


def _information_regime(model_config: ModelConfig, datasets, allow_reference: bool) -> str:
    if model_config.mode not in CENTER_MODES:
        return "sequence_substrate_ec" if model_config.use_ec2 else "sequence_substrate"
    if any(ds.center_uses_ec for ds in datasets) and not model_config.use_ec2:
        raise ValueError(
            "EC-informed centers require model.use_ec2: true. Compare against an "
            "EC-enabled baseline to separate EC information from architectural benefit."
        )
    reference = any(ds.has_reference_centers for ds in datasets)
    if reference and not allow_reference:
        raise ValueError(
            "Reference centers reveal reaction information. Set "
            "training.allow_reference_centers: true (or --allow-reference-centers for inference) "
            "only for a declared reference-center/oracle diagnostic."
        )
    if reference:
        return "reference_center_diagnostic"
    return "predicted_centers_with_ec" if model_config.use_ec2 else "predicted_centers"


def _sequence_hash(record: dict) -> str:
    sequence = "".join(str(record["sequence"]).split()).upper()
    return hashlib.sha256(sequence.encode()).hexdigest()


def _record_identity(records: list[dict]) -> dict:
    return {
        "sample_ids": sorted(str(row["sample_id"]) for row in records),
        "sequence_hashes": sorted({_sequence_hash(row) for row in records}),
        "group_ids": sorted({str(row.get("group_id", "")).strip() for row in records}
                            - {"", "None"}),
    }


def _split_ids(path: Path) -> dict[str, list[str]]:
    """Read split metadata only; never parse or summarize held-out targets."""
    result: dict[str, list[str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            result.setdefault(row["split"].strip(), []).append(row["sample_id"].strip())
    return {key: sorted(values) for key, values in result.items()}


def _loader(dataset, config: TrainingConfig) -> DataLoader:
    return DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers,
        collate_fn=collate_graphs,
    )


def perturb_centers(batch: GraphBatch, intervention: str, seed: int = 0) -> GraphBatch:
    if intervention == "original":
        return batch
    if intervention == "zero":
        return replace(batch, center_scores=torch.zeros_like(batch.center_scores),
                       center_available=torch.zeros_like(batch.center_available))
    if intervention != "permuted":
        raise ValueError(f"Unknown center intervention: {intervention}")
    scores = batch.center_scores.clone()
    if len(batch.sample_ids) != batch.batch_size:
        raise ValueError("Center permutation audit requires stable sample IDs")
    for index, sample_id in enumerate(batch.sample_ids):
        positions = torch.nonzero(batch.atom_mask[index], as_tuple=False).flatten()
        digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
        sample_seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
        generator = torch.Generator().manual_seed(sample_seed)
        permutation = torch.randperm(positions.numel(), generator=generator).to(scores.device)
        scores[index, positions] = scores[index, positions[permutation]]
    return replace(batch, center_scores=scores)


def collect_predictions(model, loader, device, *, need_targets: bool,
                        intervention: str = "original", seed: int = 0) -> dict:
    model.eval()
    predictions, targets, sample_ids = [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch = perturb_centers(batch.to(device), intervention, seed)
            prediction = model(batch)
            if prediction.ndim != 1 or prediction.shape[0] != batch.batch_size:
                raise ValueError("Model predictions must have exactly shape [B]")
            if not torch.isfinite(prediction).all():
                raise ValueError("Model produced nonfinite predictions")
            predictions.extend(prediction.detach().cpu().double().tolist())
            sample_ids.extend(batch.sample_ids)
            if need_targets:
                if batch.target is None:
                    raise ValueError("Evaluation requires targets for every record")
                checked_mse(prediction, batch.target)
                targets.extend(batch.target.detach().cpu().double().tolist())
    if not predictions:
        raise ValueError("No selected samples to predict")
    result = {"sample_ids": sample_ids, "prediction": predictions}
    if need_targets:
        result["target"] = targets
        result["metrics"] = regression_metrics(targets, predictions)
    return result


def _write_predictions(path: Path, collected: dict) -> None:
    columns = ["sample_id", "prediction_log10_kcat"]
    if "target" in collected:
        columns.append("observed_log10_kcat")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for i, sample_id in enumerate(collected["sample_ids"]):
            row = {"sample_id": sample_id, "prediction_log10_kcat": collected["prediction"][i]}
            if "target" in collected:
                row["observed_log10_kcat"] = collected["target"][i]
            writer.writerow(row)


def validate(config: RunConfig) -> dict:
    split_ids = _split_ids(config.samples)
    # A training-only vocabulary, even when validating other splits.
    train = _dataset(config, ["train"])
    model_config = _make_model_config(config, train)
    datasets = {"train": train}
    for split in ("val", "test"):
        if split_ids.get(split):
            datasets[split] = _dataset(config, [split], ec_vocab=train.ec_vocab)
    regime = _information_regime(
        model_config, list(datasets.values()), config.training.allow_reference_centers
    )
    return {
        "schema_version": CHECKPOINT_VERSION,
        "information_regime": regime,
        "counts": {split: len(dataset) for split, dataset in datasets.items()},
        "center_coverage": {split: _center_coverage(dataset).get(split, dict.fromkeys(COVERAGE_FIELDS, 0))
                            for split, dataset in datasets.items()},
        "feature_dimensions": {"protein": 1024, "molt5": 768, "maccs": 167,
                               "atom": train.atom_dim, "bond": train.bond_dim},
        "ec_vocab_size": model_config.ec_vocab_size,
        "model_config": asdict(model_config),
        "source_hashes": input_hashes(config),
        "note": "Validation checks data; it does not establish predictive accuracy.",
    }


def fit(config: RunConfig, *, overwrite: bool = False,
        init_catapro: str | Path | None = None) -> dict:
    device = seed_everything(config.training)
    train = _dataset(config, ["train"])
    val = _dataset(config, ["val"], ec_vocab=train.ec_vocab)
    if len(train) < 2 or len(val) < 1:
        raise ValueError("Training requires >=2 training records and >=1 validation record")
    model_config = _make_model_config(config, train)
    regime = _information_regime(
        model_config, [train, val], config.training.allow_reference_centers
    )
    model = NewKcat(model_config)
    initialization = None
    if init_catapro is not None:
        init_path = Path(init_catapro).resolve()
        initial_state = torch.load(init_path, map_location="cpu", weights_only=True)
        if isinstance(initial_state, dict) and "state_dict" in initial_state:
            initial_state = initial_state["state_dict"]
        model.load_catapro_weights(initial_state)
        initialization = {"path": str(init_path), "sha256": sha256_file(init_path)}
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
    )
    sampler = NonSingletonBatchSampler(len(train), config.training.batch_size, config.training.seed)
    train_loader = DataLoader(
        train, batch_sampler=sampler, collate_fn=collate_graphs,
        num_workers=config.training.num_workers,
    )
    val_loader = _loader(val, config.training)
    _prepare_output(config.output_dir, overwrite)
    run_id = uuid.uuid4().hex
    provenance = {
        "format_version": CHECKPOINT_VERSION,
        "package_version": __version__,
        "run_id": run_id,
        "model_config": asdict(model_config),
        "ec_vocab": dict(train.ec_vocab),
        "resolved_config": config.snapshot(),
        "source_config": str(config.config_path),
        "source_hashes": input_hashes(config),
        "split_ids": _split_ids(config.samples),
        "train_identity": _record_identity(train.records),
        "val_identity": _record_identity(val.records),
        "information_regime": regime,
        "center_coverage": {**_center_coverage(train), **_center_coverage(val)},
        "git": _git_provenance(),
        "initialization": initialization,
        "target_definition": "log10(kcat / s^-1)",
    }
    _json_write(config.output_dir / "run.json", {**provenance, "status": "running"})
    with (config.output_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.snapshot(), handle, sort_keys=False)
    best_rmse, patience_anchor, best_epoch, stale = float("inf"), float("inf"), 0, 0
    history = []
    checkpoint_path = config.output_dir / "best.pt"
    try:
        for epoch in range(1, config.training.epochs + 1):
            sampler.set_epoch(epoch)
            model.train()
            total_squared_error, count = 0.0, 0
            for batch in train_loader:
                batch = batch.to(device)
                if batch.target is None:
                    raise ValueError("Training batch is missing a target")
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch)
                loss = checked_mse(prediction, batch.target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.grad_clip, error_if_nonfinite=True
                )
                optimizer.step()
                total_squared_error += float(loss.detach().cpu()) * batch.batch_size
                count += batch.batch_size
            if count != len(train):
                raise RuntimeError("Training sampler failed to visit every record exactly once")
            validation = collect_predictions(model, val_loader, device, need_targets=True)
            val_rmse = validation["metrics"]["rmse"]
            row = {"epoch": epoch, "train_rmse": math.sqrt(total_squared_error / count),
                   "val": validation["metrics"]}
            history.append(row)
            with (config.output_dir / "history.jsonl").open(
                "w" if epoch == 1 else "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            # Save every genuine improvement; min_delta only governs patience.
            if val_rmse < best_rmse:
                best_rmse, best_epoch = val_rmse, epoch
                checkpoint = {
                    **provenance,
                    "epoch": epoch,
                    "best_val_rmse": best_rmse,
                    "model_state": {key: value.detach().cpu().clone()
                                    for key, value in model.state_dict().items()},
                }
                temp = checkpoint_path.with_suffix(".pt.tmp")
                torch.save(checkpoint, temp)
                temp.replace(checkpoint_path)
            if val_rmse < patience_anchor - config.training.min_delta:
                patience_anchor, stale = val_rmse, 0
            else:
                stale += 1
            print(json.dumps({"epoch": epoch, "train_rmse": row["train_rmse"],
                              "val_rmse": val_rmse, "best_epoch": best_epoch}, allow_nan=False),
                  flush=True)
            if stale >= config.training.patience:
                break
        best = load_checkpoint(checkpoint_path)
        model.load_state_dict(best["model_state"], strict=True)
        locked_validation = collect_predictions(model, val_loader, device, need_targets=True)
        _write_predictions(config.output_dir / "validation_predictions.csv", locked_validation)
        result = {
            "status": "completed", "run_id": run_id, "best_epoch": best_epoch,
            "epochs_completed": len(history), "best_validation": locked_validation["metrics"],
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
            "train_count": len(train), "val_count": len(val),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "information_regime": regime, "test_evaluated": False,
            "center_coverage": provenance["center_coverage"],
            "model_config": asdict(model_config),
        }
        _json_write(config.output_dir / "summary.json", result)
        _json_write(config.output_dir / "run.json", {**provenance, "status": "completed"})
        return result
    except BaseException as error:
        _json_write(config.output_dir / "run.json", {
            **provenance, "status": "failed", "error_type": type(error).__name__,
            "error": str(error),
        })
        raise


def load_checkpoint(path: str | Path) -> dict:
    try:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError, RuntimeError) as error:
        raise ValueError("Cannot read a safe weights-only newkcat checkpoint") from error
    if (not isinstance(checkpoint, dict) or type(checkpoint.get("format_version")) is not int
            or checkpoint.get("format_version") != CHECKPOINT_VERSION):
        raise ValueError("Unsupported newkcat checkpoint format")
    required = {"model_config", "model_state", "ec_vocab", "source_hashes", "split_ids",
                "train_identity", "val_identity", "information_regime", "run_id"}
    if required - checkpoint.keys():
        raise ValueError(f"Checkpoint lacks provenance: {sorted(required-checkpoint.keys())}")
    try:
        model_config = ModelConfig(**_mapping(checkpoint["model_config"], "checkpoint model_config"))
    except TypeError as error:
        raise ValueError("Invalid checkpoint model configuration") from error
    vocabulary = _mapping(checkpoint["ec_vocab"], "checkpoint ec_vocab")
    if (any(ec2_token(key) != key or type(value) is not int or value < 1
            for key, value in vocabulary.items())
            or set(vocabulary.values()) != set(range(1, len(vocabulary) + 1))
            or model_config.ec_vocab_size != len(vocabulary) + 1):
        raise ValueError("Checkpoint EC vocabulary is invalid or inconsistent with the model")
    state = _mapping(checkpoint["model_state"], "checkpoint model_state")
    if not state or any(not isinstance(value, Tensor) for value in state.values()):
        raise ValueError("Checkpoint model_state must be a nonempty mapping of tensors")
    if any(value.is_complex() or not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Checkpoint model weights must be finite real tensors")

    def valid_hash(value):
        return (isinstance(value, str) and len(value) == 64
                and all(character in "0123456789abcdef" for character in value))

    hashes = _mapping(checkpoint["source_hashes"], "checkpoint source_hashes")
    if (not all(valid_hash(hashes.get(name)) for name in ("samples_sha256", "features_sha256"))
            or "centers_sha256" not in hashes
            or (hashes["centers_sha256"] is not None and not valid_hash(hashes["centers_sha256"]))):
        raise ValueError("Checkpoint source hashes are invalid")

    def string_list(value, label):
        if (not isinstance(value, list) or not all(isinstance(item, str) and item for item in value)
                or len(value) != len(set(value))):
            raise ValueError(f"Checkpoint {label} must be a list of unique nonempty strings")
        return set(value)

    splits = _mapping(checkpoint["split_ids"], "checkpoint split_ids")
    if not {"train", "val"}.issubset(splits) or set(splits) - {"train", "val", "test"}:
        raise ValueError("Checkpoint split_ids must include train and val")
    seen = set()
    for split, ids in splits.items():
        current = string_list(ids, f"split_ids.{split}")
        if seen & current:
            raise ValueError("Checkpoint contains sample IDs shared between splits")
        seen.update(current)
    for split in ("train", "val"):
        identity = _mapping(checkpoint[f"{split}_identity"], f"checkpoint {split}_identity")
        for key in ("sample_ids", "sequence_hashes", "group_ids"):
            string_list(identity.get(key), f"{split}_identity.{key}")
        if (set(identity["sample_ids"]) != set(splits[split])
                or not identity["sequence_hashes"]
                or not all(valid_hash(value) for value in identity["sequence_hashes"])):
            raise ValueError("Checkpoint split identities are inconsistent")
    for key in ("sequence_hashes", "group_ids"):
        if set(checkpoint["train_identity"][key]) & set(checkpoint["val_identity"][key]):
            raise ValueError("Checkpoint train/validation identities overlap")
    if not isinstance(checkpoint["run_id"], str) or not checkpoint["run_id"]:
        raise ValueError("Checkpoint run_id must be a nonempty string")
    if not isinstance(checkpoint["information_regime"], str) or checkpoint["information_regime"] not in {
        "sequence_substrate", "sequence_substrate_ec", "predicted_centers",
        "predicted_centers_with_ec", "reference_center_diagnostic",
    }:
        raise ValueError("Checkpoint information regime is invalid")
    return checkpoint


def _check_test_overlap(dataset: KcatDataset, checkpoint: dict) -> None:
    current = _record_identity(dataset.records)
    for source in ("train_identity", "val_identity"):
        for field in ("sample_ids", "sequence_hashes", "group_ids"):
            overlap = set(current[field]) & set(checkpoint[source][field])
            if overlap:
                raise ValueError(
                    f"Held-out test overlaps checkpoint {source} in {field} "
                    f"({len(overlap)} overlap(s)). This cannot be reported as a held-out test."
                )


def _inference_context(config: RunConfig, checkpoint_path: str | Path, split: str | None,
                       need_targets: bool, allow_reference: bool):
    device = seed_everything(config.training)
    checkpoint = load_checkpoint(checkpoint_path)
    model_config = ModelConfig(**checkpoint["model_config"])
    dataset = _dataset(config, [split] if split else None,
                       ec_vocab=checkpoint["ec_vocab"], require_targets=need_targets)
    if model_config.atom_dim != dataset.atom_dim or model_config.bond_dim != dataset.bond_dim:
        raise ValueError("Checkpoint and current graph featurizer dimensions differ")
    regime = _information_regime(
        model_config, [dataset], allow_reference or config.training.allow_reference_centers
    )
    if need_targets and split == "test":
        _check_test_overlap(dataset, checkpoint)
    model = NewKcat(model_config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    metadata = {
        "format_version": CHECKPOINT_VERSION,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_run_id": checkpoint["run_id"],
        "model_config": asdict(model_config),
        "training_information_regime": checkpoint["information_regime"],
        "information_regime": regime,
        "center_coverage": _center_coverage(dataset),
        "oracle_diagnostic": "reference_center_diagnostic" in {
            regime, checkpoint["information_regime"]
        },
        "split": split or "all",
        "source_hashes": input_hashes(config),
        "training_source_hashes": checkpoint["source_hashes"],
        "target_definition": "log10(kcat / s^-1)",
        "note": "Architecture and EC vocabulary come from the checkpoint, not the evaluation YAML.",
    }
    return model, _loader(dataset, config.training), device, metadata


def evaluate(config: RunConfig, checkpoint_path: str | Path, *, split: str = "test",
             output_dir: str | Path, overwrite: bool = False,
             allow_reference: bool = False) -> dict:
    model, loader, device, metadata = _inference_context(
        config, checkpoint_path, split, True, allow_reference
    )
    output = Path(output_dir).resolve()
    _prepare_output(output, overwrite)
    collected = collect_predictions(model, loader, device, need_targets=True)
    _write_predictions(output / "predictions.csv", collected)
    result = {**metadata, "metrics": collected["metrics"]}
    _json_write(output / "metrics.json", result)
    return result


def predict(config: RunConfig, checkpoint_path: str | Path, *, split: str | None = None,
            output_dir: str | Path, overwrite: bool = False,
            allow_reference: bool = False) -> dict:
    model, loader, device, metadata = _inference_context(
        config, checkpoint_path, split, False, allow_reference
    )
    output = Path(output_dir).resolve()
    _prepare_output(output, overwrite)
    collected = collect_predictions(model, loader, device, need_targets=False)
    _write_predictions(output / "predictions.csv", collected)
    result = {**metadata, "n": len(collected["prediction"]), "metrics_computed": False}
    _json_write(output / "prediction_metadata.json", result)
    return result


def audit_centers(config: RunConfig, checkpoint_path: str | Path, *, split: str = "val",
                  output_dir: str | Path, overwrite: bool = False,
                  allow_reference: bool = False) -> dict:
    """Fixed-weight interventions; never select/retrain weights with this audit."""
    model, loader, device, metadata = _inference_context(
        config, checkpoint_path, split, True, allow_reference
    )
    output = Path(output_dir).resolve()
    _prepare_output(output, overwrite)
    results = {}
    original = None
    for intervention in ("original", "zero", "permuted"):
        collected = collect_predictions(
            model, loader, device, need_targets=True, intervention=intervention,
            seed=config.training.seed,
        )
        values = np.asarray(collected["prediction"])
        if original is None:
            original = values.copy()
        results[intervention] = {
            **collected["metrics"],
            "mean_abs_prediction_change": float(np.mean(np.abs(values-original))),
        }
        _write_predictions(output / f"predictions_{intervention}.csv", collected)
    result = {
        **metadata, "interventions": results, "weights_changed": False,
        "note": "Permutation/zero-center checks are sensitivity diagnostics, not causal proof. "
                "Use validation for method development; do not tune repeatedly on test audits.",
    }
    _json_write(output / "center_audit.json", result)
    return result
