"""Training and locked-checkpoint evaluation for replacement chemical encoders."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields, replace
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
import uuid

import torch
from torch.utils.data import DataLoader
import yaml

from ..chem import ATOM_DIM, BOND_DIM
from ..metrics import regression_metrics
from ..model import ModelConfig, NewKcat
from ..train import (
    NonSingletonBatchSampler, TrainingConfig, _UniqueConfigLoader, _check_test_overlap,
    _git_provenance, _json_write, _mapping, _prepare_output, _record_identity,
    checked_mse, seed_everything, sha256_file,
)
from .data import SequenceDataset, collate_sequences
from .model import ReactionSequenceModel, SequenceModelConfig
from .tokenizer import ReactionTokenizer


ENCODERS = ("catapro_features", "substrate_smiles", "reaction_smiles", "sr_smiles")


@dataclass(frozen=True)
class SequenceRunConfig:
    config_path: Path
    samples: Path
    features: Path
    chemical_encoder: str
    model: SequenceModelConfig
    training: TrainingConfig
    output_dir: Path
    missing_reaction: str = "error"
    encoder_checkpoint: Path | None = None
    freeze_encoder: bool = False

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


def load_config(path: str | Path, *, encoder: str | None = None,
                pooling: str | None = None) -> SequenceRunConfig:
    path = Path(path).resolve()
    try:
        with path.open(encoding="utf-8") as handle:
            raw = _mapping(yaml.load(handle, Loader=_UniqueConfigLoader), "sequence config")
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML: {error}") from error
    allowed = {"data", "model", "training", "output_dir", "encoder_checkpoint", "freeze_encoder"}
    if set(raw)-allowed:
        raise ValueError(f"Unknown sequence config fields: {sorted(set(raw)-allowed)}")

    def resolve(value, label):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must be a nonempty path")
        p = Path(value).expanduser()
        return (path.parent / p).resolve() if not p.is_absolute() else p.resolve()

    data = _mapping(raw.get("data"), "data")
    if set(data)-{"samples", "features", "missing_reaction"}:
        raise ValueError("Sequence data supports samples, features and missing_reaction")
    missing = data.get("missing_reaction", "error")
    if missing not in {"error", "substrate", "reaction"}:
        raise ValueError("missing_reaction must be error, substrate or reaction")
    model_raw = dict(_mapping(raw.get("model", {}), "model"))
    representation = encoder or model_raw.pop("chemical_encoder", "sr_smiles")
    model_raw.pop("chemical_encoder", None)
    if representation not in ENCODERS:
        raise ValueError(f"chemical_encoder must be one of {ENCODERS}")
    if pooling is not None:
        model_raw["pooling"] = pooling
    if set(model_raw)-{f.name for f in fields(SequenceModelConfig)}:
        raise ValueError("Unknown sequence model fields")
    model = SequenceModelConfig(**model_raw)
    if model.pooling == "edit_conditioned" and representation != "sr_smiles":
        raise ValueError("edit_conditioned requires sr_smiles with parsed dynamic spans")
    options = dict(_mapping(raw.get("training", {}), "training"))
    training_fields = {f.name for f in fields(TrainingConfig)}-{"allow_reference_centers"}
    if set(options)-training_fields:
        raise ValueError(f"Unknown sequence training fields: {sorted(set(options)-training_fields)}")
    training = TrainingConfig(**options)
    freeze = raw.get("freeze_encoder", False)
    if type(freeze) is not bool:
        raise ValueError("freeze_encoder must be a YAML boolean")
    checkpoint = (resolve(raw["encoder_checkpoint"], "encoder_checkpoint")
                  if raw.get("encoder_checkpoint") is not None else None)
    if freeze and checkpoint is None:
        raise ValueError("Do not freeze a random encoder: supply encoder_checkpoint")
    if representation == "catapro_features" and (checkpoint or freeze):
        raise ValueError("catapro_features does not use a sequence encoder checkpoint")
    return SequenceRunConfig(
        path, resolve(data.get("samples"), "data.samples"),
        resolve(data.get("features"), "data.features"), representation, model, training,
        resolve(raw.get("output_dir", "runs/sequence"), "output_dir"), missing, checkpoint, freeze,
    )


def _hashes(config: SequenceRunConfig) -> dict:
    return {"samples_sha256": sha256_file(config.samples),
            "features_sha256": sha256_file(config.features)}


def _protect_output(output: Path, inputs: tuple[Path | None, ...]) -> None:
    for path in inputs:
        if path is not None and (path.resolve() == output or output in path.resolve().parents):
            raise ValueError(f"Output directory must not contain input artifact {path}")


def _dataset(config, splits=None, require_targets=True):
    _verify_preparation(config)
    return SequenceDataset(
        config.samples, config.features, chemical_encoder=config.chemical_encoder,
        splits=splits, max_length=config.model.max_length, require_targets=require_targets,
        missing_reaction=config.missing_reaction,
    )


def _verify_preparation(config) -> dict | None:
    if config.chemical_encoder not in {"reaction_smiles", "sr_smiles"}:
        return None
    report_path = config.samples.parent / "coverage.json"
    provenance_path = config.samples.parent / "reaction_provenance.jsonl"
    if not report_path.exists() or not provenance_path.exists():
        raise ValueError("R/C inputs require sequence prepare artifacts: coverage.json and "
                         "reaction_provenance.jsonl beside samples.csv")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("Invalid preparation coverage.json") from error
    if (not isinstance(report, dict) or report.get("schema_version") != 1
            or report.get("samples_written") is not True or report.get("rows_dropped") != 0
            or report.get("tokenizer") != ReactionTokenizer().metadata()):
        raise ValueError("Incompatible preparation manifest")
    if (report.get("samples_sha256") != sha256_file(config.samples)
            or report.get("provenance_sha256") != sha256_file(provenance_path)):
        raise ValueError("Prepared samples or reaction provenance changed; rerun sequence prepare")
    return {"coverage_sha256": sha256_file(report_path),
            "provenance_sha256": report["provenance_sha256"],
            "converter_versions": report.get("converter_versions", []),
            "decoder_compat": report.get("decoder_compat", []),
            "preparation_mode": report.get("mode", "paired"),
            "policy": report.get("policy"), "verification": "prepared_artifact_hashes_match"}


def _model(config):
    if config.chemical_encoder == "catapro_features":
        return NewKcat(ModelConfig(atom_dim=ATOM_DIM, bond_dim=BOND_DIM,
                                   mode="baseline", dropout=config.model.dropout))
    return ReactionSequenceModel(config.model)


def _loader(dataset, config):
    return DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False,
                      num_workers=config.training.num_workers, collate_fn=collate_sequences)


def _safe_load(path):
    try:
        return torch.load(Path(path), map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError, RuntimeError) as error:
        raise ValueError("Cannot load a safe weights-only sequence checkpoint") from error


def _load_initial_encoder(model, config):
    if config.encoder_checkpoint is None:
        return {"kind": "random_initialization", "pretrained": False}
    checkpoint = _safe_load(config.encoder_checkpoint)
    if (not isinstance(checkpoint, dict) or checkpoint.get("kind") != "newkcat_sequence_mlm"
            or checkpoint.get("format_version") != 1):
        raise ValueError("encoder_checkpoint must be a newkcat sequence MLM checkpoint")
    if checkpoint.get("chemical_encoder") != config.chemical_encoder:
        raise ValueError("Pretrained representation differs from the requested chemical_encoder")
    tokenizer = ReactionTokenizer()
    if checkpoint.get("tokenizer") != tokenizer.metadata():
        raise ValueError("Pretrained tokenizer metadata does not match")
    model.encoder.load_encoder(checkpoint.get("encoder"), tokenizer.fingerprint)
    model.set_encoder_frozen(config.freeze_encoder)
    return {"kind": "masked_language_model", "pretrained": True,
            "checkpoint_sha256": sha256_file(config.encoder_checkpoint),
            "path": str(config.encoder_checkpoint), "frozen": config.freeze_encoder,
            "pretraining_provenance": checkpoint.get("provenance", {})}


def collect_predictions(model, loader, device, *, need_targets=True):
    model.eval()
    predictions, targets, ids, statuses = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            values = model(batch)
            if values.shape != (batch.batch_size,) or not bool(torch.isfinite(values).all()):
                raise ValueError("Predictions must be finite vectors [B]")
            predictions.extend(values.cpu().tolist())
            ids.extend(batch.sample_ids)
            statuses.extend(batch.statuses)
            if need_targets:
                if batch.target is None:
                    raise ValueError("Evaluation requires observed targets")
                targets.extend(batch.target.cpu().tolist())
    if not ids:
        raise ValueError("No selected samples to predict")
    result = {"sample_ids": ids, "prediction": predictions, "statuses": statuses}
    if need_targets:
        result.update(target=targets, metrics=regression_metrics(targets, predictions))
        result["metrics_by_input_status"] = {
            status: regression_metrics(
                [value for value, observed in zip(targets, statuses) if observed == status],
                [value for value, observed in zip(predictions, statuses) if observed == status],
            ) for status in sorted(set(statuses))
        }
    return result


def _write_predictions(path, result):
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["sample_id", "prediction", "input_status"]
        if "target" in result:
            fieldnames.append("target")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, sample_id in enumerate(result["sample_ids"]):
            row = {"sample_id": sample_id, "prediction": result["prediction"][i],
                   "input_status": result["statuses"][i]}
            if "target" in result:
                row["target"] = result["target"][i]
            writer.writerow(row)


def validate(config: SequenceRunConfig, *, split: str | None = None) -> dict:
    dataset = _dataset(config, [split] if split else None)
    if not len(dataset):
        raise ValueError("No samples selected")
    model = _model(config)
    return {
        "chemical_encoder": config.chemical_encoder, "n": len(dataset),
        "split": split or "all", "coverage": dataset.coverage,
        "model_config": asdict(config.model), "tokenizer": ReactionTokenizer().metadata(),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "source_hashes": _hashes(config), "missing_reaction": config.missing_reaction,
        "preparation": _verify_preparation(config),
        "information_regime": _regime(config.chemical_encoder),
        "representation_policy": _representation_policy(config),
    }


def _regime(encoder):
    return "enzyme_substrate" if encoder in {"catapro_features", "substrate_smiles"} else "enzyme_known_reaction"


def _representation_policy(config) -> str:
    if config.chemical_encoder == "sr_smiles" and config.missing_reaction == "reaction":
        return "sr_smiles_with_full_reaction_fallback"
    if config.chemical_encoder in {"reaction_smiles", "sr_smiles"} and config.missing_reaction == "substrate":
        return f"{config.chemical_encoder}_with_substrate_fallback"
    return config.chemical_encoder


def fit(config: SequenceRunConfig, *, overwrite: bool = False) -> dict:
    device = seed_everything(config.training)
    train, val = _dataset(config, ["train"]), _dataset(config, ["val"])
    if len(train) < 2 or len(val) < 1:
        raise ValueError("Need at least two training rows and one validation row")
    model = _model(config)
    initialization = ({"kind": "cached_catapro_features", "regression_head": "random_initialization"}
                      if config.chemical_encoder == "catapro_features"
                      else _load_initial_encoder(model, config))
    model.to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config.training.lr, weight_decay=config.training.weight_decay)
    sampler = NonSingletonBatchSampler(len(train), config.training.batch_size, config.training.seed)
    train_loader = DataLoader(train, batch_sampler=sampler, collate_fn=collate_sequences,
                              num_workers=config.training.num_workers)
    val_loader = _loader(val, config)
    _protect_output(config.output_dir, (config.samples, config.features, config.config_path,
                                        config.encoder_checkpoint))
    _prepare_output(config.output_dir, overwrite)
    splits = {split: [row["sample_id"] for row in train.all_records if row["split"] == split]
              for split in ("train", "val", "test")}
    metadata = {
        "kind": "newkcat_sequence_regressor", "format_version": 1,
        "chemical_encoder": config.chemical_encoder, "model_config": asdict(config.model),
        "tokenizer": ReactionTokenizer().metadata(), "missing_reaction": config.missing_reaction,
        "initialization": initialization, "source_hashes": _hashes(config),
        "train_identity": _record_identity(train.records), "val_identity": _record_identity(val.records),
        "split_ids": splits, "run_id": uuid.uuid4().hex, "git": _git_provenance(),
        "resolved_config": config.snapshot(), "target_definition": "log10(kcat / s^-1)",
        "information_regime": _regime(config.chemical_encoder),
        "representation_policy": _representation_policy(config),
        "coverage": {"train": train.coverage, "val": val.coverage},
        "preparation": _verify_preparation(config),
    }
    _json_write(config.output_dir / "run.json", {**metadata, "status": "running"})
    with (config.output_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.snapshot(), handle, sort_keys=False)
    best, anchor, stale, best_epoch = float("inf"), float("inf"), 0, 0
    history = []
    destination = config.output_dir / "best.pt"
    try:
        for epoch in range(1, config.training.epochs+1):
            sampler.set_epoch(epoch)
            model.train()
            total, count = 0.0, 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = checked_mse(model(batch), batch.target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip,
                                              error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach().cpu()) * batch.batch_size
                count += batch.batch_size
            if count != len(train):
                raise RuntimeError("Training did not visit every row exactly once")
            validation = collect_predictions(model, val_loader, device)
            score = validation["metrics"]["rmse"]
            entry = {"epoch": epoch, "train_rmse": (total/count)**.5,
                     "val": validation["metrics"]}
            history.append(entry)
            with (config.output_dir / "history.jsonl").open("w" if epoch == 1 else "a") as handle:
                handle.write(json.dumps(entry, allow_nan=False)+"\n")
            if score < best:
                best, best_epoch = score, epoch
                checkpoint = {**metadata, "epoch": epoch, "best_val_rmse": score,
                              "model_state": {k: v.detach().cpu().clone()
                                              for k, v in model.state_dict().items()}}
                temporary = destination.with_suffix(".pt.tmp")
                torch.save(checkpoint, temporary)
                temporary.replace(destination)
            if score < anchor-config.training.min_delta:
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
        result = {"status": "completed", "chemical_encoder": config.chemical_encoder,
                  "best_epoch": best_epoch, "epochs_completed": len(history),
                  "best_validation": locked["metrics"], "train_count": len(train),
                  "val_count": len(val), "test_evaluated": False,
                  "checkpoint": str(destination), "checkpoint_sha256": sha256_file(destination),
                  "parameter_count": sum(p.numel() for p in model.parameters()),
                  "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
                  "initialization": initialization, "coverage": metadata["coverage"],
                  "representation_policy": _representation_policy(config),
                  "validation_by_input_status": locked["metrics_by_input_status"]}
        _json_write(config.output_dir / "summary.json", result)
        _json_write(config.output_dir / "run.json", {**metadata, "status": "completed"})
        return result
    except BaseException as error:
        _json_write(config.output_dir / "run.json", {**metadata, "status": "failed",
                                                   "error": str(error)})
        raise


def load_checkpoint(path) -> dict:
    checkpoint = _safe_load(path)
    if (not isinstance(checkpoint, dict) or checkpoint.get("kind") != "newkcat_sequence_regressor"
            or type(checkpoint.get("format_version")) is not int or checkpoint["format_version"] != 1):
        raise ValueError("Unsupported sequence regressor checkpoint")
    required = {"chemical_encoder", "model_config", "tokenizer", "missing_reaction", "model_state",
                "source_hashes", "split_ids", "train_identity", "val_identity", "run_id"}
    if required-checkpoint.keys():
        raise ValueError(f"Checkpoint lacks provenance: {sorted(required-checkpoint.keys())}")
    encoder = checkpoint["chemical_encoder"]
    if encoder not in ENCODERS or checkpoint["missing_reaction"] not in {"error", "substrate", "reaction"}:
        raise ValueError("Invalid checkpoint representation or missing-reaction policy")
    try:
        model = SequenceModelConfig(**_mapping(checkpoint["model_config"], "model_config"))
    except TypeError as error:
        raise ValueError("Invalid checkpoint model_config") from error
    if model.pooling == "edit_conditioned" and encoder != "sr_smiles":
        raise ValueError("Checkpoint uses edit pooling with incompatible representation")
    if checkpoint["tokenizer"] != ReactionTokenizer().metadata():
        raise ValueError("Checkpoint tokenizer metadata differs from this implementation")
    state = _mapping(checkpoint["model_state"], "model_state")
    if not state or any(not isinstance(value, torch.Tensor) or value.is_complex()
                        or not bool(torch.isfinite(value).all()) for value in state.values()):
        raise ValueError("Checkpoint weights must be finite real tensors")
    expected = _model(SimpleNamespace(chemical_encoder=encoder, model=model)).state_dict()
    if set(state) != set(expected) or any(
            state[key].shape != expected[key].shape or state[key].dtype != expected[key].dtype
            for key in expected if key in state):
        raise ValueError("Checkpoint tensor keys, shapes or dtypes disagree with the stored architecture")
    def string_set(value):
        if (not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value)
                or len(set(value)) != len(value)):
            raise ValueError("Checkpoint identity lists must contain unique nonempty strings")
        return set(value)
    splits = _mapping(checkpoint["split_ids"], "split_ids")
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("Checkpoint split metadata must contain train, val, test")
    seen = set()
    for key, values in splits.items():
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
                or any(c not in "0123456789abcdef" for c in value)):
            raise ValueError("Invalid checkpoint source hash")
    return checkpoint


def infer(config: SequenceRunConfig, checkpoint_path: str | Path, *, output_dir: str | Path,
          split: str | None = "test", evaluate: bool = True, overwrite: bool = False) -> dict:
    if evaluate and split not in {"train", "val", "test"}:
        raise ValueError("Evaluation requires one explicit split")
    checkpoint = load_checkpoint(checkpoint_path)
    # Input files/device may change; trained representation, architecture and fallback policy cannot.
    effective = replace(config, chemical_encoder=checkpoint["chemical_encoder"],
                        model=SequenceModelConfig(**checkpoint["model_config"]),
                        missing_reaction=checkpoint["missing_reaction"])
    device = seed_everything(effective.training)
    dataset = _dataset(effective, [split] if split else None, require_targets=evaluate)
    if evaluate and split == "test":
        _check_test_overlap(dataset, checkpoint)
    model = _model(effective)
    try:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    except RuntimeError as error:
        raise ValueError("Checkpoint weights disagree with the stored architecture") from error
    model.to(device)
    output = Path(output_dir).resolve()
    _protect_output(output, (effective.samples, effective.features, Path(checkpoint_path),
                             effective.config_path))
    _prepare_output(output, overwrite)
    collected = collect_predictions(model, _loader(dataset, effective), device, need_targets=evaluate)
    _write_predictions(output / "predictions.csv", collected)
    result = {"chemical_encoder": effective.chemical_encoder, "model_config": asdict(effective.model),
              "split": split or "all", "n": len(dataset), "coverage": dataset.coverage,
              "checkpoint_sha256": sha256_file(checkpoint_path), "source_hashes": _hashes(effective),
              "training_source_hashes": checkpoint["source_hashes"], "weights_changed": False,
              "preparation": _verify_preparation(effective),
              "information_regime": _regime(effective.chemical_encoder),
              "representation_policy": _representation_policy(effective),
              "metrics_computed": evaluate,
              "note": "Architecture, tokenizer and fallback policy are locked by the checkpoint."}
    if evaluate:
        result["metrics"] = collected["metrics"]
        result["metrics_by_input_status"] = collected["metrics_by_input_status"]
    _json_write(output / ("metrics.json" if evaluate else "prediction_metadata.json"), result)
    return result
