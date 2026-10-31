"""Optional masked-character pretraining on explicitly supplied reaction corpora.

No pretrained weights or datasets are downloaded. Corpus separation and optional
exclusion are based on exact text, not chemical graph identity; equivalent SMILES
spellings can therefore require additional external leakage checks.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import re

import torch
from torch import Tensor
from torch.nn import functional as F

from ..train import (
    NonSingletonBatchSampler, TrainingConfig, _git_provenance, _json_write,
    _prepare_output, seed_everything, sha256_file,
)
from .model import ReactionMLM, SequenceModelConfig
from .tokenizer import ReactionTokenizer


REPRESENTATIONS = {"substrate_smiles", "reaction_smiles", "sr_smiles"}


def _read_corpus(path: Path, column: str, role: str) -> tuple[list[str], dict]:
    path = path.resolve()
    values = []
    if path.suffix.lower() == ".txt":
        with path.open(encoding="utf-8-sig") as handle:
            for line, value in enumerate(handle, 1):
                text = value.rstrip("\r\n")
                if not text:
                    raise ValueError(f"Empty corpus text at {path}:{line}")
                values.append(text)
    elif path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            if len(fields) != len(set(fields)) or column not in fields:
                raise ValueError(f"Corpus CSV requires a unique {column!r} column: {path}")
            for line, row in enumerate(reader, 2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Inconsistent corpus CSV column count at {path}:{line}")
                if "split" in row and role != "exclude":
                    allowed = {"train"} if role == "train" else {"val", "valid"}
                    if row["split"] not in allowed:
                        raise ValueError(f"{role} corpus contains a different split at {path}:{line}")
                if not row[column]:
                    raise ValueError(f"Empty corpus text at {path}:{line}")
                values.append(row[column])
    else:
        raise ValueError("MLM corpora must be .txt or .csv files")
    unique = list(dict.fromkeys(values))
    return unique, {"path": str(path), "sha256": sha256_file(path),
                    "rows": len(values), "unique_texts": len(unique),
                    "duplicates_removed": len(values) - len(unique)}


def _validate_representation(text: str, representation: str) -> None:
    if re.search(r":\d+\]", text):
        raise ValueError("MLM text must not contain atom-map IDs; prepare map-free text first")
    if representation == "reaction_smiles":
        if text.count(">") != 2 or ">>" not in text or any(not part for part in text.split(">>")):
            raise ValueError("reaction_smiles MLM text must have reactants>>products form")
    elif ">" in text:
        raise ValueError(f"{representation} MLM text must not contain reaction separators")
    if representation == "substrate_smiles" and ("{" in text or "}" in text):
        raise ValueError("substrate_smiles MLM text must not contain sr-SMILES dynamic groups")


def mask_tokens(
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    probability: float,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Apply reproducible 80/10/10 MLM corruption to CPU character batches.

    Every sequence contributes at least one supervised character. BOS, EOS and
    PAD are never corruption targets. Returns new IDs and labels with -100 at
    all unsupervised positions; the input tensors are not modified.
    """
    tokenizer = ReactionTokenizer()
    if (isinstance(probability, bool) or not isinstance(probability, (int, float))
            or not math.isfinite(probability) or not 0 < probability <= 1):
        raise ValueError("mask_probability must be finite and lie in (0, 1]")
    if (input_ids.ndim != 2 or input_ids.dtype != torch.int64
            or attention_mask.shape != input_ids.shape or attention_mask.dtype != torch.bool
            or input_ids.device.type != "cpu" or attention_mask.device.type != "cpu"):
        raise ValueError("MLM corruption requires CPU int64 IDs and boolean masks of shape [B, L]")
    if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
        raise ValueError("MLM corruption requires a dedicated CPU torch.Generator")
    if bool(((input_ids < 0) | (input_ids >= tokenizer.vocab_size)).any()):
        raise ValueError("MLM input token outside the fixed vocabulary")
    eligible = attention_mask & (input_ids >= 4)
    if eligible.shape[0] < 1 or not bool(eligible.any(dim=1).all()):
        raise ValueError("Each MLM sequence must contain at least one character")
    selected = (torch.rand(input_ids.shape, generator=generator) < probability) & eligible
    for row in range(input_ids.shape[0]):
        if not bool(selected[row].any()):
            candidates = torch.where(eligible[row])[0]
            index = torch.randint(len(candidates), (1,), generator=generator).item()
            selected[row, candidates[index]] = True
    actions = torch.rand(input_ids.shape, generator=generator)
    corrupted = input_ids.clone()
    corrupted[selected & (actions < 0.8)] = tokenizer.mask_id
    random_positions = selected & (actions >= 0.8) & (actions < 0.9)
    replacements = torch.randint(4, tokenizer.vocab_size, input_ids.shape, generator=generator)
    corrupted[random_positions] = replacements[random_positions]
    labels = input_ids.masked_fill(~selected, -100)
    return corrupted, labels


def _padded(items: list[list[int]]) -> tuple[Tensor, Tensor]:
    width = max(map(len, items))
    ids = torch.full((len(items), width), ReactionTokenizer.pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for row, values in enumerate(items):
        ids[row, :len(values)] = torch.tensor(values, dtype=torch.long)
        mask[row, :len(values)] = True
    return ids, mask


def _fixed_validation(items: list[list[int]], texts: list[str], seed: int,
                      probability: float) -> list[tuple[Tensor, Tensor]]:
    cached = []
    for ids, text in zip(items, texts):
        digest = hashlib.sha256(f"mlm-validation:{seed}:{text}".encode()).digest()
        local_seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
        tensor = torch.tensor([ids], dtype=torch.long)
        corrupted, labels = mask_tokens(
            tensor, torch.ones_like(tensor, dtype=torch.bool), probability=probability,
            generator=torch.Generator().manual_seed(local_seed),
        )
        cached.append((corrupted[0], labels[0]))
    return cached


def _validation_loss(model: ReactionMLM, examples: list[tuple[Tensor, Tensor]],
                     batch_size: int, device: torch.device) -> tuple[float, int]:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            rows = examples[start:start + batch_size]
            ids, mask = _padded([row[0].tolist() for row in rows])
            labels = torch.full_like(ids, -100)
            for index, (_, values) in enumerate(rows):
                labels[index, :len(values)] = values
            logits = model(ids.to(device), mask.to(device))
            loss = F.cross_entropy(logits.reshape(-1, model.config.vocab_size),
                                   labels.to(device).reshape(-1), ignore_index=-100, reduction="sum")
            if not bool(torch.isfinite(loss)):
                raise ValueError("MLM validation loss is nonfinite")
            total += float(loss)
            count += int((labels != -100).sum())
    return total / count, count


def pretrain(
    train_path: str | Path,
    val_path: str | Path,
    *,
    output_dir: str | Path,
    model_config: SequenceModelConfig,
    training: TrainingConfig,
    chemical_encoder: str,
    text_column: str | None = None,
    exclude_path: str | Path | None = None,
    mask_probability: float = 0.15,
    overwrite: bool = False,
) -> dict:
    """Train an optional MLM encoder using only caller-declared train/val texts."""
    if chemical_encoder not in REPRESENTATIONS:
        raise ValueError(f"MLM chemical_encoder must be one of {sorted(REPRESENTATIONS)}")
    if not isinstance(model_config, SequenceModelConfig) or not isinstance(training, TrainingConfig):
        raise ValueError("MLM requires SequenceModelConfig and TrainingConfig instances")
    if type(overwrite) is not bool:
        raise ValueError("overwrite must be a boolean")
    if (isinstance(mask_probability, bool) or not isinstance(mask_probability, (int, float))
            or not math.isfinite(mask_probability) or not 0 < mask_probability <= 1):
        raise ValueError("mask_probability must be finite and lie in (0, 1]")
    column = chemical_encoder if text_column is None else text_column
    if not isinstance(column, str) or not column:
        raise ValueError("text_column must be a nonempty string")
    train_path, val_path, output_dir = map(lambda p: Path(p).resolve(),
                                         (train_path, val_path, output_dir))
    sources = [train_path, val_path]
    if exclude_path is not None:
        exclude_path = Path(exclude_path).resolve()
        sources.append(exclude_path)
    if any(source == output_dir or output_dir in source.parents for source in sources):
        raise ValueError("MLM output directory must not contain or overwrite any input corpus")
    train_texts, train_info = _read_corpus(train_path, column, "train")
    val_texts, val_info = _read_corpus(val_path, column, "val")
    exclude_info = None
    excluded: set[str] = set()
    if exclude_path is not None:
        excluded_texts, exclude_info = _read_corpus(exclude_path, column, "exclude")
        excluded = set(excluded_texts)
    train_info["excluded_texts"] = sum(text in excluded for text in train_texts)
    val_info["excluded_texts"] = sum(text in excluded for text in val_texts)
    train_texts = [text for text in train_texts if text not in excluded]
    val_texts = [text for text in val_texts if text not in excluded]
    train_info["used_texts"], val_info["used_texts"] = len(train_texts), len(val_texts)
    if not train_texts or not val_texts:
        raise ValueError("MLM train and validation corpora must both be nonempty after exclusion")
    overlap = set(train_texts) & set(val_texts)
    if overlap:
        raise ValueError(f"MLM train/validation exact-text leakage: {len(overlap)} shared texts")
    tokenizer = ReactionTokenizer()
    encoded = {}
    for text in [*train_texts, *val_texts]:
        _validate_representation(text, chemical_encoder)
        encoded[text] = tokenizer.encode(text, max_length=model_config.max_length)["input_ids"]
    train_items = [encoded[text] for text in train_texts]
    val_items = [encoded[text] for text in val_texts]
    validation = _fixed_validation(val_items, val_texts, training.seed, mask_probability)
    device = seed_everything(training)
    model = ReactionMLM(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.lr,
                                 weight_decay=training.weight_decay)
    sampler = (NonSingletonBatchSampler(len(train_items), training.batch_size, training.seed)
               if len(train_items) >= 2 else None)
    provenance = {
        "initialization": "random_from_scratch", "chemical_encoder": chemical_encoder,
        "model_config": asdict(model_config), "training": asdict(training),
        "mask_probability": mask_probability, "mask_policy": "80_mask_10_random_10_unchanged",
        "validation_masking": "fixed_per_exact_text_and_seed", "text_column": column,
        "train_corpus": train_info, "validation_corpus": val_info,
        "exclusion_corpus": exclude_info,
        "leakage_check": "exact_text_only_not_chemical_identity",
        "tokenizer": tokenizer.metadata(), "git": _git_provenance(),
        "torch_version": str(torch.__version__),
    }
    _prepare_output(output_dir, overwrite)
    _json_write(output_dir / "run.json", {**provenance, "status": "running"})
    best_loss, patience_anchor, best_epoch, stale = float("inf"), float("inf"), 0, 0
    history = []
    checkpoint_path = output_dir / "best_encoder.pt"
    try:
        for epoch in range(1, training.epochs + 1):
            model.train()
            if sampler is not None:
                sampler.set_epoch(epoch)
                batches = iter(sampler)
            else:
                batches = iter([[0]])
            generator = torch.Generator().manual_seed(training.seed + epoch)
            total, token_count = 0.0, 0
            for indices in batches:
                ids, mask = _padded([train_items[index] for index in indices])
                corrupted, labels = mask_tokens(ids, mask, probability=mask_probability,
                                               generator=generator)
                optimizer.zero_grad(set_to_none=True)
                logits = model(corrupted.to(device), mask.to(device))
                supervised = int((labels != -100).sum())
                loss_sum = F.cross_entropy(logits.reshape(-1, model_config.vocab_size),
                                           labels.to(device).reshape(-1),
                                           ignore_index=-100, reduction="sum")
                loss = loss_sum / supervised
                if not bool(torch.isfinite(loss)):
                    raise ValueError("MLM training loss is nonfinite")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), training.grad_clip,
                                               error_if_nonfinite=True)
                optimizer.step()
                total += float(loss_sum.detach())
                token_count += supervised
            val_loss, val_tokens = _validation_loss(model, validation, training.batch_size, device)
            row = {"epoch": epoch, "train_token_ce": total / token_count,
                   "val_token_ce": val_loss, "train_supervised_tokens": token_count,
                   "val_supervised_tokens": val_tokens}
            history.append(row)
            with (output_dir / "history.jsonl").open("w" if epoch == 1 else "a",
                                                     encoding="utf-8") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
                checkpoint = {
                    "kind": "newkcat_sequence_mlm", "format_version": 1,
                    "chemical_encoder": chemical_encoder, "tokenizer": tokenizer.metadata(),
                    "encoder": model.encoder.export_encoder(tokenizer.fingerprint),
                    "provenance": provenance, "best_val_loss": best_loss, "epoch": epoch,
                }
                temporary = checkpoint_path.with_suffix(".pt.tmp")
                torch.save(checkpoint, temporary)
                temporary.replace(checkpoint_path)
            if val_loss < patience_anchor - training.min_delta:
                patience_anchor, stale = val_loss, 0
            else:
                stale += 1
            if stale >= training.patience:
                break
        summary = {
            "status": "completed", "best_epoch": best_epoch, "best_val_loss": best_loss,
            "epochs_completed": len(history), "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "train_texts": len(train_texts), "validation_texts": len(val_texts),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "chemical_encoder": chemical_encoder, "test_evaluated": False,
            "initialization": "random_from_scratch",
            "leakage_check": "exact_text_only_not_chemical_identity",
        }
        _json_write(output_dir / "summary.json", summary)
        _json_write(output_dir / "run.json", {**provenance, **summary})
        return summary
    except BaseException as error:
        _json_write(output_dir / "run.json", {**provenance, "status": "failed",
                                             "error_type": type(error).__name__,
                                             "error": str(error)})
        raise
