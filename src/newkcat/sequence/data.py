"""Graph-independent, ID-aligned inputs for chemical sequence experiments.

Only selected splits have targets, features, and chemical strings materialized.
The entire manifest still undergoes the existing ID/sequence/group leakage checks.
Native reaction text and sr conversion have separate availability. Explicit
fallback policies never drop a row or invent a reaction.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, fields, replace
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..data import SPLITS, _selected_features, read_manifest
from .tokenizer import BOS_ID, EOS_ID, PAD_ID, ReactionTokenizer


CHEMICAL_ENCODERS = frozenset({
    "catapro_features", "substrate_smiles", "reaction_smiles", "sr_smiles",
})
REACTION_ENCODERS = frozenset({"reaction_smiles", "sr_smiles"})


def _selected_protein(path: str | Path, sample_ids: list[str]) -> np.ndarray:
    """Read only the protein feature family; MolT5/MACCS are not required."""
    with np.load(path, allow_pickle=False) as archive:
        if not {"sample_ids", "protein"}.issubset(archive.files):
            raise ValueError("Feature NPZ needs sample_ids and protein arrays")
        ids = archive["sample_ids"]
        if ids.ndim != 1 or ids.dtype.kind != "U":
            raise ValueError("Feature sample_ids must be a 1D Unicode array, never object/pickle")
        id_list = ids.tolist()
        if any(not value or value != value.strip() for value in id_list):
            raise ValueError("Feature sample_ids must be nonempty with no surrounding whitespace")
        if len(set(id_list)) != len(id_list):
            raise ValueError("Duplicate sample_ids in feature cache")
        lookup = {value: index for index, value in enumerate(id_list)}
        missing = [value for value in sample_ids if value not in lookup]
        if missing:
            raise ValueError(f"Selected samples missing feature rows: {missing[:5]}")
        array = archive["protein"]
        if array.shape != (len(ids), 1024):
            raise ValueError(f"Feature protein must have shape [{len(ids)}, 1024]")
        if array.dtype.kind not in "fiu":
            raise ValueError("Feature protein must be a real numeric array")
        indices = np.asarray([lookup[value] for value in sample_ids], dtype=np.int64)
        selected = array[indices]
        if not np.isfinite(selected).all():
            raise ValueError("Selected protein features contain nonfinite values")
        with np.errstate(over="ignore", invalid="ignore"):
            selected = selected.astype(np.float32, copy=True)
        if not np.isfinite(selected).all():
            raise ValueError("Selected protein features overflow float32")
        return selected


def _parse_spans(raw: str, text: str) -> list[list[int]]:
    """Reject invalid annotation metadata instead of turning it into fallback."""
    try:
        spans = json.loads(raw) if raw else []
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("edit_spans must be a JSON list of [start, end] pairs") from error
    if not isinstance(spans, list):
        raise ValueError("edit_spans must be a JSON list")
    previous_end = 0
    for pair in spans:
        if (not isinstance(pair, list) or len(pair) != 2
                or any(type(value) is not int for value in pair)):
            raise ValueError("Each edit span must be [integer start, integer end]")
        start, end = pair
        if not 0 <= start < end <= len(text):
            raise ValueError("edit_spans contains an out-of-range or empty span")
        if start < previous_end:
            raise ValueError("edit_spans must be ordered and non-overlapping")
        previous_end = end
    return spans


def _target(row: dict[str, Any], require_targets: bool) -> float | None:
    if not require_targets:
        # Prediction neither needs nor materializes labels, including partially
        # labeled inputs. Evaluation uses require_targets=True explicitly.
        return None
    value = row["log10_kcat"]
    if value == "":
        raise ValueError("Missing log10_kcat target")
    try:
        target = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid log10_kcat target") from error
    if not math.isfinite(target) or abs(target) > float(np.finfo(np.float32).max):
        raise ValueError("log10_kcat must be finite float32")
    return target


def _substrate_text(row: dict[str, Any]) -> str:
    # The sequence path permits RDKit dummy atoms. It does not pass through the
    # stricter graph featurizer used by the original graph branch.
    from .chemistry import normalize_substrate

    text = normalize_substrate(row.get("substrate_smiles") or row["smiles"])
    if row.get("substrate_smiles"):
        original = normalize_substrate(row["smiles"])
        if text != original:
            raise ValueError("Prepared substrate_smiles differs from original substrate")
    return text


def _reaction_text(row: dict[str, Any]) -> str:
    """Validate prepared full reaction text without requiring atom mapping."""
    from .chemistry import assert_map_free, normalize_reaction

    text = row.get("reaction_smiles", "")
    if not text:
        raise ValueError("Declared reaction text is empty")
    assert_map_free(text)
    normalize_reaction(text)
    return text


class SequenceDataset(Dataset):
    """Use one manifest and protein cache across replacement/baseline arms.

    ``catapro_features`` reads the existing three cached feature arrays without
    parsing chemical graphs or requiring reaction annotations. Other encoders
    read only ``protein``. ``missing_reaction='substrate'`` is an explicit
    experiment setting shared by R/C. ``missing_reaction='reaction'`` lets C
    read explicitly prepared full R text if only its sr conversion failed.
    """

    def __init__(
        self,
        samples_path: str | Path,
        features_path: str | Path,
        chemical_encoder: str = "sr_smiles",
        splits: list[str] | tuple[str, ...] | set[str] | None = None,
        tokenizer: ReactionTokenizer | None = None,
        max_length: int = 2048,
        require_targets: bool = True,
        missing_reaction: str = "error",
    ):
        if chemical_encoder not in CHEMICAL_ENCODERS:
            raise ValueError(f"Unknown chemical_encoder: {chemical_encoder!r}")
        if missing_reaction not in {"error", "substrate", "reaction"}:
            raise ValueError("missing_reaction must be 'error', 'substrate' or 'reaction'")
        if type(require_targets) is not bool:
            raise ValueError("require_targets must be a boolean")
        if type(max_length) is not int or max_length < 3:
            raise ValueError("max_length must be an integer >= 3")
        if splits is not None and (
            isinstance(splits, str) or not splits or not set(splits).issubset(SPLITS)
        ):
            raise ValueError("splits must be a nonempty collection of train/val/test, or None")
        self.chemical_encoder = chemical_encoder
        self.missing_reaction = missing_reaction
        self.tokenizer = tokenizer if tokenizer is not None else ReactionTokenizer()
        self.max_length = max_length
        selected_splits = SPLITS if splits is None else set(splits)
        self.all_records = read_manifest(samples_path)
        self.records = [dict(row) for row in self.all_records if row["split"] in selected_splits]
        self.rows = self.records
        self.sample_ids = [row["sample_id"] for row in self.records]
        if chemical_encoder == "catapro_features":
            features = _selected_features(features_path, self.sample_ids)
        else:
            features = {"protein": _selected_protein(features_path, self.sample_ids)}
        self.items: list[dict[str, Any]] = []
        for index, row in enumerate(self.records):
            sample_id = row["sample_id"]
            try:
                target = _target(row, require_targets)
                text, spans, status = self._text(row)
                if chemical_encoder == "catapro_features":
                    encoded = {"input_ids": [BOS_ID, EOS_ID], "edit_mask": [False, False]}
                else:
                    encoded = self.tokenizer.encode(text, edit_spans=spans, max_length=max_length)
            except (ValueError, TypeError) as error:
                raise ValueError(f"Sample {sample_id}: {error}") from error
            row["log10_kcat"] = target
            item = {
                **encoded,
                "protein": features["protein"][index],
                "target": target,
                "sample_id": sample_id,
                "status": status,
                "text": text,
            }
            if chemical_encoder == "catapro_features":
                item["molt5"] = features["molt5"][index]
                item["maccs"] = features["maccs"][index]
            self.items.append(item)

    def _text(self, row: dict[str, Any]) -> tuple[str, list[list[int]], str]:
        if self.chemical_encoder == "catapro_features":
            return "", [], "catapro_features"
        if self.chemical_encoder == "substrate_smiles":
            return _substrate_text(row), [], "substrate_smiles"
        status = row.get("reaction_status", "")
        if status not in {"ok", "", "missing", "unavailable", "fallback_substrate",
                          "fallback_reaction", "reaction_only"}:
            raise ValueError(f"Unknown reaction_status: {status!r}")
        if status == "reaction_only":
            if self.chemical_encoder != "reaction_smiles":
                raise ValueError("reaction_only inputs have not undergone sr preparation; "
                                 "use reaction_smiles or prepare the sr representation")
            return _reaction_text(row), [], "reaction_only"
        if status == "fallback_reaction":
            if self.chemical_encoder == "reaction_smiles":
                return _reaction_text(row), [], "fallback_reaction"
            if self.missing_reaction != "reaction":
                raise ValueError("sr conversion failed but full reaction text is available; "
                                 "select missing_reaction='reaction' explicitly for hybrid input")
            if row.get("sr_smiles", "") or _parse_spans(row.get("edit_spans", ""), ""):
                raise ValueError("fallback_reaction must have no sr text or edit annotations")
            return _reaction_text(row), [], "fallback_reaction"
        if status != "ok":
            if self.missing_reaction != "substrate":
                raise ValueError(
                    f"Reaction unavailable (reaction_status={status!r}); prepare the manifest "
                    "or explicitly select missing_reaction='substrate'"
                )
            return _substrate_text(row), [], "fallback_substrate"
        text = row.get(self.chemical_encoder, "")
        if not text:
            raise ValueError(f"reaction_status='ok' but {self.chemical_encoder} is empty")
        if self.chemical_encoder == "sr_smiles":
            from .chemistry import dynamic_spans

            spans = _parse_spans(row.get("edit_spans", ""), text)
            if spans != dynamic_spans(text):
                raise ValueError("Stored edit_spans do not match the sr-SMILES dynamic fields")
        else:
            # The reaction-SMILES control has no supplied center/edit labels.
            text = _reaction_text(row)
            spans = []
        return text, spans, "ok"

    @property
    def coverage(self) -> dict[str, int]:
        return dict(sorted(Counter(item["status"] for item in self.items).items()))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


@dataclass(frozen=True)
class SequenceBatch:
    protein: Tensor
    input_ids: Tensor
    attention_mask: Tensor
    edit_mask: Tensor
    target: Tensor | None
    sample_ids: tuple[str, ...]
    statuses: tuple[str, ...]
    molt5: Tensor | None = None
    maccs: Tensor | None = None

    @property
    def batch_size(self) -> int:
        return self.protein.shape[0]

    def to(self, device: torch.device | str) -> SequenceBatch:
        updates = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Tensor):
                updates[field.name] = value.to(device)
        return replace(self, **updates)


def collate_sequences(items: list[dict[str, Any]]) -> SequenceBatch:
    """Pad on the right; edit flags never apply to BOS/EOS or padding."""
    if not items:
        raise ValueError("Cannot collate an empty batch")
    width = max(len(item["input_ids"]) for item in items)
    if width < 2:
        raise ValueError("Each sequence needs BOS and EOS")
    input_ids = torch.full((len(items), width), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    edit_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for index, item in enumerate(items):
        ids, edits = item["input_ids"], item["edit_mask"]
        if (len(ids) < 2 or ids[0] != BOS_ID or ids[-1] != EOS_ID
                or len(edits) != len(ids) or any(type(v) is not int for v in ids)
                or any(type(v) is not bool for v in edits)):
            raise ValueError("Malformed encoded sequence or edit mask")
        if PAD_ID in ids or edits[0] or edits[-1]:
            raise ValueError("Unpadded input cannot contain PAD; BOS/EOS cannot be edited")
        count = len(ids)
        input_ids[index, :count] = torch.tensor(ids, dtype=torch.long)
        attention_mask[index, :count] = True
        edit_mask[index, :count] = torch.tensor(edits, dtype=torch.bool)
    targets = [item["target"] for item in items]
    if any(value is None for value in targets) and not all(value is None for value in targets):
        raise ValueError("Cannot mix labeled and unlabeled targets in one batch")
    target = None if targets[0] is None else torch.tensor(targets, dtype=torch.float32)
    if target is not None and not bool(torch.isfinite(target).all()):
        raise ValueError("Batch targets must be finite")

    def feature(name: str, dimension: int, optional: bool = False) -> Tensor | None:
        present = [name in item for item in items]
        if optional and not any(present):
            return None
        if not all(present):
            raise ValueError(f"Inconsistent {name} feature availability in batch")
        arrays = [np.asarray(item[name]) for item in items]
        if any(array.shape != (dimension,) or array.dtype.kind not in "fiu" for array in arrays):
            raise ValueError(f"Each {name} feature must be a real [{dimension}] vector")
        with np.errstate(over="ignore", invalid="ignore"):
            result = torch.from_numpy(np.stack(arrays).astype(np.float32))
        if not bool(torch.isfinite(result).all()):
            raise ValueError(f"Batch {name} features must be finite float32")
        return result

    return SequenceBatch(
        protein=feature("protein", 1024),
        input_ids=input_ids,
        attention_mask=attention_mask,
        edit_mask=edit_mask,
        target=target,
        sample_ids=tuple(item["sample_id"] for item in items),
        statuses=tuple(item["status"] for item in items),
        molt5=feature("molt5", 768, optional=True),
        maccs=feature("maccs", 167, optional=True),
    )
