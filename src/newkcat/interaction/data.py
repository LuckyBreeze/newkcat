"""ID-aligned, disk-backed residue/reaction caches for interaction models.

The manifest remains the source of row order, split membership and labels. A
cache never supplies labels or silently removes a sample. Protein ``states``
must contain one row per input residue; a pooled feature cannot masquerade as
a residue sequence. Optional global protein features are joined by sample ID.
Without that file, the global feature is the mean over *all* cached residues.

NPZ archives are opened with ``allow_pickle=False`` and loaded individually.
Compressed NPZ does not support useful memory mapping, so a bounded LRU holds
only a small number of decoded entries, never the complete residue corpus.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data import SPLITS, read_manifest


CACHE_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REACTION_FIELDS = (
    "states", "roles", "atom_mask", "topology_mask", "edge_index", "edge_types"
)


def text_sha256(text: str) -> str:
    """Hash the exact manifest-normalized text passed to an encoder."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(value: Any, name: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    try:
        return json.loads(json.dumps(dict(value), allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON-serializable values") from error


def _encoder_metadata(encoder: Any, name: str) -> dict:
    metadata = getattr(encoder, "metadata", None)
    if callable(metadata):
        metadata = metadata()
    return _json_object(metadata, f"{name} encoder metadata")


def _read_json(path: Path) -> dict:
    def reject_constant(value: str):
        raise ValueError(f"Nonfinite JSON value {value} in {path}")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    with path.open(encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject_constant, object_pairs_hook=unique_keys)
    return _json_object(value, str(path))


def _relative_file(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{name} must be a nonempty relative path")
    resolved = (base / value).resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError as error:
        raise ValueError(f"{name} must stay inside its index directory") from error
    if not resolved.is_file():
        raise ValueError(f"{name} file does not exist: {value}")
    return resolved


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _records(samples: str | Path | Sequence[Mapping]) -> list[dict]:
    if isinstance(samples, (str, Path)):
        result = read_manifest(samples)
    else:
        # Programmatic callers can pass read_manifest output. Recheck the same
        # ID/split/sequence invariants without ever converting a target value.
        result = []
        ids: set[str] = set()
        sequence_splits: dict[str, str] = {}
        group_splits: dict[str, str] = {}
        required = {"sample_id", "sequence", "smiles", "log10_kcat", "split"}
        for raw in samples:
            if not isinstance(raw, Mapping) or not required.issubset(raw):
                raise ValueError(f"Each manifest record requires {sorted(required)}")
            row = {key: value.strip() if isinstance(value, str) else value
                   for key, value in raw.items()}
            for key in ("sample_id", "sequence", "smiles", "split"):
                if not isinstance(row[key], str) or not row[key]:
                    raise ValueError(f"Manifest record has empty or non-string {key}")
            sid = row["sample_id"]
            if sid in ids:
                raise ValueError(f"Duplicate sample_id in manifest: {sid!r}")
            ids.add(sid)
            split = row["split"]
            if split not in SPLITS:
                raise ValueError(f"Sample {sid}: split must be train, val or test")
            sequence = "".join(row["sequence"].split()).upper()
            if not sequence or not sequence.isascii() or not sequence.isalpha():
                raise ValueError(f"Sample {sid}: sequence must contain amino-acid letters")
            row["sequence"] = sequence
            if sequence in sequence_splits and sequence_splits[sequence] != split:
                raise ValueError(f"Sequence leakage across splits at sample {sid!r}")
            sequence_splits[sequence] = split
            group = row.get("group_id", "")
            if group:
                if not isinstance(group, str):
                    raise ValueError(f"Sample {sid}: group_id must be a string")
                if group in group_splits and group_splits[group] != split:
                    raise ValueError(f"Group leakage across splits for group_id={group!r}")
                group_splits[group] = split
            result.append(row)
        if not result:
            raise ValueError("Samples manifest has no data records")
    for row in result:
        reaction = row.get("reaction_smiles")
        if not isinstance(reaction, str) or not reaction.strip():
            raise ValueError(f"Sample {row['sample_id']}: missing reaction_smiles")
        row["reaction_smiles"] = reaction.strip()
    return result


def _float_matrix(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or min(array.shape) < 1 or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a nonempty real matrix [length, dimension]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} overflows float32")
    return array


def _protein_arrays(value: Any, sequence_length: int, dimension: int | None = None) -> dict:
    if isinstance(value, Mapping):
        state = value.get("states", value.get("residue_states"))
    else:
        state = value
    array = _float_matrix(state, "Protein residue states")
    if array.shape[0] != sequence_length:
        raise ValueError(
            "Protein residue states must have one row per sequence residue "
            f"({sequence_length}), got {array.shape[0]}; pooled/global vectors are not residues"
        )
    if dimension is not None and array.shape[1] != dimension:
        raise ValueError(f"Protein dimension {array.shape[1]} does not match {dimension}")
    return {"states": array}


def _integer_array(value: Any, shape: tuple, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer array of shape {shape}")
    return array.astype(np.int64, copy=False)


def _bool_array(value: Any, length: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (length,) or array.dtype.kind != "b":
        raise ValueError(f"{name} must be a boolean array of shape [{length}]")
    return array


def _reaction_arrays(value: Mapping, dimension: int | None = None) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("Reaction encoder must return a dictionary of token arrays")
    states = _float_matrix(value.get("states", value.get("reaction_states")), "Reaction states")
    length = len(states)
    if dimension is not None and states.shape[1] != dimension:
        raise ValueError(f"Reaction dimension {states.shape[1]} does not match {dimension}")
    roles = _integer_array(value.get("roles"), (length,), "Reaction roles")
    if not np.isin(roles, [0, 1, 2, 3]).all():
        raise ValueError("Reaction roles must be 0=reactant, 1=agents, 2=product, 3=separator")
    atom_mask = _bool_array(value.get("atom_mask"), length, "atom_mask")
    topology_mask = _bool_array(value.get("topology_mask"), length, "topology_mask")
    if np.any(topology_mask & ~atom_mask):
        raise ValueError("topology_mask must be a subset of atom_mask")
    if np.any(atom_mask & (roles == 3)):
        raise ValueError("Separator tokens cannot be atoms")
    edge_index = np.asarray(value.get("edge_index"))
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype.kind not in "iu":
        raise ValueError("edge_index must be an integer array [2, edge_count]")
    edge_index = edge_index.astype(np.int64, copy=False)
    edge_types = _integer_array(value.get("edge_types"), (edge_index.shape[1],), "edge_types")
    if not np.isin(edge_types, [1, 2, 3, 4, 5, 6]).all():
        raise ValueError("edge_types must be in 1..6; zero is reserved for absent/padded edges")
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= length):
        raise ValueError("edge_index contains an out-of-range token index")
    if edge_index.size and not topology_mask[edge_index].all():
        raise ValueError("Edges may only connect tokens with known topology")
    if edge_index.size and np.any(roles[edge_index[0]] != roles[edge_index[1]]):
        raise ValueError("Chemical topology edges cannot connect different reaction sides")
    edges = {}
    for (source, target), kind in zip(edge_index.T.tolist(), edge_types.tolist()):
        if (source, target) in edges:
            raise ValueError("Duplicate directed edge in reaction topology")
        if (source == target) != (kind == 1):
            raise ValueError("Self edges must have type 1; type 1 cannot label a chemical bond")
        edges[source, target] = kind
    for node in np.flatnonzero(topology_mask):
        if edges.get((int(node), int(node))) != 1:
            raise ValueError("Each topology token requires a type-1 self edge")
    for (source, target), kind in edges.items():
        if edges.get((target, source)) != kind:
            raise ValueError("Chemical bonds require matching directed reverse edges")
    return {
        "states": states, "roles": roles, "atom_mask": atom_mask,
        "topology_mask": topology_mask, "edge_index": edge_index, "edge_types": edge_types,
    }


def _load_npz(path: Path, *, array_key: str | None = None) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as archive:
        keys = [array_key] if array_key is not None else [
            key for key in (*_REACTION_FIELDS, "residue_states", "reaction_states")
            if key in archive.files
        ]
        if array_key is not None and array_key not in archive.files:
            raise ValueError(f"Imported NPZ {path.name} has no array {array_key!r}")
        arrays = {key: np.array(archive[key], copy=True) for key in keys}
        if array_key is not None:
            arrays = {"states": arrays[array_key]}
        metadata = {}
        if "metadata" in archive.files:
            raw = archive["metadata"]
            if raw.ndim != 0 or raw.dtype.kind != "U":
                raise ValueError("NPZ metadata must be a scalar Unicode JSON string")
            metadata = _json_object(json.loads(str(raw.item())), "NPZ metadata")
    return arrays, metadata


def _save_npz(path: Path, arrays: dict, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(
                handle, **arrays,
                metadata=np.asarray(json.dumps(metadata, allow_nan=False, sort_keys=True)),
            )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _save_json(path: Path, value: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, allow_nan=False, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _protein_imports(path: str | Path | None) -> tuple[dict, dict]:
    """Read an explicit per-sequence index, never infer residue ordering.

    Import schema: ``{version: 1, encoders: {protein: {...}}, sequences:
    {sequence_sha256: {path: 'relative.npz', array_key: 'states', sha256: ...}}}``.
    ``sha256`` is optional for an external import, and verified when supplied.
    Standard interaction-cache ``samples`` indices can also supply proteins.
    Every imported matrix must still have exactly len(sequence) residue rows.
    """
    if path is None:
        return {}, {}
    index_path = Path(path)
    if index_path.is_dir():
        index_path = index_path / "index.json"
    document = _read_json(index_path)
    if document.get("version") != CACHE_VERSION:
        raise ValueError("Protein import index must have version 1")
    metadata = _json_object(document.get("encoders", {}).get("protein"), "Imported protein encoder")
    entries = {}
    if "sequences" in document:
        source = document["sequences"]
        if not isinstance(source, dict):
            raise ValueError("Import sequences must be an object keyed by sequence SHA256")
        rows = [(sha, entry) for sha, entry in source.items()]
    elif isinstance(document.get("samples"), dict):
        rows = [(entry.get("sequence_sha256"), {
            "path": entry.get("protein"), "array_key": "states",
            "sha256": entry.get("protein_file_sha256"),
        }) for entry in document["samples"].values()]
    else:
        raise ValueError("Protein import index requires sequences or samples")
    for sha, entry in rows:
        _require_sha(sha, "Imported sequence_sha256")
        if not isinstance(entry, dict):
            raise ValueError("Each protein import entry must be an object")
        if "sequence_sha256" in entry and entry["sequence_sha256"] != sha:
            raise ValueError("Protein import key and sequence_sha256 disagree")
        file = _relative_file(index_path.parent, entry.get("path"), "Imported protein")
        array_key = entry.get("array_key", "states")
        if not isinstance(array_key, str) or not array_key:
            raise ValueError("Imported protein array_key must be a nonempty string")
        checksum = entry.get("sha256")
        if checksum is not None:
            _require_sha(checksum, "Imported protein sha256")
        candidate = {"path": file, "array_key": array_key, "sha256": checksum}
        if sha in entries and entries[sha] != candidate:
            raise ValueError("Conflicting import paths for the same sequence")
        entries[sha] = candidate
    return entries, metadata


def create_cache(
    samples: str | Path | Sequence[Mapping],
    output: str | Path,
    protein_encoder: Any,
    reaction_encoder: Any,
    *,
    protein_import_index: str | Path | None = None,
) -> dict:
    """Encode every manifest row, deduplicating by exact sequence/reaction text.

    Encoders expose ``metadata`` (a dict/property or method) and ``encode(text)``.
    Protein results may be a matrix or a dict with ``residue_states``/``states``;
    reactions use ``reaction_states``/``states`` and the topology arrays. A
    matching existing NPZ is reused only after input/provenance validation.
    Failures raise with the sample ID; no row removal, random feature, or pooled
    residue fallback is used. ``index.json`` is published atomically on success.

    To import every protein, pass ``protein_encoder=None`` and an explicit
    import index. Partial imports require an encoder with identical provenance.
    """
    records = _records(samples)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    imports, imported_metadata = _protein_imports(protein_import_index)
    protein_metadata = (
        _encoder_metadata(protein_encoder, "Protein")
        if protein_encoder is not None else imported_metadata
    )
    if protein_encoder is None and not imports:
        raise ValueError("Provide a protein encoder or an explicit per-sequence import index")
    if imports and protein_encoder is not None and imported_metadata != protein_metadata:
        raise ValueError("Imported protein provenance must match the protein encoder metadata")
    reaction_metadata = _encoder_metadata(reaction_encoder, "Reaction")
    provenance = {"protein": protein_metadata, "reaction": reaction_metadata}
    old_index = output / "index.json"
    previous_checksums: dict[str, str] = {}
    if old_index.exists():
        previous = _read_json(old_index)
        if previous.get("version") != CACHE_VERSION or previous.get("encoders") != provenance:
            raise ValueError("Existing cache uses different encoders/version; use another directory")
        for old_entry in previous.get("samples", {}).values():
            for kind in ("protein", "reaction"):
                relative = old_entry.get(kind)
                checksum = _require_sha(old_entry.get(f"{kind}_file_sha256"), "Existing cache checksum")
                if relative in previous_checksums and previous_checksums[relative] != checksum:
                    raise ValueError("Existing cache has conflicting checksums for one file")
                previous_checksums[relative] = checksum
    dimensions: dict[str, int] = {}
    seen: dict[str, dict[str, dict]] = {"protein": {}, "reaction": {}}
    result = {"version": CACHE_VERSION, "encoders": provenance, "dimensions": dimensions,
              "samples": {}}
    for row in records:
        sid = row["sample_id"]
        entry = {}
        for kind, text_key, encoder in (
            ("protein", "sequence", protein_encoder),
            ("reaction", "reaction_smiles", reaction_encoder),
        ):
            text = row[text_key]
            sha = text_sha256(text)
            entry["sequence_sha256" if kind == "protein" else "reaction_sha256"] = sha
            try:
                if sha not in seen[kind]:
                    relative = f"{'proteins' if kind == 'protein' else 'reactions'}/{sha}.npz"
                    destination = output / relative
                    record_metadata = {"input_sha256": sha, "encoder": provenance[kind]}
                    if destination.exists():
                        if relative in previous_checksums and file_sha256(destination) != previous_checksums[relative]:
                            raise ValueError("Existing cache file checksum mismatch")
                        raw_arrays, saved_metadata = _load_npz(destination)
                        if any(saved_metadata.get(key) != value
                               for key, value in record_metadata.items()):
                            raise ValueError("Existing NPZ input/encoder provenance mismatch")
                        encode_metadata = saved_metadata.get("encoding", {})
                    elif kind == "protein" and sha in imports:
                        source = imports[sha]
                        checksum = file_sha256(source["path"])
                        if source["sha256"] is not None and checksum != source["sha256"]:
                            raise ValueError("Imported protein file checksum mismatch")
                        raw_arrays, source_metadata = _load_npz(
                            source["path"], array_key=source["array_key"]
                        )
                        if source_metadata.get("input_sha256", sha) != sha:
                            raise ValueError("Imported NPZ sequence identity mismatch")
                        encode_metadata = {"import_sha256": checksum,
                                           "import_metadata": source_metadata}
                    else:
                        if encoder is None:
                            raise ValueError("No imported residue matrix for this sequence")
                        raw_arrays = encoder.encode(text)
                        encode_metadata = (
                            raw_arrays.get("metadata", {}) if isinstance(raw_arrays, Mapping) else {}
                        )
                    arrays = (
                        _protein_arrays(raw_arrays, len(text), dimensions.get(kind))
                        if kind == "protein" else _reaction_arrays(raw_arrays, dimensions.get(kind))
                    )
                    dimensions.setdefault(kind, int(arrays["states"].shape[1]))
                    record_metadata["encoding"] = _json_object(encode_metadata, "Encoding metadata")
                    reported_sha = record_metadata["encoding"].get("raw_text_sha256", sha)
                    if reported_sha != sha:
                        raise ValueError("Encoder output metadata does not match the input text")
                    if not destination.exists():
                        _save_npz(destination, arrays, record_metadata)
                    seen[kind][sha] = {"path": relative, "sha256": file_sha256(destination),
                                       "length": int(len(arrays["states"]))}
                cached = seen[kind][sha]
                entry[kind] = cached["path"]
                entry[f"{kind}_file_sha256"] = cached["sha256"]
                entry[f"{kind}_length"] = cached["length"]
            except (ValueError, TypeError, OSError, KeyError) as error:
                raise ValueError(f"Sample {sid}: cannot cache {kind}: {error}") from error
        result["samples"][sid] = entry
    result["sample_count"] = len(records)
    result["unique_counts"] = {kind: len(entries) for kind, entries in seen.items()}
    _save_json(old_index, result)
    return result


def _global_features(path: str | Path, sample_ids: list[str], dimension: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if not {"sample_ids", "protein"}.issubset(archive.files):
            raise ValueError("Global feature NPZ requires sample_ids and protein")
        ids = archive["sample_ids"]
        if ids.ndim != 1 or ids.dtype.kind != "U":
            raise ValueError("Global feature sample_ids must be a 1D Unicode array")
        names = ids.tolist()
        if any(not value or value != value.strip() for value in names):
            raise ValueError("Global sample_ids must be nonempty and have no surrounding whitespace")
        if len(set(names)) != len(names):
            raise ValueError("Duplicate sample_ids in global features")
        lookup = {value: index for index, value in enumerate(names)}
        missing = [sid for sid in sample_ids if sid not in lookup]
        if missing:
            raise ValueError(f"Selected samples missing global features: {missing[:5]}")
        array = archive["protein"]
        if array.shape != (len(names), dimension):
            raise ValueError(f"Global protein features must have shape [{len(names)}, {dimension}]")
        selected = array[[lookup[sid] for sid in sample_ids]]
        return _float_matrix(selected, "Selected global protein features")


class InteractionDataset(Dataset):
    """Load selected samples lazily without loading other splits' labels/states."""

    def __init__(
        self,
        samples_path: str | Path,
        cache_path: str | Path,
        splits: Sequence[str] | set[str] | None = ("train",),
        global_features: str | Path | None = None,
        require_targets: bool = True,
        cache_size: int = 8,
    ):
        if type(require_targets) is not bool:
            raise ValueError("require_targets must be a boolean")
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size must be a nonnegative integer")
        if splits is not None and (
            isinstance(splits, str) or not splits or not set(splits).issubset(SPLITS)
        ):
            raise ValueError("splits must be a nonempty collection of train/val/test, or None")
        selected = SPLITS if splits is None else set(splits)
        self.all_records = _records(samples_path)
        self.records = [dict(row) for row in self.all_records if row["split"] in selected]
        if not self.records:
            raise ValueError("No manifest records in the selected splits")
        self.sample_ids = [row["sample_id"] for row in self.records]
        index_path = Path(cache_path)
        if index_path.is_dir():
            index_path = index_path / "index.json"
        self.cache_root = index_path.parent.resolve()
        self.index = _read_json(index_path)
        if self.index.get("version") != CACHE_VERSION or not isinstance(self.index.get("samples"), dict):
            raise ValueError("Interaction cache requires version 1 and a samples object")
        self.encoders = _json_object(self.index.get("encoders"), "Cache encoders")
        for kind in ("protein", "reaction"):
            _json_object(self.encoders.get(kind), f"Cached {kind} provenance")
        self.dimensions = _json_object(self.index.get("dimensions", {}), "Cache dimensions")
        if any(type(self.dimensions.get(kind)) is not int or self.dimensions[kind] < 1
               for kind in ("protein", "reaction")):
            raise ValueError("Cache dimensions must contain positive protein and reaction dimensions")
        self.protein_dim = self.dimensions["protein"]
        self.reaction_dim = self.dimensions["reaction"]
        self.cache_size = cache_size
        self._lru: OrderedDict[tuple[str, str, str], dict] = OrderedDict()
        self._entries = []
        self.targets = []
        for row in self.records:
            sid = row["sample_id"]
            entry = self.index["samples"].get(sid)
            if not isinstance(entry, dict):
                raise ValueError(f"Selected sample {sid!r} missing from interaction cache")
            entry = dict(entry)
            for kind, text_key, sha_key in (
                ("protein", "sequence", "sequence_sha256"),
                ("reaction", "reaction_smiles", "reaction_sha256"),
            ):
                expected = text_sha256(row[text_key])
                if entry.get(sha_key) != expected:
                    raise ValueError(f"Sample {sid}: cached {text_key} does not match manifest")
                _require_sha(entry.get(f"{kind}_file_sha256"), f"Sample {sid} {kind}_file_sha256")
                entry[f"_{kind}_path"] = _relative_file(self.cache_root, entry.get(kind), kind)
            self._entries.append(entry)
            # Inference never needs to parse a label, even if a CSV retains one.
            target = None
            if require_targets:
                try:
                    target = float(row["log10_kcat"])
                except (ValueError, TypeError) as error:
                    raise ValueError(f"Sample {sid}: missing or invalid log10_kcat target") from error
                if not math.isfinite(target) or abs(target) > np.finfo(np.float32).max:
                    raise ValueError(f"Sample {sid}: log10_kcat must be finite float32")
            self.targets.append(target)
        self.protein_global_source = "external_id_join" if global_features is not None else "mean_all_residues"
        self.global_features = (
            _global_features(global_features, self.sample_ids, self.protein_dim)
            if global_features is not None else None
        )

    def __len__(self) -> int:
        return len(self.records)

    def _load(self, index: int, kind: str) -> dict:
        entry = self._entries[index]
        row = self.records[index]
        key = (kind, entry[kind], entry[f"{kind}_file_sha256"])
        if key in self._lru:
            self._lru.move_to_end(key)
            return self._lru[key]
        path = entry[f"_{kind}_path"]
        if file_sha256(path) != entry[f"{kind}_file_sha256"]:
            raise ValueError(f"Sample {row['sample_id']}: {kind} cache file checksum mismatch")
        raw, metadata = _load_npz(path)
        sha_key = "sequence_sha256" if kind == "protein" else "reaction_sha256"
        if metadata.get("input_sha256") != entry[sha_key] or metadata.get("encoder") != self.encoders[kind]:
            raise ValueError(f"Sample {row['sample_id']}: {kind} cache provenance mismatch")
        arrays = (
            _protein_arrays(raw, len(row["sequence"]), self.protein_dim)
            if kind == "protein" else _reaction_arrays(raw, self.reaction_dim)
        )
        for value in arrays.values():
            value.flags.writeable = False
        if self.cache_size:
            self._lru[key] = arrays
            while len(self._lru) > self.cache_size:
                self._lru.popitem(last=False)
        return arrays

    def __getitem__(self, index: int) -> dict:
        protein = self._load(index, "protein")
        reaction = self._load(index, "reaction")
        global_vector = (
            self.global_features[index] if self.global_features is not None
            else protein["states"].mean(axis=0, dtype=np.float64).astype(np.float32)
        )
        return {
            "sample_id": self.sample_ids[index], "target": self.targets[index],
            "protein_states": protein["states"], "protein_global": global_vector,
            "reaction_states": reaction["states"], "roles": reaction["roles"],
            "atom_mask": reaction["atom_mask"], "topology_mask": reaction["topology_mask"],
            "edge_index": reaction["edge_index"], "edge_types": reaction["edge_types"],
        }


def collate_interactions(items: list[dict]) -> dict:
    """Pad sequences with explicit masks and materialize per-batch bond types."""
    if not items:
        raise ValueError("Cannot collate an empty interaction batch")
    batch = len(items)
    protein_dim = items[0]["protein_states"].shape[1]
    reaction_dim = items[0]["reaction_states"].shape[1]
    protein_width = max(len(item["protein_states"]) for item in items)
    reaction_width = max(len(item["reaction_states"]) for item in items)
    protein_states = np.zeros((batch, protein_width, protein_dim), dtype=np.float32)
    reaction_states = np.zeros((batch, reaction_width, reaction_dim), dtype=np.float32)
    protein_mask = np.zeros((batch, protein_width), dtype=bool)
    reaction_mask = np.zeros((batch, reaction_width), dtype=bool)
    atom_mask = np.zeros((batch, reaction_width), dtype=bool)
    topology_mask = np.zeros((batch, reaction_width), dtype=bool)
    roles = np.full((batch, reaction_width), 3, dtype=np.int64)
    bond_types = np.zeros((batch, reaction_width, reaction_width), dtype=np.int64)
    globals_ = np.zeros((batch, protein_dim), dtype=np.float32)
    for index, item in enumerate(items):
        protein = _protein_arrays(item["protein_states"], len(item["protein_states"]), protein_dim)["states"]
        reaction = _reaction_arrays(item, reaction_dim)
        global_vector = np.asarray(item["protein_global"])
        if global_vector.shape != (protein_dim,) or global_vector.dtype.kind not in "fiu":
            raise ValueError("Global protein vector must match the residue feature dimension")
        if not np.isfinite(global_vector).all():
            raise ValueError("Global protein vector contains nonfinite values")
        n, t = len(protein), len(reaction["states"])
        protein_states[index, :n] = protein
        reaction_states[index, :t] = reaction["states"]
        protein_mask[index, :n] = True
        reaction_mask[index, :t] = True
        globals_[index] = global_vector
        roles[index, :t] = reaction["roles"]
        atom_mask[index, :t] = reaction["atom_mask"]
        topology_mask[index, :t] = reaction["topology_mask"]
        edge = reaction["edge_index"]
        bond_types[index, edge[0], edge[1]] = reaction["edge_types"]
    targets = [item.get("target") for item in items]
    if any(value is None for value in targets) and not all(value is None for value in targets):
        raise ValueError("A batch cannot mix labeled and unlabeled samples")
    arrays = {
        "protein_states": protein_states, "protein_mask": protein_mask,
        "protein_global": globals_, "reaction_states": reaction_states,
        "reaction_mask": reaction_mask, "roles": roles, "atom_mask": atom_mask,
        "topology_mask": topology_mask, "bond_types": bond_types,
    }
    if not np.isfinite(globals_).all():
        raise ValueError("Global protein vector overflows float32")
    result = {key: torch.from_numpy(value) for key, value in arrays.items()}
    result["targets"] = None if targets[0] is None else torch.tensor(targets, dtype=torch.float32)
    if result["targets"] is not None and not torch.isfinite(result["targets"]).all():
        raise ValueError("Batch targets must be finite float32")
    result["sample_ids"] = [item["sample_id"] for item in items]
    return result
