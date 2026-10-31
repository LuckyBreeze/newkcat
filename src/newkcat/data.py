"""ID-joined cached features, conservative split checks, and padded graph batches.

Only selected rows are chemically featurized or have their targets/features
validated. Full-manifest identifiers and split metadata are still checked for
leakage. Unknown centers are distinct from an observed all-zero center vector.
"""

import csv
import math
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .centers import load_center_records, validate_center_record
from .chem import ATOM_DIM, BOND_DIM, featurize_smiles
from .types import GraphBatch


FEATURE_DIMS = {"protein": 1024, "molt5": 768, "maccs": 167}
SPLITS = {"train", "val", "test"}


def ec2_token(value: str | None) -> str | None:
    """Return one unambiguous two-level EC code, or unknown.

    Multiple EC annotations are not arbitrarily reduced to the first code.
    Partial codes such as ``1.1.-.-`` retain their known first two levels.
    EC annotations are optional; missing/ambiguous values never become a class.
    """
    text = (value or "").strip()
    if text.lower().startswith("ec:"):
        text = text[3:].strip()
    elif text.lower().startswith("ec "):
        text = text[3:].strip()
    if not re.fullmatch(r"\d+\.\d+(?:\.(?:\d+|-)){0,2}", text):
        return None
    parts = text.split(".")
    if any(int(level) <= 0 for level in parts[:2]):
        return None
    return ".".join(str(int(level)) for level in parts[:2])


def fit_ec_vocab(records: list[dict]) -> dict[str, int]:
    """Fit categorical IDs on selected training records only; 0 stays unknown."""
    codes = {
        ec2_token(row.get("ec")) for row in records if row.get("split") == "train"
    }
    return {code: i + 1 for i, code in enumerate(sorted(codes - {None}))}


def _validate_ec_vocab(vocabulary: dict[str, int]) -> dict[str, int]:
    if not isinstance(vocabulary, dict):
        raise ValueError("ec_vocab must be a dictionary fitted on training data")
    for key, value in vocabulary.items():
        if not isinstance(key, str) or ec2_token(key) != key:
            raise ValueError("ec_vocab keys must be normalized two-level EC codes")
        if type(value) is not int or value < 1:
            raise ValueError("ec_vocab IDs must be positive integers; 0 is reserved for unknown")
    if set(vocabulary.values()) != set(range(1, len(vocabulary) + 1)):
        raise ValueError("ec_vocab IDs must be unique and contiguous starting at 1")
    return dict(vocabulary)


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    """Validate IDs/split metadata without parsing any target or molecular graph."""
    required = {"sample_id", "sequence", "smiles", "log10_kcat", "split"}
    records = []
    id_seen = set()
    sequence_split: dict[str, str] = {}
    group_split: dict[str, str] = {}
    pair_split: dict[tuple[str, str], str] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if not required.issubset(headers):
            raise ValueError(f"Samples CSV must contain columns {sorted(required)}")
        if len(set(headers)) != len(headers):
            raise ValueError("Samples CSV has duplicate column names")
        for line, raw in enumerate(reader, 2):
            if None in raw or any(raw.get(key) is None for key in required):
                raise ValueError(f"Samples CSV line {line} has an inconsistent column count")
            row = {key: value.strip() if value is not None else "" for key, value in raw.items()}
            for key in ("sample_id", "sequence", "smiles", "split"):
                if not row[key]:
                    raise ValueError(f"Samples CSV line {line} has empty {key}")
            sample_id = row["sample_id"]
            if sample_id in id_seen:
                raise ValueError(f"Duplicate sample_id in manifest: {sample_id!r}")
            id_seen.add(sample_id)
            split = row["split"]
            if split not in SPLITS:
                raise ValueError(f"Sample {sample_id}: split must be train, val or test")
            sequence = "".join(row["sequence"].split()).upper()
            if not sequence or not sequence.isascii() or not sequence.isalpha():
                raise ValueError(f"Sample {sample_id}: sequence must contain amino-acid letters")
            row["sequence"] = sequence
            # Exact sequence disjointness also rules out cross-split copies of
            # the same enzyme-substrate pair, including alternate SMILES spellings.
            if sequence in sequence_split and sequence_split[sequence] != split:
                raise ValueError(f"Sequence leakage across splits at sample {sample_id!r}")
            sequence_split[sequence] = split
            pair = (sequence, row["smiles"])
            if pair in pair_split and pair_split[pair] != split:
                raise ValueError(f"Sequence-SMILES pair leakage at sample {sample_id!r}")
            pair_split[pair] = split
            group = row.get("group_id", "")
            if group:
                if group in group_split and group_split[group] != split:
                    raise ValueError(f"Group leakage across splits for group_id={group!r}")
                group_split[group] = split
            records.append(row)
    if not records:
        raise ValueError("Samples CSV has no data records")
    return records


def _selected_features(path: str | Path, sample_ids: list[str]) -> dict[str, np.ndarray]:
    """Validate the global cache schema, then read/validate only requested rows."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"sample_ids", *FEATURE_DIMS}
        if not required.issubset(archive.files):
            raise ValueError(f"Feature NPZ needs arrays {sorted(required)}")
        ids = archive["sample_ids"]
        if ids.ndim != 1 or ids.dtype.kind != "U":
            raise ValueError("Feature sample_ids must be a 1D Unicode array, never object/pickle")
        id_list = ids.tolist()
        if any(not value or value != value.strip() for value in id_list):
            raise ValueError("Feature sample_ids must be nonempty with no surrounding whitespace")
        if len(set(id_list)) != len(id_list):
            raise ValueError("Duplicate sample_ids in feature cache")
        lookup = {value: i for i, value in enumerate(id_list)}
        missing = [value for value in sample_ids if value not in lookup]
        if missing:
            raise ValueError(f"Selected samples missing feature rows: {missing[:5]}")
        indices = np.asarray([lookup[value] for value in sample_ids], dtype=np.int64)
        output = {}
        for name, dimension in FEATURE_DIMS.items():
            array = archive[name]
            if array.shape != (len(ids), dimension):
                raise ValueError(
                    f"Feature {name} must have shape [{len(ids)}, {dimension}], got {array.shape}"
                )
            if array.dtype.kind not in "fiu":
                raise ValueError(f"Feature {name} must be a real numeric array")
            selected = array[indices]
            if not np.isfinite(selected).all():
                raise ValueError(f"Selected {name} features contain nonfinite values")
            if name == "maccs" and (
                not np.isin(selected, [0, 1]).all() or np.any(selected[:, 0] != 0)
            ):
                raise ValueError("MACCS must contain 167 binary RDKit bits with unused bit 0 = 0")
            with np.errstate(over="ignore", invalid="ignore"):
                converted = selected.astype(np.float32, copy=True)
            if not np.isfinite(converted).all():
                raise ValueError(f"Selected {name} features overflow float32")
            output[name] = converted
    return output


class KcatDataset(Dataset):
    """Samples whose global features are externally prepared and ID-aligned.

    Centers determine the graph's atom order through their own mapped SMILES;
    map-keyed scores therefore never need a guessed isomorphism permutation.
    ``ec_vocab=None`` fits only selected train rows. Pass that vocabulary to
    validation/inference datasets; unseen EC codes become 0.
    """

    atom_dim = ATOM_DIM
    bond_dim = BOND_DIM

    def __init__(
        self,
        samples_path: str | Path,
        features_path: str | Path,
        centers_path: str | Path | None = None,
        splits: list[str] | tuple[str, ...] | set[str] | None = None,
        ec_vocab: dict[str, int] | None = None,
        require_targets: bool = True,
    ):
        if type(require_targets) is not bool:
            raise ValueError("require_targets must be a boolean")
        if splits is not None and (
            isinstance(splits, str) or not splits or not set(splits).issubset(SPLITS)
        ):
            raise ValueError("splits must be a nonempty collection of train/val/test, or None")
        selected_splits = SPLITS if splits is None else set(splits)
        self.all_records = read_manifest(samples_path)
        self.records = [dict(row) for row in self.all_records if row["split"] in selected_splits]
        self.sample_ids = [row["sample_id"] for row in self.records]
        self.ec_vocab = (
            fit_ec_vocab(self.records) if ec_vocab is None else _validate_ec_vocab(ec_vocab)
        )
        self.center_uses_ec = False
        self.has_reference_centers = False
        features = _selected_features(features_path, self.sample_ids)
        centers = load_center_records(centers_path)
        self.items = []
        for index, row in enumerate(self.records):
            sample_id = row["sample_id"]
            raw_target = row["log10_kcat"]
            if raw_target == "":
                if require_targets:
                    raise ValueError(f"Sample {sample_id}: missing log10_kcat target")
                target = None
            else:
                try:
                    target = float(raw_target)
                except ValueError as error:
                    raise ValueError(f"Sample {sample_id}: invalid log10_kcat target") from error
                if not math.isfinite(target) or abs(target) > np.finfo(np.float32).max:
                    raise ValueError(f"Sample {sample_id}: log10_kcat must be finite float32")
            row["log10_kcat"] = target
            try:
                if sample_id in centers:
                    annotation = validate_center_record(centers[sample_id], row["smiles"])
                    graph = featurize_smiles(annotation.mapped_smiles)
                    scores = annotation.scores
                    center_kind = annotation.kind
                    self.center_uses_ec |= annotation.uses_ec
                    self.has_reference_centers |= center_kind == "reference"
                    available = True
                    center_uses_ec = annotation.uses_ec
                    center_source = annotation.source
                else:
                    graph = featurize_smiles(row["smiles"])
                    scores = np.zeros(graph.num_atoms, dtype=np.float32)
                    center_kind = "missing"
                    available = False
                    center_uses_ec = False
                    center_source = "missing"
            except ValueError as error:
                raise ValueError(f"Sample {sample_id}: {error}") from error
            self.items.append({
                "graph": graph,
                "center_scores": scores,
                "center_available": available,
                "protein": features["protein"][index],
                "molt5": features["molt5"][index],
                "maccs": features["maccs"][index],
                "ec2": self.ec_vocab.get(ec2_token(row.get("ec")), 0),
                "target": target,
                "sample_id": sample_id,
                "center_kind": center_kind,
                "center_uses_ec": center_uses_ec,
                "center_source": center_source,
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def collate_graphs(items: list[dict]) -> GraphBatch:
    """Pad nodes/distances and offset directed edges by the *padded* graph width."""
    if not items:
        raise ValueError("Cannot collate an empty graph batch")
    batch_size = len(items)
    width = max(item["graph"].num_atoms for item in items)
    atoms = np.zeros((batch_size, width, ATOM_DIM), dtype=np.float32)
    distances = np.full((batch_size, width, width), -1, dtype=np.int64)
    mask = np.zeros((batch_size, width), dtype=bool)
    scores = np.zeros((batch_size, width), dtype=np.float32)
    edges = []
    edge_features = []
    for i, item in enumerate(items):
        graph = item["graph"]
        n = graph.num_atoms
        atoms[i, :n] = graph.atom_features
        distances[i, :n, :n] = graph.distances
        mask[i, :n] = True
        scores[i, :n] = item["center_scores"]
        edges.append(graph.edge_index + i * width)
        edge_features.append(graph.edge_features)
    targets = [item["target"] for item in items]
    return GraphBatch(
        atom_features=torch.from_numpy(atoms),
        edge_index=torch.from_numpy(np.concatenate(edges, axis=1)),
        edge_features=torch.from_numpy(np.concatenate(edge_features, axis=0)),
        distances=torch.from_numpy(distances),
        atom_mask=torch.from_numpy(mask),
        center_scores=torch.from_numpy(scores),
        center_available=torch.tensor([item["center_available"] for item in items], dtype=torch.bool),
        protein=torch.from_numpy(np.stack([item["protein"] for item in items])),
        molt5=torch.from_numpy(np.stack([item["molt5"] for item in items])),
        maccs=torch.from_numpy(np.stack([item["maccs"] for item in items])),
        ec2=torch.tensor([item["ec2"] for item in items], dtype=torch.long),
        target=None if any(value is None for value in targets) else torch.tensor(
            targets, dtype=torch.float32
        ),
        sample_ids=tuple(item["sample_id"] for item in items),
        center_kinds=tuple(item["center_kind"] for item in items),
    )
