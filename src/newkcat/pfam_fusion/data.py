"""Pfam hit joins, train-only vocabulary, padding, and CataPro cache alignment."""

from __future__ import annotations

import csv
import gzip
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

PFAM_PATTERN = re.compile(r"^PF\d{5,}(?:\.\d+)?$", re.IGNORECASE)
PAD, UNK = "<PAD>", "<UNK>"


def canonical_pfam(value: str) -> str:
    value = value.strip().upper()
    if not PFAM_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid Pfam accession: {value!r}")
    return value.split(".", 1)[0]


def read_split(path: str | Path, split: str) -> list[dict]:
    """Read one original CSV split without changing labels or row order."""
    required = {"row_uid", "sequence", "log10kcat_max", "split"}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} missing columns: {sorted(required - set(reader.fieldnames or []))}")
        if len(set(reader.fieldnames or [])) != len(reader.fieldnames or []):
            raise ValueError(f"Duplicate CSV columns in {path}")
        rows = []
        seen = set()
        for line_no, row in enumerate(reader, 2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"Malformed CSV row {line_no} in {path}")
            uid = row["row_uid"].strip()
            if not uid or uid in seen:
                raise ValueError(f"Empty or duplicate row_uid {uid!r} in {path}")
            seen.add(uid)
            given = row["split"].strip().lower()
            allowed = {"valid", "val"} if split == "valid" else {split}
            if given not in allowed:
                raise ValueError(f"row_uid={uid}: expected split={split}, got {given}")
            if not row["sequence"].strip():
                raise ValueError(f"row_uid={uid}: empty sequence")
            try:
                target = float(row["log10kcat_max"])
            except ValueError as exc:
                raise ValueError(f"row_uid={uid}: invalid log10kcat_max") from exc
            if not math.isfinite(target):
                raise ValueError(f"row_uid={uid}: nonfinite target")
            rows.append({"row_uid": uid, "sequence": row["sequence"].strip().upper(),
                         "target": target, "split": split,
                         "cluster_id": row.get("cluster_id", "").strip()})
    if not rows:
        raise ValueError(f"Empty {split} split: {path}")
    return rows


def check_disjoint(*splits: list[dict]) -> None:
    """Reject cross-split reused IDs, sequences or cluster identifiers."""
    for field in ("row_uid", "sequence", "cluster_id"):
        seen = {}
        for i, rows in enumerate(splits):
            for row in rows:
                value = row.get(field)
                if not value:
                    continue
                if value in seen and seen[value] != i:
                    raise ValueError(f"Cross-split {field} leakage: {str(value)[:60]}")
                seen[value] = i


def read_hits(path: str | Path) -> dict[str, list[tuple[str, int | None, int | None]]]:
    """Read Pfam annotations: row_uid,pfam_id,start,end; 1-based coordinates.

    Do not infer Pfam IDs from enzyme class, UniProt, or kcat. Exact duplicate
    annotation rows are removed; repeated domains at different positions stay.
    """
    result = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or [])
        if not {"row_uid", "pfam_id"}.issubset(headers):
            raise ValueError("Pfam hit CSV requires row_uid,pfam_id (start,end for ordered mode)")
        if ("start" in headers) != ("end" in headers):
            raise ValueError("Pfam hit CSV must have both start and end or neither")
        seen = set()
        for line_no, row in enumerate(reader, 2):
            uid = (row.get("row_uid") or "").strip()
            raw = (row.get("pfam_id") or "").strip()
            if not uid:
                raise ValueError(f"Pfam hit line {line_no} has blank row_uid")
            if not raw:
                continue
            pfam_id = canonical_pfam(raw)
            start = end = None
            if "start" in headers:
                ss, ee = (row.get("start") or "").strip(), (row.get("end") or "").strip()
                if ss and ee:
                    try:
                        start, end = int(ss), int(ee)
                    except ValueError as exc:
                        raise ValueError(f"Bad Pfam coordinates at line {line_no}") from exc
                    if start <= 0 or end < start:
                        raise ValueError(f"Invalid Pfam start/end at line {line_no}")
                elif ss or ee:
                    raise ValueError(f"Partial Pfam coordinate at line {line_no}")
            hit = (pfam_id, start, end)
            if (uid, hit) not in seen:
                result[uid].append(hit)
                seen.add((uid, hit))
    return dict(result)


def fit_vocab(train: list[dict], hits: dict) -> dict[str, int]:
    train_ids = {r["row_uid"] for r in train}
    terms = sorted({hit[0] for uid in train_ids for hit in hits.get(uid, ())})
    if not terms:
        raise ValueError("No train Pfam hits matched row_uid; check Pfam annotation join")
    return {token: i for i, token in enumerate((PAD, UNK, *terms))}


def choose_nmax(train: list[dict], hits: dict, quantile: float = 0.99,
                cap: int = 32) -> int:
    """Count un-deduplicated train Pfam occurrences; NEVER examine test rows."""
    if not (0 < quantile <= 1) or cap < 1:
        raise ValueError("quantile must be in (0,1], cap >=1")
    counts = sorted(len(hits.get(r["row_uid"], ())) for r in train)
    index = min(len(counts) - 1, math.ceil(quantile * len(counts)) - 1)
    return max(1, min(cap, counts[index]))


def ordered_tokens(uid: str, hits: dict, ordered: bool) -> list[str]:
    found = hits.get(uid, [])
    if not ordered:
        return sorted({hit[0] for hit in found})
    if any(start is None or end is None for _, start, end in found):
        raise ValueError(f"row_uid={uid}: ordered mode requires start/end coordinates")
    return [p for p, _, _ in sorted(found, key=lambda x: (x[1], x[2], x[0]))]


def encode(uid: str, hits: dict, vocab: dict, nmax: int, ordered: bool) -> np.ndarray:
    out = np.zeros(nmax, dtype=np.int64)
    tokens = ordered_tokens(uid, hits, ordered)[:nmax]
    out[:len(tokens)] = [vocab.get(token, 1) for token in tokens]
    return out


def audit(rows: list[dict], hits: dict, vocab: dict, nmax: int,
          ordered: bool, vector_ids: set[str] | None = None) -> dict:
    total, matched, raw, raw_known, truncated, unk, encoded_tokens, pretrained = (
        len(rows), 0, 0, 0, 0, 0, 0, 0
    )
    for row in rows:
        uid = row["row_uid"]
        actual = hits.get(uid, ())
        matched += bool(actual)
        raw += len(actual)
        raw_known += sum(h[0] in vocab for h in actual)
        tokens = ordered_tokens(uid, hits, ordered)
        truncated += max(0, len(tokens) - nmax)
        remaining = tokens[:nmax]
        encoded_tokens += len(remaining)
        unk += sum(t not in vocab for t in remaining)
        if vector_ids is not None:
            pretrained += sum(t in vector_ids for t in remaining)
    result = {"n_samples": total, "samples_with_pfam": matched,
              "annotation_hit_fraction": matched / total, "raw_pfam_occurrences": raw,
              "raw_train_vocab_coverage": raw_known / raw if raw else None,
              "encoded_tokens": encoded_tokens, "encoded_unknown_tokens": unk,
              "truncated_tokens": truncated, "nmax": nmax}
    if vector_ids is not None:
        result["deepbgc_coverage_on_encoded_tokens"] = (
            pretrained / encoded_tokens if encoded_tokens else None
        )
    return result


def read_deepbgc(path: str | Path) -> dict[str, np.ndarray]:
    """Read Merck DeepBGC v0.1.0 pfam2vec.csv(.gz), 100 floats per Pfam."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    result = {}
    with opener(path, "rt", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "pfam_id" not in fields:
            raise ValueError("DeepBGC CSV must have pfam_id column")
        dims = [x for x in fields if x != "pfam_id"]
        if len(dims) != 100:
            raise ValueError(f"Expected official 100D DeepBGC vectors, got {len(dims)} columns")
        for row in reader:
            key = canonical_pfam(row["pfam_id"])
            if key in result:
                raise ValueError(f"Duplicate DeepBGC Pfam accession after version normalization: {key}")
            try:
                values = np.array([float(row[k]) for k in dims], dtype=np.float32)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid DeepBGC vector for {key}") from exc
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite DeepBGC vector: {key}")
            result[key] = values
    if not result:
        raise ValueError("DeepBGC file contains no vectors")
    return result


def select_features(path: str | Path, rows: list[dict]) -> dict[str, torch.Tensor]:
    """Join by row_uid; cache format exactly matches newkcat features.npz."""
    with np.load(path, allow_pickle=False) as arrays:
        if not {"sample_ids", "protein", "molt5", "maccs"}.issubset(arrays.files):
            raise ValueError("features.npz needs sample_ids,protein,molt5,maccs")
        ids = arrays["sample_ids"]
        if ids.ndim != 1 or ids.dtype.kind != "U":
            raise ValueError("feature sample_ids must be a 1-D Unicode array")
        names = ids.tolist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicated row_uid in feature cache")
        lookup = {value: i for i, value in enumerate(names)}
        missing = [r["row_uid"] for r in rows if r["row_uid"] not in lookup]
        if missing:
            raise ValueError(f"features.npz missing {len(missing)} row_uids, e.g. {missing[:3]}")
        index = np.array([lookup[r["row_uid"]] for r in rows])
        selected = {}
        for name, dim in (("protein", 1024), ("molt5", 768), ("maccs", 167)):
            arr = arrays[name]
            if arr.shape != (len(names), dim) or arr.dtype.kind not in "fiu":
                raise ValueError(f"{name} must have shape [{len(names)},{dim}] and numeric dtype")
            with np.errstate(over="ignore", invalid="ignore"):
                part = np.array(arr[index], dtype=np.float32)
            if not np.isfinite(part).all():
                raise ValueError(f"Nonfinite {name} features")
            if name == "maccs" and (not np.isin(part, (0, 1)).all() or np.any(part[:, 0])):
                raise ValueError("MACCS cache must contain 167 binary bits and zero reserved bit 0")
            selected[name] = torch.from_numpy(part)
    return selected


class PfamDataset(Dataset):
    def __init__(self, rows: list[dict], hits: dict, vocab: dict, nmax: int,
                 ordered: bool, feature_path: str | Path):
        self.rows = rows
        self.global_features = select_features(feature_path, rows)
        self.tokens = torch.from_numpy(np.stack([
            encode(r["row_uid"], hits, vocab, nmax, ordered) for r in rows
        ]))
        self.labels = torch.tensor([r["target"] for r in rows], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return {**{k: v[index] for k, v in self.global_features.items()},
                "pfam_ids": self.tokens[index], "target": self.labels[index]}
