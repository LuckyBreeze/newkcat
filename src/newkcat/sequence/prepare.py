"""Audited imports and reaction preparations without dropping kinetic rows."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from ..data import read_manifest
from ..train import _json_write, _prepare_output, sha256_file
from .chemistry import normalize_reaction, normalize_substrate, prepare_reaction
from .tokenizer import ReactionTokenizer


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def _protect_sources(output: Path, paths) -> None:
    for path in paths:
        if path is not None and (path == output or output in path.parents):
            raise ValueError(f"Output directory must not contain input artifact {path}")


def import_splits(train: str | Path, val: str | Path, test: str | Path, *,
                  output_dir: str | Path, id_column: str = "row_uid",
                  smiles_column: str = "substrate", target_column: str = "log10kcat_max",
                  sequence_column: str = "sequence", reaction_column: str = "reaction_smiles",
                  group_column: str = "cluster_id", overwrite: bool = False) -> dict:
    """Rename explicit columns; never recalculate targets or regenerate splits."""
    paths = {key: Path(path).resolve() for key, path in (("train", train), ("val", val),
                                                       ("test", test))}
    if len(set(paths.values())) != 3:
        raise ValueError("Train, validation and test must be three distinct input files")
    output = Path(output_dir).resolve()
    _protect_sources(output, paths.values())
    destination = output / "samples.csv"
    if destination in paths.values():
        raise ValueError("Import must not overwrite an input samples file")
    rows = []
    counts = {}
    required = {id_column, smiles_column, target_column, sequence_column}
    for split, path in paths.items():
        count = 0
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames or []
            if len(headers) != len(set(headers)) or not required.issubset(headers):
                raise ValueError(f"{path}: need unique columns including {sorted(required)}")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"{path}: inconsistent CSV field count")
                original_split = row.get("split", "").strip()
                if original_split and {"valid": "val", "validation": "val"}.get(
                        original_split, original_split) != split:
                    raise ValueError(f"{path}: row split disagrees with the supplied split file")
                converted = dict(row)
                aliases = {
                    "sample_id": row[id_column], "smiles": row[smiles_column],
                    "sequence": row[sequence_column], "log10_kcat": row[target_column],
                    "split": split,
                }
                if reaction_column in row:
                    aliases["reaction_smiles"] = row[reaction_column]
                if group_column and group_column in row:
                    aliases["group_id"] = row[group_column]
                for key, value in aliases.items():
                    if key in converted and key != "split" and converted[key] != value:
                        raise ValueError(f"Conflicting existing {key} column in {path}")
                converted.update(aliases)
                rows.append(converted)
                count += 1
        counts[split] = count
    _prepare_output(output, overwrite)
    _write_csv(destination, rows)
    try:
        read_manifest(destination)
    except ValueError:
        destination.unlink()
        raise
    report = {
        "counts": counts, "n": len(rows), "samples": str(destination),
        "id_column": id_column, "target_column": target_column,
        "source_sha256": {key: sha256_file(path) for key, path in paths.items()},
        "labels_recomputed": False, "splits_regenerated": False,
        "note": "IDs must match the existing feature cache exactly. valid is renamed to val.",
    }
    _json_write(output / "import_report.json", report)
    return report


def _mapping_records(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    lookup, seen = {}, set()
    with path.open(encoding="utf-8") as handle:
        for line, text in enumerate(handle, 1):
            if not text.strip():
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid mapping JSONL at line {line}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Mapping line {line} must be an object")
            key, mapped = row.get("reaction_smiles"), row.get("mapped_rxn")
            if not isinstance(key, str) or not key:
                raise ValueError(f"Mapping line {line} needs reaction_smiles")
            if key in seen:
                raise ValueError(f"Duplicate reaction key in mapping JSONL at line {line}")
            seen.add(key)
            # Mapping audit files deliberately retain failed reactions with no
            # mapped_rxn. These are missing inputs, not malformed successful maps.
            if row.get("map_status", "ok") != "ok":
                continue
            if not isinstance(mapped, str) or not mapped:
                raise ValueError(f"Successful mapping line {line} needs mapped_rxn")
            lookup[key] = mapped
    return lookup


def prepare_sequences(samples: str | Path, *, output_dir: str | Path,
                      mapping_jsonl: str | Path | None = None,
                      mapped_column: str = "mapped_rxn", max_length: int = 2048,
                      on_error: str = "error", overwrite: bool = False,
                      reaction_only: bool = False) -> dict:
    """Keep sample identity separate from the availability of a representation.

    With on_error=error, a coverage report is written but no usable samples.csv
    is produced if any conversion fails. on_error=substrate explicitly retains
    every row using substrate-only input in BOTH reaction representations.

    on_error=reaction retains the original full reaction whenever sr conversion
    fails. It requires valid, untruncated source reaction text for every row;
    an annotation disagreement is recorded and never silently repaired.
    reaction_only prepares R without loading mappings or calling a converter.
    """
    if on_error not in {"error", "substrate", "reaction"}:
        raise ValueError("on_error must be error, substrate or reaction")
    if type(reaction_only) is not bool:
        raise ValueError("reaction_only must be a boolean")
    if type(max_length) is not int or max_length < 3:
        raise ValueError("max_length must be an integer >=3")
    source = Path(samples).resolve()
    mapping_path = (Path(mapping_jsonl).resolve()
                    if mapping_jsonl and not reaction_only else None)
    output = Path(output_dir).resolve()
    _protect_sources(output, (source, mapping_path))
    if source == output / "samples.csv":
        raise ValueError("Prepare must write to a new directory, not overwrite input samples")
    records = read_manifest(source)
    mappings = {} if reaction_only else _mapping_records(mapping_path)
    tokenizer = ReactionTokenizer()
    cache: dict[tuple[str, str], tuple[dict | None, str | None]] = {}
    source_cache: dict[str, tuple[str | None, int | None, str | None]] = {}
    result_rows, audit_rows = [], []
    lengths = {"reaction_smiles": [], "sr_smiles": []}
    counts: dict[str, Counter] = {split: Counter() for split in ("train", "val", "test")}
    versions, decoder_compat = set(), set()
    reaction_available = sr_available = annotation_issues = 0
    for row in records:
        result = dict(row)
        # A malformed substrate is not a recoverable missing-reaction condition.
        substrate = normalize_substrate(row["smiles"])
        tokenizer.encode(substrate, max_length=max_length)
        original = row.get("reaction_smiles", "")
        if original not in source_cache:
            try:
                normalized = normalize_reaction(original)
                n_tokens = len(tokenizer.encode(normalized, max_length=max_length)["input_ids"])
                source_cache[original] = (normalized, n_tokens, None)
            except ValueError as error:
                source_cache[original] = (None, None, str(error))
        source_reaction, source_length, source_error = source_cache[original]
        if (reaction_only or on_error == "reaction") and source_error:
            raise ValueError(f"Sample {row['sample_id']}: full reaction text unavailable: {source_error}")

        mapped, converted, reason = row.get(mapped_column, "") if reaction_only else "", None, None
        mapping_source = "reaction_only"
        if not reaction_only:
            mapped = row.get(mapped_column, "")
            mapping_source = ("inline" if mapped else "mapping_jsonl"
                              if mapping_path else "reaction_column")
            if not mapped:
                mapped = mappings.get(original, "") if mapping_path else original
            cache_key = (original, mapped)
            if cache_key not in cache:
                try:
                    if not mapped:
                        raise ValueError("No mapped reaction matched this sample")
                    if original and normalize_reaction(original) != normalize_reaction(mapped):
                        raise ValueError("Source and mapped reaction structures differ; refusing substitution")
                    converted = prepare_reaction(mapped)
                    rxn_tokens = tokenizer.encode(converted["reaction_smiles"],
                                                  max_length=max_length)
                    sr_tokens = tokenizer.encode(converted["sr_smiles"],
                                                 edit_spans=converted["edit_spans"],
                                                 max_length=max_length)
                    converted = {**converted, "reaction_length": len(rxn_tokens["input_ids"]),
                                 "sr_length": len(sr_tokens["input_ids"])}
                    cache[cache_key] = (converted, None)
                except ValueError as error:
                    cache[cache_key] = (None, str(error))
            converted, reason = cache[cache_key]

        available_text = source_reaction or (converted["reaction_smiles"] if converted else None)
        annotation_issue = ""
        if available_text:
            reactants = available_text.split(">>")[0]
            substrate_matches = Counter(substrate.split(".")) <= Counter(reactants.split("."))
            if not substrate_matches:
                annotation_issue = "Recorded substrate components are not contained in the reaction reactants"
                annotation_issues += 1
                if converted is not None:
                    converted, reason = None, annotation_issue
        else:
            substrate_matches = None

        result["source_reaction_smiles"] = original
        result["substrate_smiles"] = substrate
        result["mapped_rxn"] = mapped
        result["mapping_source"] = mapping_source
        result["annotation_issue"] = annotation_issue
        result["substrate_match"] = ("unknown" if substrate_matches is None
                                      else str(substrate_matches).lower())
        if source_reaction is not None:
            reaction_available += 1
            lengths["reaction_smiles"].append(source_length)
        elif converted is not None:
            # Legacy strict inputs may supply only mapped_rxn, without source text.
            reaction_available += 1
            lengths["reaction_smiles"].append(converted["reaction_length"])

        if reaction_only:
            status = "reaction_only"
            result.update(reaction_smiles=source_reaction, sr_smiles="", edit_spans="[]")
        elif converted is None:
            status = "fallback_reaction" if on_error == "reaction" else "fallback_substrate"
            result.update(reaction_smiles=source_reaction if on_error == "reaction" else "",
                          sr_smiles="", edit_spans="[]")
            audit_rows.append({"sample_id": row["sample_id"], "split": row["split"],
                               "reason": reason})
        else:
            status = "ok"
            sr_available += 1
            versions.add(str(converted["converter_version"]))
            if converted.get("sr_decoder_compat"):
                decoder_compat.add(str(converted["sr_decoder_compat"]))
            result.update(reaction_smiles=converted["reaction_smiles"],
                          sr_smiles=converted["sr_smiles"],
                          edit_spans=json.dumps(converted["edit_spans"], separators=(",", ":")))
            lengths["sr_smiles"].append(converted["sr_length"])
        result["reaction_status"] = status
        result["reaction_error"] = reason or ""
        result["reaction_input_format"] = ("reaction_smiles" if result["reaction_smiles"]
                                            else "substrate_smiles")
        result["sr_input_format"] = ("sr_smiles" if status == "ok" else "reaction_smiles"
                                      if status == "fallback_reaction" else ""
                                      if status == "reaction_only" else "substrate_smiles")
        result["source_reaction_verified"] = str(bool(original and status == "ok")).lower()
        counts[row["split"]][status] += 1
        result_rows.append(result)
    _prepare_output(output, overwrite)
    # A failed strict rerun must not leave an old runnable manifest behind.
    destination = output / "samples.csv"
    samples_written = not audit_rows or on_error != "error"
    if not samples_written:
        if destination.exists():
            destination.unlink()
    else:
        _write_csv(destination, result_rows)
    _write_csv(output / "conversion_errors.csv", audit_rows, ["sample_id", "split", "reason"])
    with (output / "reaction_provenance.jsonl").open("w", encoding="utf-8") as handle:
        if reaction_only:
            for original, (normalized, _, error) in source_cache.items():
                handle.write(json.dumps({"source_reaction_smiles": original,
                                         "reaction_smiles": normalized, "mapped_rxn": "",
                                         "conversion": None, "error": error,
                                         "mode": "reaction_only"},
                                        ensure_ascii=False, allow_nan=False)+"\n")
        for (original, mapped), (converted, reason) in cache.items():
            handle.write(json.dumps({"source_reaction_smiles": original, "mapped_rxn": mapped,
                                     "conversion": converted, "error": reason},
                                    ensure_ascii=False, allow_nan=False)+"\n")
    length_summary = {}
    for name, values in lengths.items():
        if values:
            array = np.asarray(values)
            length_summary[name] = {"n": len(values), "median": float(np.median(array)),
                                    "p95": float(np.quantile(array, .95)),
                                    "maximum": int(array.max())}
    report = {
        "schema_version": 1, "n": len(records), "converted": len(records)-len(audit_rows),
        "failed": len(audit_rows), "policy": on_error,
        "mode": "reaction_only" if reaction_only else "paired",
        "reaction_available": reaction_available, "sr_available": sr_available,
        "sr_preparation_attempted": not reaction_only,
        "fallback_reaction_rows": sum(c["fallback_reaction"] for c in counts.values()),
        "fallback_substrate_rows": (sum(c["fallback_substrate"] for c in counts.values())
                                    if on_error == "substrate" else 0),
        "annotation_issue_rows": annotation_issues,
        "counts": {split: dict(count) for split, count in counts.items()},
        "source_sha256": sha256_file(source),
        "mapping_sha256": sha256_file(mapping_path) if mapping_path else None,
        "converter_versions": sorted(versions), "decoder_compat": sorted(decoder_compat),
        "max_length": max_length,
        "tokenizer": tokenizer.metadata(), "lengths_including_bos_eos": length_summary,
        "samples_written": samples_written,
        "samples_sha256": sha256_file(destination) if destination.exists() else None,
        "provenance_sha256": sha256_file(output / "reaction_provenance.jsonl"),
        "rows_dropped": 0, "labels_recomputed": False,
        "local_definition": "characters inside parsed sr dynamic atom/bond spans; no radius expansion",
        "note": "Availability measures representation support, not validity of kcat labels. "
                "Roundtrip validates representation integrity, not biochemical mapping correctness. "
                "The reaction policy preserves source reactions, including flagged annotation conflicts; "
                "sr loading requires explicit missing_reaction: reaction. Legacy substrate fallback "
                "remains shared by R/C and requires missing_reaction: substrate.",
    }
    _json_write(output / "coverage.json", report)
    if audit_rows and on_error == "error":
        raise ValueError(f"{len(audit_rows)} reaction rows failed preparation. Read {output / 'coverage.json'} "
                         "and conversion_errors.csv; no training manifest was written. "
                         "Choose --on-error reaction to retain valid full reactions, or "
                         "--on-error substrate explicitly for the legacy fallback.")
    return report
