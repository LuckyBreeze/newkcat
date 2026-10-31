"""Preparation gates: paired information, immutable labels, explicit failures."""

import csv
import hashlib
import importlib.util
import json

import pytest

from newkcat.sequence import prepare
from newkcat.sequence.chemistry import dynamic_spans


MAPPED = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]"
SR = "[CH3]{[CH2]|[CH]}{-|=}{[OH]|[O]}"


def _rows():
    return [{
        "sample_id": f"sample{i}", "sequence": "M"+"ACD"[i]*5,
        "smiles": "CCO", "log10_kcat": ["1.0000", "-2.300", "0.00"][i],
        "split": ["train", "val", "test"][i], "group_id": f"cluster{i}",
        "reaction_smiles": "CCO>>CC=O", "mapped_rxn": MAPPED,
    } for i in range(3)]


def _write(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _converted():
    return {"reaction_smiles": "CCO>>CC=O", "sr_smiles": SR,
            "edit_spans": dynamic_spans(SR), "mapping_provenance": [],
            "converter_version": "mock_test_only"}


def _mock_conversion(monkeypatch, *, bad="BAD"):
    calls = []

    def convert(text):
        calls.append(text)
        if text == bad:
            raise ValueError("Unmapped atom or incompatible supplied reaction")
        return _converted()

    monkeypatch.setattr(prepare, "prepare_reaction", convert)
    return calls


def test_successful_preparation_preserves_ids_labels_splits_and_caches_reactions(tmp_path,
                                                                             monkeypatch):
    calls = _mock_conversion(monkeypatch)
    rows = _rows()
    source = _write(tmp_path / "source.csv", rows)
    before = source.read_bytes()
    report = prepare.prepare_sequences(source, output_dir=tmp_path / "prepared")
    assert report["n"] == 3 and report["converted"] == 3 and report["rows_dropped"] == 0
    assert report["failed"] == 0 and report["labels_recomputed"] is False
    assert report["lengths_including_bos_eos"]["sr_smiles"]["maximum"] == len(SR)+2
    assert calls == [MAPPED]
    with (tmp_path / "prepared" / "samples.csv").open() as handle:
        prepared = list(csv.DictReader(handle))
    for actual, original in zip(prepared, rows, strict=True):
        for field in ("sample_id", "sequence", "log10_kcat", "split", "smiles", "group_id"):
            assert actual[field] == original[field]
        assert actual["source_reaction_smiles"] == original["reaction_smiles"]
        assert actual["reaction_status"] == "ok"
        assert json.loads(actual["edit_spans"]) == dynamic_spans(actual["sr_smiles"])
    assert source.read_bytes() == before
    prepared_file = tmp_path / "prepared" / "samples.csv"
    provenance = tmp_path / "prepared" / "reaction_provenance.jsonl"
    assert report["samples_sha256"] == hashlib.sha256(prepared_file.read_bytes()).hexdigest()
    assert report["provenance_sha256"] == hashlib.sha256(provenance.read_bytes()).hexdigest()
    audit = [json.loads(line) for line in provenance.read_text().splitlines()]
    assert len(audit) == 1 and audit[0]["mapped_rxn"] == MAPPED
    assert audit[0]["conversion"]["sr_smiles"] == SR


def test_explicit_fallback_is_shared_by_both_representations_and_retains_rows(tmp_path,
                                                                          monkeypatch):
    _mock_conversion(monkeypatch)
    rows = _rows()
    rows[1]["mapped_rxn"] = "BAD"
    source = _write(tmp_path / "source.csv", rows)
    report = prepare.prepare_sequences(source, output_dir=tmp_path / "prepared", on_error="substrate")
    assert report["n"] == 3 and report["failed"] == 1 and report["rows_dropped"] == 0
    assert report["counts"]["val"] == {"fallback_substrate": 1}
    with (tmp_path / "prepared" / "samples.csv").open() as handle:
        prepared = list(csv.DictReader(handle))
    assert [row["sample_id"] for row in prepared] == [row["sample_id"] for row in rows]
    assert prepared[1]["reaction_smiles"] == prepared[1]["sr_smiles"] == ""
    assert prepared[1]["edit_spans"] == "[]"
    assert prepared[1]["log10_kcat"] == rows[1]["log10_kcat"]


def test_full_reaction_policy_retains_every_row_and_source_information(tmp_path, monkeypatch):
    _mock_conversion(monkeypatch)
    rows = _rows()
    rows[1]["mapped_rxn"] = "BAD"
    source = _write(tmp_path / "source.csv", rows)
    report = prepare.prepare_sequences(source, output_dir=tmp_path / "prepared", on_error="reaction")
    assert report["n"] == 3 and report["rows_dropped"] == 0
    assert report["reaction_available"] == 3 and report["sr_available"] == 2
    assert report["fallback_reaction_rows"] == 1 and report["fallback_substrate_rows"] == 0
    assert report["samples_written"] is True
    with (tmp_path / "prepared" / "samples.csv").open() as handle:
        prepared = list(csv.DictReader(handle))
    for actual, original in zip(prepared, rows, strict=True):
        for field in ("sample_id", "sequence", "smiles", "log10_kcat", "split", "group_id"):
            assert actual[field] == original[field]
        assert actual["reaction_smiles"] == "CCO>>CC=O"
    assert prepared[1]["reaction_status"] == "fallback_reaction"
    assert prepared[1]["sr_smiles"] == "" and prepared[1]["edit_spans"] == "[]"
    assert prepared[1]["sr_input_format"] == "reaction_smiles"
    assert prepared[1]["reaction_error"]


@pytest.mark.parametrize("reaction", ["", "not_a_reaction", "C>>"])
def test_full_reaction_policy_does_not_hide_invalid_source_text(tmp_path, monkeypatch, reaction):
    _mock_conversion(monkeypatch)
    rows = _rows()
    rows[0]["reaction_smiles"] = reaction
    source = _write(tmp_path / "source.csv", rows)
    with pytest.raises(ValueError, match="full reaction text unavailable"):
        prepare.prepare_sequences(source, output_dir=tmp_path / "prepared", on_error="reaction")
    assert not (tmp_path / "prepared" / "samples.csv").exists()


def test_reaction_only_needs_neither_mapping_file_nor_converter(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Native reaction preparation must not load mappings or sr converter")

    monkeypatch.setattr(prepare, "prepare_reaction", forbidden)
    monkeypatch.setattr(prepare, "_mapping_records", forbidden)
    rows = _rows()
    for row in rows:
        row["mapped_rxn"] = "bad_uninspected_mapping"
    source = _write(tmp_path / "source.csv", rows)
    report = prepare.prepare_sequences(source, output_dir=tmp_path / "prepared", reaction_only=True,
                                       mapping_jsonl=tmp_path / "missing_mapping.jsonl")
    assert report["mode"] == "reaction_only" and report["samples_written"] is True
    assert report["n"] == report["reaction_available"] == 3
    assert report["sr_available"] == report["failed"] == 0
    assert report["converter_versions"] == report["decoder_compat"] == []
    assert report["mapping_sha256"] is None
    with (tmp_path / "prepared" / "samples.csv").open() as handle:
        prepared = list(csv.DictReader(handle))
    assert all(row["reaction_status"] == "reaction_only" for row in prepared)
    assert all(row["reaction_smiles"] == "CCO>>CC=O" for row in prepared)
    assert all(row["sr_smiles"] == "" and row["edit_spans"] == "[]" for row in prepared)
    assert [row["log10_kcat"] for row in prepared] == [row["log10_kcat"] for row in rows]


@pytest.mark.parametrize("reaction_only", [False, True])
def test_full_reaction_entries_flag_substrate_disagreement_without_substitution(
        tmp_path, monkeypatch, reaction_only):
    _mock_conversion(monkeypatch)
    rows = _rows()
    rows[0]["smiles"] = "CC"
    source = _write(tmp_path / "source.csv", rows)
    report = prepare.prepare_sequences(source, output_dir=tmp_path / "prepared",
                                       on_error="reaction", reaction_only=reaction_only)
    assert report["annotation_issue_rows"] == 1 and report["reaction_available"] == 3
    with (tmp_path / "prepared" / "samples.csv").open() as handle:
        prepared = list(csv.DictReader(handle))
    assert prepared[0]["smiles"] == "CC"
    assert prepared[0]["reaction_smiles"] == rows[0]["reaction_smiles"]
    assert prepared[0]["annotation_issue"] and prepared[0]["substrate_match"] == "false"
    assert prepared[0]["reaction_status"] == ("reaction_only" if reaction_only else "fallback_reaction")


def test_decoder_compatibility_is_persisted_in_preparation_report(tmp_path, monkeypatch):
    value = {**_converted(), "sr_decoder_compat": "example_verified_adapter_v1"}
    monkeypatch.setattr(prepare, "prepare_reaction", lambda _: value)
    report = prepare.prepare_sequences(_write(tmp_path / "source.csv", _rows()),
                                       output_dir=tmp_path / "prepared")
    assert report["decoder_compat"] == ["example_verified_adapter_v1"]


def test_failed_strict_run_reports_coverage_and_leaves_no_runnable_manifest(tmp_path, monkeypatch):
    _mock_conversion(monkeypatch)
    rows = _rows()
    rows[0]["mapped_rxn"] = "BAD"
    source = _write(tmp_path / "source.csv", rows)
    output = tmp_path / "prepared"
    output.mkdir()
    (output / "samples.csv").write_text("old_runnable_manifest")
    with pytest.raises(ValueError, match="failed preparation"):
        prepare.prepare_sequences(source, output_dir=output, overwrite=True)
    assert not (output / "samples.csv").exists()
    report = json.loads((output / "coverage.json").read_text())
    assert report["failed"] == 1 and report["samples_written"] is False
    assert (output / "conversion_errors.csv").exists()


def test_dependency_or_programming_errors_do_not_turn_into_substrate_fallback(tmp_path,
                                                                           monkeypatch):
    def unavailable(_):
        raise ImportError("Optional converter missing")

    monkeypatch.setattr(prepare, "prepare_reaction", unavailable)
    source = _write(tmp_path / "source.csv", _rows())
    with pytest.raises(ImportError, match="converter"):
        prepare.prepare_sequences(source, output_dir=tmp_path / "prepared", on_error="substrate")
    assert not (tmp_path / "prepared" / "samples.csv").exists()


def test_invalid_substrate_is_fatal_not_reaction_fallback(tmp_path, monkeypatch):
    _mock_conversion(monkeypatch)
    rows = _rows()
    rows[0]["smiles"] = "not_a_smiles"
    source = _write(tmp_path / "source.csv", rows)
    with pytest.raises(ValueError, match="SMILES"):
        prepare.prepare_sequences(source, output_dir=tmp_path / "prepared", on_error="substrate")


def test_mapping_jsonl_is_joined_by_exact_source_reaction(tmp_path, monkeypatch):
    calls = _mock_conversion(monkeypatch)
    rows = _rows()
    for row in rows:
        row.pop("mapped_rxn")
    source = _write(tmp_path / "source.csv", rows)
    mapping = tmp_path / "mappings.jsonl"
    mapping.write_text(json.dumps({"reaction_smiles": "CCO>>CC=O", "mapped_rxn": MAPPED,
                                   "map_status": "ok"})+"\n")
    report = prepare.prepare_sequences(source, output_dir=tmp_path / "prepared", mapping_jsonl=mapping)
    assert report["converted"] == 3 and calls == [MAPPED]
    assert len(report["mapping_sha256"]) == 64


def test_duplicate_mapping_keys_are_rejected(tmp_path, monkeypatch):
    _mock_conversion(monkeypatch)
    source = _write(tmp_path / "source.csv", _rows())
    mapping = tmp_path / "mappings.jsonl"
    entry = json.dumps({"reaction_smiles": "CCO>>CC=O", "mapped_rxn": MAPPED})
    mapping.write_text(entry+"\n"+entry+"\n")
    with pytest.raises(ValueError, match="Duplicate reaction key"):
        prepare.prepare_sequences(source, output_dir=tmp_path / "prepared", mapping_jsonl=mapping)


def test_failed_mapping_audit_records_allow_empty_mapped_rxn(tmp_path, monkeypatch):
    _mock_conversion(monkeypatch)
    rows = _rows()
    for row in rows:
        row.pop("mapped_rxn")
    source = _write(tmp_path / "source.csv", rows)
    mapping = tmp_path / "mappings.jsonl"
    mapping.write_text("\n".join(json.dumps(entry) for entry in [
        {"reaction_smiles": "CC>>C=C", "mapped_rxn": "", "map_status": "unbalanced"},
        {"reaction_smiles": "CCO>>CC=O", "mapped_rxn": MAPPED, "map_status": "ok"},
    ])+"\n")
    report = prepare.prepare_sequences(source, output_dir=tmp_path / "prepared", mapping_jsonl=mapping)
    assert report["converted"] == 3
    mapping.write_text(json.dumps({"reaction_smiles": "CCO>>CC=O", "mapped_rxn": "",
                                   "map_status": "mapper_fail"})+"\n")
    report = prepare.prepare_sequences(source, output_dir=tmp_path / "fallback", mapping_jsonl=mapping,
                                       on_error="substrate")
    assert report["failed"] == 3


def test_failed_mapping_keys_still_participate_in_duplicate_detection(tmp_path):
    mapping = tmp_path / "mappings.jsonl"
    mapping.write_text("\n".join(json.dumps(entry) for entry in [
        {"reaction_smiles": "CCO>>CC=O", "map_status": "unbalanced"},
        {"reaction_smiles": "CCO>>CC=O", "mapped_rxn": MAPPED, "map_status": "ok"},
    ])+"\n")
    with pytest.raises(ValueError, match="Duplicate reaction key"):
        prepare._mapping_records(mapping)


def test_overlong_representation_uses_one_paired_availability_decision(tmp_path, monkeypatch):
    _mock_conversion(monkeypatch)
    source = _write(tmp_path / "source.csv", _rows())
    report = prepare.prepare_sequences(source, output_dir=tmp_path / "prepared",
                                       max_length=12, on_error="substrate")
    assert report["failed"] == 3 and report["converted"] == 0
    with (tmp_path / "prepared" / "samples.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["reaction_status"] == "fallback_substrate" for row in rows)


def test_prepare_cannot_write_over_inputs_inside_output_directory(tmp_path, monkeypatch):
    _mock_conversion(monkeypatch)
    output = tmp_path / "prepared"
    output.mkdir()
    source = _write(output / "conversion_errors.csv", _rows())
    before = source.read_bytes()
    with pytest.raises(ValueError, match="input|contain|new directory"):
        prepare.prepare_sequences(source, output_dir=output, overwrite=True)
    assert source.read_bytes() == before


def test_supplied_mapping_cannot_substitute_a_different_source_reaction(tmp_path, monkeypatch):
    _mock_conversion(monkeypatch)
    rows = _rows()
    rows[0]["reaction_smiles"] = "CC>>C=C"
    source = _write(tmp_path / "source.csv", rows)
    report = prepare.prepare_sequences(source, output_dir=tmp_path / "prepared", on_error="substrate")
    assert report["failed"] == 1 and report["converted"] == 2
    with (tmp_path / "prepared" / "conversion_errors.csv").open() as handle:
        errors = list(csv.DictReader(handle))
    assert errors[0]["sample_id"] == "sample0"


def test_cached_reaction_still_checks_each_rows_substrate(tmp_path, monkeypatch):
    calls = _mock_conversion(monkeypatch)
    rows = _rows()
    rows[0]["smiles"] = "CC"
    source = _write(tmp_path / "source.csv", rows)
    report = prepare.prepare_sequences(source, output_dir=tmp_path / "prepared", on_error="substrate")
    assert report["failed"] == 1 and report["converted"] == 2
    assert calls == [MAPPED]
    with (tmp_path / "prepared" / "samples.csv").open() as handle:
        prepared = list(csv.DictReader(handle))
    assert [row["reaction_status"] for row in prepared] == ["fallback_substrate", "ok", "ok"]


def _split_sources(tmp_path):
    paths = {}
    labels = ["1.2300", "-4.000", "0.000"]
    for index, split in enumerate(("train", "valid", "test")):
        rows = [{"row_uid": f"mckcat:{index}", "sequence": "M"+"ACD"[index]*5,
                 "substrate": "*CCO", "log10kcat_max": labels[index],
                 "split": split, "cluster_id": f"group{index}",
                 "reaction_smiles": "CCO>>CC=O"}]
        paths[split] = _write(tmp_path / f"{split}.csv", rows)
    return paths, labels


def test_import_renames_columns_without_recomputing_labels_or_splits(tmp_path):
    paths, labels = _split_sources(tmp_path)
    report = prepare.import_splits(paths["train"], paths["valid"], paths["test"],
                                    output_dir=tmp_path / "imported")
    with (tmp_path / "imported" / "samples.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["log10_kcat"] for row in rows] == labels
    assert [row["split"] for row in rows] == ["train", "val", "test"]
    assert [row["sample_id"] for row in rows] == ["mckcat:0", "mckcat:1", "mckcat:2"]
    assert report["counts"] == {"train": 1, "val": 1, "test": 1}
    assert report["labels_recomputed"] is False and report["splits_regenerated"] is False


def test_import_rejects_repeated_input_paths_and_split_disagreement(tmp_path):
    paths, _ = _split_sources(tmp_path)
    with pytest.raises(ValueError, match="distinct"):
        prepare.import_splits(paths["train"], paths["train"], paths["test"],
                              output_dir=tmp_path / "bad")
    with pytest.raises(ValueError, match="split disagrees"):
        prepare.import_splits(paths["valid"], paths["train"], paths["test"],
                              output_dir=tmp_path / "bad")


def test_import_does_not_publish_manifest_with_group_leakage(tmp_path):
    paths, _ = _split_sources(tmp_path)
    with paths["valid"].open() as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["cluster_id"] = "group0"
    _write(paths["valid"], rows)
    with pytest.raises(ValueError, match="leakage"):
        prepare.import_splits(paths["train"], paths["valid"], paths["test"],
                              output_dir=tmp_path / "bad")
    assert not (tmp_path / "bad" / "samples.csv").exists()


@pytest.mark.skipif(importlib.util.find_spec("sr_smiles") is None,
                    reason="Optional audited sr-smiles converter not installed")
def test_real_converter_preparation_roundtrip_smoke(tmp_path):
    source = _write(tmp_path / "source.csv", _rows())
    report = prepare.prepare_sequences(source, output_dir=tmp_path / "prepared")
    assert report["converted"] == 3 and report["failed"] == 0
    assert report["converter_versions"] == ["0.2.0"]
