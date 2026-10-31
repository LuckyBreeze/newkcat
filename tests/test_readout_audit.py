"""Row identity, paired estimates, and honest cohort reporting for readout audits."""

import csv
import json

import numpy as np
import pytest

from newkcat.readout.audit import audit_predictions


def write_csv(path, rows, fields=None):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def manifest(tmp_path):
    rows = [
        {"sample_id": "a", "split": "test", "log10_kcat": 0.0, "group_id": "g1",
         "reaction_status": "ok", "reaction_smiles": "CC>>CC", "sr_smiles": "CC",
         "edit_spans": "[]"},
        {"sample_id": "b", "split": "test", "log10_kcat": 1.0, "group_id": "g1",
         "reaction_status": "fallback_reaction", "reaction_smiles": "C>>C",
         "sr_smiles": "", "edit_spans": "[]"},
        {"sample_id": "c", "split": "test", "log10_kcat": 2.0, "group_id": "g2",
         "reaction_status": "ok", "reaction_smiles": "CO>>C=O",
         "sr_smiles": "C{-|=}O", "edit_spans": "[[1,6]]"},
        {"sample_id": "d", "split": "test", "log10_kcat": 3.0, "group_id": "g3",
         "reaction_status": "fallback_reaction", "reaction_smiles": "C"*260,
         "sr_smiles": "", "edit_spans": "[]"},
        {"sample_id": "train", "split": "train", "log10_kcat": 9.0, "group_id": "g4",
         "reaction_status": "ok", "reaction_smiles": "C>>C", "sr_smiles": "C",
         "edit_spans": "[]"},
    ]
    return write_csv(tmp_path / "samples.csv", rows), rows


def predictions(tmp_path, name, values, ids=("a", "b", "c", "d"), aliases=False):
    target_key, prediction_key = ("y_true", "y_pred") if aliases else ("target", "prediction")
    labels = {"a": 0, "b": 1, "c": 2, "d": 3, "extra": 4}
    return write_csv(tmp_path / f"{name}.csv", [
        {"sample_id": sid, target_key: labels[sid], prediction_key: value,
         "reaction_status": "different_prediction_status"}
        for sid, value in zip(ids, values)
    ])


def test_metadata_only_preserves_all_split_rows_and_groups(manifest, tmp_path):
    source, _ = manifest
    report = audit_predictions(source, None, "test", tmp_path / "audit")
    assert report["n"] == 4 and report["rows_dropped"] == 0
    assert report["models"] == [] and report["model_metrics"] == []
    statuses = {r["group"]: r for r in report["coverage"] if r["grouping"] == "reaction_status"}
    assert statuses["ok"]["n"] == statuses["fallback_reaction"]["n"] == 2
    assert statuses["ok"]["label_mean"] == 1
    assert statuses["fallback_reaction"]["label_mean"] == 2
    length_groups = [r for r in report["coverage"] if r["grouping"] == "selected_input_tokens"]
    assert sum(r["n"] for r in length_groups) == 4
    joined = list(csv.DictReader((tmp_path / "audit/joined_predictions.csv").open()))
    assert [r["sample_id"] for r in joined] == ["a", "b", "c", "d"]
    assert joined[2]["dynamic_characters"] == "5"
    assert json.loads((tmp_path / "audit/audit.json").read_text()) == report


def test_id_join_uses_manifest_status_and_exact_paired_mse(manifest, tmp_path):
    source, _ = manifest
    a = predictions(tmp_path, "A", [3, 2, 1, 0], ids=("d", "c", "b", "a"))
    b = predictions(tmp_path, "B", [1, 2, 3, 4], aliases=True)
    report = audit_predictions(source, {"A": a, "B": b}, "test", tmp_path / "audit")
    overall = {r["model"]: r for r in report["model_metrics"] if r["grouping"] == "overall"}
    assert overall["A"]["mse"] == 0 and overall["B"]["mse"] == 1
    groups = [r for r in report["paired_differences"] if r["grouping"] == "reaction_status"]
    assert {r["group"] for r in groups} == {"ok", "fallback_reaction"}
    assert sum(r["weighted_delta_mse"] for r in groups) == -1
    assert all(r["bootstrap_status"] == "disabled" for r in groups)


@pytest.mark.parametrize("ids,values", [(("a", "b", "c"), [0, 1, 2]),
                                       (("a", "b", "c", "d", "extra"), [0, 1, 2, 3, 4])])
def test_rejects_row_loss_or_extra_rows(manifest, tmp_path, ids, values):
    source, _ = manifest
    p = predictions(tmp_path, "bad", values, ids=ids)
    with pytest.raises(ValueError, match="ID set mismatch"):
        audit_predictions(source, {"bad": p}, "test", tmp_path / "audit")
    assert not (tmp_path / "audit").exists()


def test_duplicate_ids_rejected_in_prediction_and_manifest(manifest, tmp_path):
    source, rows = manifest
    p = predictions(tmp_path, "duplicate", [0, 0, 2, 3], ids=("a", "a", "c", "d"))
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        audit_predictions(source, {"bad": p}, "test", tmp_path / "audit")
    rows[-1]["sample_id"] = "a"
    write_csv(source, rows)
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        audit_predictions(source, None, "test", tmp_path / "audit")


def test_float32_labels_accepted_but_changed_labels_rejected(manifest, tmp_path):
    source, rows = manifest
    rows[0]["log10_kcat"] = 1.234567891234
    write_csv(source, rows)
    data = [{"sample_id": r["sample_id"], "target": float(np.float32(r["log10_kcat"])),
             "prediction": r["log10_kcat"]} for r in rows[:4]]
    p = write_csv(tmp_path / "pred.csv", data)
    report = audit_predictions(source, {"model": p}, "test", tmp_path / "ok")
    assert report["label_checks"]["model"]["max_absolute_label_difference"] > 0
    data[0]["target"] += 0.01
    write_csv(p, data)
    with pytest.raises(ValueError, match="label mismatch"):
        audit_predictions(source, {"model": p}, "test", tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_ambiguous_aliases_require_explicit_selection(manifest, tmp_path):
    source, rows = manifest
    p = write_csv(tmp_path / "pred.csv", [
        {"sample_id": r["sample_id"], "target": r["log10_kcat"], "y_true": 99,
         "prediction": r["log10_kcat"]} for r in rows[:4]
    ])
    with pytest.raises(ValueError, match="unambiguous"):
        audit_predictions(source, {"A": p}, "test", tmp_path / "bad")
    audit_predictions(source, {"A": p}, "test", tmp_path / "ok", prediction_target_column="target")


def test_cluster_bootstrap_is_reproducible_and_paired(manifest, tmp_path):
    source, _ = manifest
    a = predictions(tmp_path, "A", [0, 1, 2, 3])
    b = predictions(tmp_path, "B", [1, 2, 3, 4])
    reports = [audit_predictions(source, {"A": a, "B": b}, "test", tmp_path / name,
                                 bootstrap_repetitions=200, bootstrap_seed=19)
               for name in ["one", "two"]]
    assert reports[0]["paired_differences"] == reports[1]["paired_differences"]
    row = reports[0]["paired_differences"][0]
    assert row["bootstrap_status"] == "cluster_resampling"
    assert row["n_clusters"] == 3
    assert row["delta_mse_ci95_low"] == row["delta_mse_ci95_high"] == -1


def test_bootstrap_requires_complete_clusters(manifest, tmp_path):
    source, rows = manifest
    rows[0]["group_id"] = ""
    write_csv(source, rows)
    with pytest.raises(ValueError, match="complete, nonempty"):
        audit_predictions(source, None, "test", tmp_path / "bad", bootstrap_repetitions=10)
    report = audit_predictions(source, None, "test", tmp_path / "ok")
    assert report["coverage"][0]["missing_cluster_rows"] == 1


def test_one_cluster_does_not_claim_row_level_confidence(manifest, tmp_path):
    source, rows = manifest
    for row in rows[:4]:
        row["group_id"] = "single_correlated_family"
    write_csv(source, rows)
    a = predictions(tmp_path, "A", [0, 1, 2, 3])
    b = predictions(tmp_path, "B", [1, 2, 5, 9])
    report = audit_predictions(source, {"A": a, "B": b}, "test", tmp_path / "audit",
                               bootstrap_repetitions=200)
    overall = report["paired_differences"][0]
    assert overall["n"] == 4 and overall["n_clusters"] == 1
    assert overall["bootstrap_status"] == "insufficient_clusters"
    assert overall["delta_mse_ci95_low"] is None
    assert overall["delta_mse_ci95_high"] is None


def test_output_cannot_overwrite_inputs_or_prior_results(manifest, tmp_path):
    source, _ = manifest
    before = source.read_bytes()
    with pytest.raises(ValueError, match="ancestor"):
        audit_predictions(source, None, "test", tmp_path)
    output = tmp_path / "audit"
    audit_predictions(source, None, "test", output)
    with pytest.raises(ValueError, match="never overwrites"):
        audit_predictions(source, None, "test", output)
    assert source.read_bytes() == before


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "not_a_number"])
def test_nonfinite_or_invalid_predictions_rejected(manifest, tmp_path, value):
    source, _ = manifest
    p = predictions(tmp_path, "bad", [value, 1, 2, 3])
    with pytest.raises(ValueError, match="Nonfinite|Invalid numeric"):
        audit_predictions(source, {"A": p}, "test", tmp_path / "audit")
    assert not (tmp_path / "audit").exists()
