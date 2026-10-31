"""Data integrity tests for sequence replacements and the cached baseline."""

import csv
import json

import numpy as np
import pytest
import torch

from newkcat.model import ModelConfig, NewKcat
from newkcat.sequence.data import SequenceDataset, collate_sequences
from newkcat.sequence.tokenizer import BOS_ID, EOS_ID, PAD_ID, ReactionTokenizer


SR = "[CH3]{[CH2]|[CH]}{-|=}{[OH]|[O]}"
SPANS = [[5, 17], [17, 22], [22, 32]]


def sample_rows():
    return [
        {
            "sample_id": name, "sequence": sequence, "smiles": smiles,
            "log10_kcat": str(index - 1.25), "split": split,
            "group_id": f"cluster_{name}", "reaction_status": "ok",
            "reaction_smiles": "CCO>>CC=O", "sr_smiles": SR,
            "edit_spans": json.dumps(SPANS),
        }
        for index, (name, sequence, smiles, split) in enumerate([
            ("alpha", "MAAAAA", "CCO", "train"),
            ("beta", "MGGGGG", "CCO", "val"),
            ("gamma", "MWWWWW", "CCO", "test"),
        ])
    ]


def manifest(tmp_path, rows=None):
    rows = sample_rows() if rows is None else rows
    path = tmp_path / "samples.csv"
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return path


def cache(tmp_path, ids=("gamma", "alpha", "beta"), full=False, **arrays):
    path = tmp_path / "features.npz"
    numeric_id = {"alpha": 1.0, "beta": 2.0, "gamma": 3.0}
    contents = {
        "sample_ids": np.asarray(ids, dtype="U"),
        "protein": np.asarray([
            np.full(1024, numeric_id.get(name, 4.0), dtype=np.float32) for name in ids
        ]),
    }
    if full:
        contents["molt5"] = np.zeros((len(ids), 768), dtype=np.float32)
        contents["maccs"] = np.zeros((len(ids), 167), dtype=np.float32)
    contents.update(arrays)
    np.savez(path, **contents)
    return path


def test_protein_only_features_are_id_joined(tmp_path):
    dataset = SequenceDataset(manifest(tmp_path), cache(tmp_path), splits=["train", "val"])
    assert dataset.sample_ids == ["alpha", "beta"]
    assert dataset.rows is dataset.records
    assert len(dataset.all_records) == 3
    assert dataset.coverage == {"ok": 2}
    assert np.all(dataset[0]["protein"] == 1)
    assert np.all(dataset[1]["protein"] == 2)
    assert "molt5" not in dataset[0]
    assert "maccs" not in dataset[0]
    batch = collate_sequences([dataset[0], dataset[1]])
    assert batch.protein.shape == (2, 1024)
    assert batch.target.shape == (2,)
    assert batch.molt5 is None and batch.maccs is None
    assert batch.batch_size == 2
    assert batch.sample_ids == ("alpha", "beta")
    assert batch.to("cpu").statuses == ("ok", "ok")


def test_unselected_test_values_and_chemistry_are_not_materialized(tmp_path):
    rows = sample_rows()
    rows[-1].update(log10_kcat="not_a_label", sr_smiles="bad", edit_spans="not_json")
    protein = np.ones((3, 1024), dtype=np.float32)
    protein[0] = np.nan  # gamma is stored first in the shuffled feature archive.
    dataset = SequenceDataset(
        manifest(tmp_path, rows), cache(tmp_path, protein=protein), splits=["train", "val"]
    )
    assert len(dataset) == 2
    assert dataset[0]["target"] == -1.25


@pytest.mark.parametrize("field", ["sequence", "group_id"])
def test_manifest_leakage_checks_include_unselected_rows(tmp_path, field):
    rows = sample_rows()
    rows[-1][field] = rows[0][field]
    with pytest.raises(ValueError, match="leakage"):
        SequenceDataset(manifest(tmp_path, rows), cache(tmp_path), splits=["train"])


def test_baseline_does_not_require_a_graph_or_reaction(tmp_path, monkeypatch):
    import newkcat.sequence.chemistry as chemistry

    def forbidden(*args, **kwargs):
        raise AssertionError("Baseline must not call chemistry preprocessing")

    monkeypatch.setattr(chemistry, "normalize_substrate", forbidden)
    rows = sample_rows()
    rows[0].update(smiles="*", reaction_status="missing", sr_smiles="", reaction_smiles="")
    dataset = SequenceDataset(
        manifest(tmp_path, rows), cache(tmp_path, full=True),
        chemical_encoder="catapro_features", splits=["train", "val"],
    )
    assert dataset[0]["input_ids"] == [BOS_ID, EOS_ID]
    batch = collate_sequences([dataset[0], dataset[1]])
    assert batch.molt5.shape == (2, 768)
    assert batch.maccs.shape == (2, 167)
    model = NewKcat(ModelConfig(atom_dim=160, bond_dim=16, mode="baseline")).eval()
    with torch.no_grad():
        prediction = model(batch)
    assert prediction.shape == (2,) and torch.isfinite(prediction).all()


def test_baseline_requires_original_feature_families(tmp_path):
    with pytest.raises(ValueError, match="Feature NPZ"):
        SequenceDataset(
            manifest(tmp_path), cache(tmp_path), chemical_encoder="catapro_features"
        )


def test_wildcard_substrate_supported_without_graph_features(tmp_path):
    rows = sample_rows()
    rows[0]["smiles"] = "*CCO"
    dataset = SequenceDataset(
        manifest(tmp_path, rows), cache(tmp_path), chemical_encoder="substrate_smiles",
        splits=["train"],
    )
    assert "*" in dataset[0]["text"]
    assert not any(dataset[0]["edit_mask"])


def test_reaction_and_sr_share_explicit_substrate_fallback(tmp_path):
    rows = sample_rows()
    rows[0].update(
        smiles="*CCO", reaction_status="fallback_substrate", sr_smiles="", reaction_smiles=""
    )
    samples, features = manifest(tmp_path, rows), cache(tmp_path)
    with pytest.raises(ValueError, match="Reaction unavailable"):
        SequenceDataset(samples, features, splits=["train"])
    alternatives = [
        SequenceDataset(samples, features, chemical_encoder=encoder,
                        splits=["train"], missing_reaction="substrate")
        for encoder in ("reaction_smiles", "sr_smiles")
    ]
    assert alternatives[0][0]["input_ids"] == alternatives[1][0]["input_ids"]
    for dataset in alternatives:
        assert dataset.sample_ids == ["alpha"]
        assert dataset[0]["target"] == -1.25
        assert not any(dataset[0]["edit_mask"])
        assert dataset.coverage == {"fallback_substrate": 1}


def test_declared_ok_but_empty_text_is_not_fallback(tmp_path):
    rows = sample_rows()
    rows[0]["sr_smiles"] = ""
    with pytest.raises(ValueError, match="empty"):
        SequenceDataset(manifest(tmp_path, rows), cache(tmp_path), splits=["train"],
                        missing_reaction="substrate")


@pytest.mark.parametrize("spans", ["oops", "{}", "[[0,999]]", "[[2,1]]", "[[true,2]]",
                                   "[[0,2],[1,3]]", "[[4,5],[0,1]]"])
def test_malformed_edit_spans_do_not_trigger_fallback(tmp_path, spans):
    rows = sample_rows()
    rows[0]["edit_spans"] = spans
    with pytest.raises(ValueError, match="span"):
        SequenceDataset(manifest(tmp_path, rows), cache(tmp_path), splits=["train"],
                        missing_reaction="substrate")


def test_overlong_sequences_are_not_truncated(tmp_path):
    with pytest.raises(ValueError, match="length|max_length|long"):
        SequenceDataset(manifest(tmp_path), cache(tmp_path), max_length=5, splits=["train"])


def test_stale_prepared_substrate_is_rejected(tmp_path):
    rows = sample_rows()
    rows[0]["substrate_smiles"] = "CCC"
    with pytest.raises(ValueError, match="differs"):
        SequenceDataset(manifest(tmp_path, rows), cache(tmp_path),
                        chemical_encoder="substrate_smiles", splits=["train"])


def test_prediction_ignores_labels_and_has_no_target_tensor(tmp_path):
    rows = sample_rows()
    rows[0]["log10_kcat"] = ""
    rows[1]["log10_kcat"] = "not_used_for_prediction"
    dataset = SequenceDataset(manifest(tmp_path, rows), cache(tmp_path), require_targets=False)
    batch = collate_sequences([dataset[index] for index in range(len(dataset))])
    assert batch.target is None
    assert all(row["log10_kcat"] is None for row in dataset.records)


@pytest.mark.parametrize("bad_target", ["", "nan", "inf", "1e100", "not_a_number"])
def test_selected_training_targets_are_strict(tmp_path, bad_target):
    rows = sample_rows()
    rows[0]["log10_kcat"] = bad_target
    with pytest.raises(ValueError, match="target|log10_kcat"):
        SequenceDataset(manifest(tmp_path, rows), cache(tmp_path), splits=["train"])


@pytest.mark.parametrize("bad_protein", [
    np.zeros((3, 1023)), np.full((3, 1024), np.nan), np.full((3, 1024), 1e100),
    np.zeros((3, 1024), dtype=bool),
])
def test_invalid_protein_cache_is_fatal(tmp_path, bad_protein):
    with pytest.raises(ValueError, match="protein"):
        SequenceDataset(manifest(tmp_path), cache(tmp_path, protein=bad_protein), splits=["train"])


def test_duplicate_feature_ids_rejected(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        SequenceDataset(manifest(tmp_path), cache(tmp_path, ids=("alpha", "alpha")))


def test_missing_selected_feature_id_rejected(tmp_path):
    with pytest.raises(ValueError, match="missing feature"):
        SequenceDataset(manifest(tmp_path), cache(tmp_path, ids=("alpha",)), splits=["val"])


def test_padding_and_edit_masks_are_aligned():
    tokenizer = ReactionTokenizer()
    common = {"protein": np.ones(1024, dtype=np.float32), "target": 0.2, "status": "ok"}
    first = {**common, "sample_id": "short", **tokenizer.encode("CO", edit_spans=[[1, 2]])}
    second = {**common, "sample_id": "long", **tokenizer.encode("CCO>>CC=O")}
    batch = collate_sequences([first, second])
    assert batch.attention_mask[0].sum().item() == 4
    assert batch.input_ids[0, 4:].eq(PAD_ID).all()
    assert not batch.edit_mask[0, 4:].any()
    assert batch.edit_mask[0].nonzero().flatten().tolist() == [2]
    assert not batch.edit_mask[:, 0].any()


def test_collate_rejects_mixed_targets_and_feature_families():
    common = {"input_ids": [BOS_ID, 6, EOS_ID], "edit_mask": [False] * 3,
              "protein": np.zeros(1024, dtype=np.float32), "target": 0.0,
              "sample_id": "x", "status": "ok"}
    with pytest.raises(ValueError, match="mix labeled"):
        collate_sequences([common, {**common, "target": None}])
    with pytest.raises(ValueError, match="Inconsistent molt5"):
        collate_sequences([common, {**common, "molt5": np.zeros(768)}])


def test_reaction_control_does_not_receive_edit_labels(tmp_path):
    dataset = SequenceDataset(manifest(tmp_path), cache(tmp_path),
                              chemical_encoder="reaction_smiles", splits=["train"])
    assert not any(dataset[0]["edit_mask"])
    assert dataset[0]["text"] == "CCO>>CC=O"


def test_stale_but_in_range_edit_metadata_rejected(tmp_path):
    rows = sample_rows()
    rows[0]["edit_spans"] = "[[0,1]]"
    with pytest.raises(ValueError, match="do not match"):
        SequenceDataset(manifest(tmp_path, rows), cache(tmp_path), splits=["train"])


def test_atom_map_ids_cannot_become_reaction_features(tmp_path):
    rows = sample_rows()
    rows[0]["reaction_smiles"] = "[CH3:1][OH:2]>>[CH2:1]=[O:2]"
    with pytest.raises(ValueError, match="Atom-map"):
        SequenceDataset(manifest(tmp_path, rows), cache(tmp_path),
                        chemical_encoder="reaction_smiles", splits=["train"])


def test_status_typo_is_not_silently_fallback(tmp_path):
    rows = sample_rows()
    rows[0]["reaction_status"] = "ok_typo"
    with pytest.raises(ValueError, match="Unknown reaction_status"):
        SequenceDataset(manifest(tmp_path, rows), cache(tmp_path),
                        splits=["train"], missing_reaction="substrate")
