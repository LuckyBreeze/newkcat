import csv
import json

import numpy as np
import pytest

from newkcat.data import KcatDataset, collate_graphs, ec2_token, fit_ec_vocab


def _rows():
    return [
        dict(sample_id="a", sequence="MAC", smiles="CCO", log10_kcat="1.2", split="train",
             ec="1.1.1.1", group_id="g1"),
        dict(sample_id="b", sequence="MAD", smiles="CC", log10_kcat="0.4", split="train",
             ec="2.1.1.1", group_id="g2"),
        dict(sample_id="c", sequence="MAE", smiles="CO", log10_kcat="-0.5", split="val",
             ec="9.9.1.1", group_id="g3"),
        dict(sample_id="d", sequence="MAF", smiles="O", log10_kcat="0.9", split="test",
             ec="8.8.1.1", group_id="g4"),
    ]


def _write_samples(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _files(tmp_path, rows=None, feature_ids=None, center_records=None, feature_updates=None):
    rows = _rows() if rows is None else rows
    samples = tmp_path / "samples.csv"
    _write_samples(samples, rows)
    ids = [row["sample_id"] for row in rows] if feature_ids is None else feature_ids
    arrays = {
        "sample_ids": np.array(ids, dtype=str),
        "protein": np.stack([np.full(1024, i, np.float32) for i in range(len(ids))]),
        "molt5": np.stack([np.full(768, i + 10, np.float32) for i in range(len(ids))]),
        "maccs": np.zeros((len(ids), 167), np.float32),
    }
    arrays.update(feature_updates or {})
    features = tmp_path / "features.npz"
    np.savez(features, **arrays)
    centers = None
    if center_records is not None:
        centers = tmp_path / "centers.jsonl"
        centers.write_text("".join(json.dumps(record) + "\n" for record in center_records))
    return samples, features, centers


def _center(sample_id="a", **updates):
    record = dict(
        sample_id=sample_id, mapped_smiles="[OH:30][CH2:20][CH3:10]",
        scores={"10": 0, "20": 0.25, "30": 1}, kind="predicted", source="test", uses_ec=False,
    )
    record.update(updates)
    return record


def test_feature_rows_are_joined_by_id_not_csv_order(tmp_path):
    files = _files(tmp_path, feature_ids=["d", "b", "extra", "a", "c"])
    dataset = KcatDataset(*files, splits=["train"])
    assert dataset.sample_ids == ["a", "b"]
    np.testing.assert_array_equal(dataset[0]["protein"], np.full(1024, 3))
    np.testing.assert_array_equal(dataset[1]["protein"], np.full(1024, 1))
    assert dataset.records[0]["log10_kcat"] == 1.2
    assert dataset.all_records[0]["log10_kcat"] == "1.2"


def test_center_map_order_and_missing_versus_zero(tmp_path):
    records = [_center(scores={"10": 0, "20": 0, "30": 0})]
    dataset = KcatDataset(*_files(tmp_path, center_records=records), splits=["train"])
    assert dataset[0]["graph"].atom_maps == (30, 20, 10)
    assert dataset[0]["center_available"]
    assert not dataset[1]["center_available"]
    batch = collate_graphs([dataset[0], dataset[1]])
    assert batch.center_kinds == ("predicted", "missing")
    assert batch.center_available.tolist() == [True, False]
    assert not batch.center_scores.any()


def test_selected_centers_are_aligned_and_information_flags_set(tmp_path):
    records = [_center(kind="reference", uses_ec=True, scores={"10": 0, "20": 1, "30": 1})]
    dataset = KcatDataset(*_files(tmp_path, center_records=records), splits=["train"])
    np.testing.assert_equal(dataset[0]["center_scores"], [1, 1, 0])
    assert dataset.center_uses_ec
    assert dataset.has_reference_centers
    assert dataset[0]["center_uses_ec"]
    assert dataset[0]["center_source"] == "test"
    assert not dataset[1]["center_uses_ec"]
    assert dataset[1]["center_source"] == "missing"


def test_train_does_not_parse_test_targets_centers_or_feature_values(tmp_path):
    rows = _rows()
    rows[-1].update(log10_kcat="not-a-target", smiles="not-a-molecule")
    protein = np.ones((4, 1024), np.float32)
    protein[-1] = np.nan
    maccs = np.zeros((4, 167), np.float32)
    maccs[-1] = np.nan
    files = _files(
        tmp_path, rows=rows, center_records=[{"sample_id": "d", "mapped_smiles": "invalid"}],
        feature_updates={"protein": protein, "maccs": maccs},
    )
    dataset = KcatDataset(*files, splits=["train"])
    assert len(dataset) == 2
    assert dataset.ec_vocab == {"1.1": 1, "2.1": 2}
    assert not dataset.center_uses_ec and not dataset.has_reference_centers
    with pytest.raises(ValueError, match="nonfinite"):
        KcatDataset(*files, splits=["test"], ec_vocab=dataset.ec_vocab)


def test_ec_vocabulary_never_fits_validation_or_test(tmp_path):
    files = _files(tmp_path)
    train = KcatDataset(*files, splits=["train"])
    validation = KcatDataset(*files, splits=["val"], ec_vocab=train.ec_vocab)
    assert validation[0]["ec2"] == 0
    assert KcatDataset(*files, splits=["val"]).ec_vocab == {}
    assert KcatDataset(*files).ec_vocab == train.ec_vocab
    assert fit_ec_vocab(_rows()) == train.ec_vocab
    assert ec2_token("EC 1.1.-.-") == "1.1"
    assert ec2_token("1.1.1.1;2.1.1.1") is None


@pytest.mark.parametrize("field,value,match", [
    ("sequence", "m a c", "Sequence leakage"),
    ("group_id", "g1", "Group leakage"),
    ("sample_id", "a", "Duplicate sample_id"),
    ("split", "dev", "split must"),
])
def test_global_split_leakage_and_invalid_metadata_fail(tmp_path, field, value, match):
    rows = _rows()
    rows[-1][field] = value
    files = _files(tmp_path, rows=rows, feature_ids=["a", "b", "c", "d"])
    with pytest.raises(ValueError, match=match):
        KcatDataset(*files, splits=["train"])


def test_same_enzyme_different_smiles_spelling_still_cannot_cross_splits(tmp_path):
    rows = _rows()
    rows[-1].update(sequence=rows[0]["sequence"], smiles="OCC")
    with pytest.raises(ValueError, match="Sequence leakage"):
        KcatDataset(*_files(tmp_path, rows=rows), splits=["train"])


@pytest.mark.parametrize("ids,match", [
    (["a", "a", "c", "d"], "Duplicate sample_ids"),
    (["b", "c", "d"], "missing feature rows"),
])
def test_bad_feature_join_fails(tmp_path, ids, match):
    with pytest.raises(ValueError, match=match):
        KcatDataset(*_files(tmp_path, feature_ids=ids), splits=["train"])


@pytest.mark.parametrize("updates,match", [
    ({"protein": np.full((4, 1024), np.nan)}, "nonfinite"),
    ({"maccs": np.ones((4, 167))}, "bit 0"),
    ({"maccs": np.zeros((4, 166))}, "shape"),
    ({"molt5": np.full((4, 768), 1e100)}, "overflow"),
    ({"protein": np.ones((4, 1024), dtype=complex)}, "real numeric"),
    ({"sample_ids": np.array(["a", "b", "c", "d"], dtype=object)}, "Object arrays"),
])
def test_invalid_selected_feature_arrays_fail(tmp_path, updates, match):
    with pytest.raises(ValueError, match=match):
        KcatDataset(*_files(tmp_path, feature_updates=updates), splits=["train"])


def test_missing_targets_are_only_allowed_for_inference(tmp_path):
    rows = _rows()
    rows[0]["log10_kcat"] = ""
    files = _files(tmp_path, rows=rows)
    with pytest.raises(ValueError, match="missing log10_kcat"):
        KcatDataset(*files, splits=["train"])
    dataset = KcatDataset(*files, splits=["train"], require_targets=False)
    assert dataset[0]["target"] is None
    assert collate_graphs([dataset[0], dataset[1]]).target is None


def test_collate_offsets_by_padded_width_not_cumulative_atoms(tmp_path):
    dataset = KcatDataset(*_files(tmp_path), splits=["train"])
    batch = collate_graphs([dataset[1], dataset[0]])
    assert batch.atom_features.shape[:2] == (2, 3)
    assert batch.atom_mask.tolist() == [[True, True, False], [True, True, True]]
    assert batch.edge_index[:, :2].tolist() == [[0, 1], [1, 0]]
    assert batch.edge_index[:, 2:].min().item() == 3
    assert batch.edge_index[:, 2:].max().item() == 5
    assert batch.distances[0, 2].tolist() == [-1, -1, -1]
    assert batch.target.shape == (2,)
    assert batch.sample_ids == ("b", "a")
