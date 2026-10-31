import csv
import json

import numpy as np
import pytest

from newkcat.data import KcatDataset
from newkcat.import_catapro import import_catapro


def source_files(tmp_path):
    rows = [
        {"": "a", "Sequence": "MACDEFGH", "Smiles": "CCO", "kcat(s^-1)": "100", "fold": "0"},
        {"": "b", "Sequence": "MACDEFGI", "Smiles": "CC", "kcat(s^-1)": "10", "fold": "1"},
        {"": "c", "Sequence": "MACDEFGK", "Smiles": "CO", "kcat(s^-1)": "1", "fold": "2"},
        {"": "d", "Sequence": "MACDEFGL", "Smiles": "CN", "kcat(s^-1)": "0.1", "fold": "2"},
    ]
    data = tmp_path / "original.csv"
    with data.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    features = tmp_path / "features.npz"
    # a's feature = 1, b's = 2, c's = 3, d's = 4, deliberately reversed.
    np.savez(features, sample_ids=np.array(["d", "c", "b", "a"]),
             protein=np.tile(np.array([4, 3, 2, 1])[:, None], (1, 1024)),
             molt5=np.zeros((4, 768)), maccs=np.zeros((4, 167)))
    return data, features


def test_preserves_ids_folds_logs_and_feature_alignment(tmp_path):
    data, features = source_files(tmp_path)
    output = tmp_path / "out"
    result = import_catapro(data, features, output, test_fold="0", val_fold="1")
    assert result["counts"] == {"train": 2, "val": 1, "test": 1}
    dataset = KcatDataset(output / "samples.csv", output / "features.npz", splits=["test"])
    assert dataset[0]["sample_id"] == "a"
    assert dataset[0]["target"] == 2
    assert dataset[0]["protein"][0] == 1
    assert json.loads((output / "import_manifest.json").read_text())["original_folds_preserved"]
    with pytest.raises(FileExistsError):
        import_catapro(data, features, output, test_fold="0", val_fold="1")


def test_rejects_overlapping_fold_before_writes(tmp_path):
    data, features = source_files(tmp_path)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="folds must differ"):
        import_catapro(data, features, output, test_fold="1.0", val_fold="1")
    assert not output.exists()


def test_pickle_requires_explicit_trust_before_read(tmp_path):
    data, _ = source_files(tmp_path)
    untrusted = tmp_path / "features.pkl"
    untrusted.write_bytes(b"do not deserialize this")
    with pytest.raises(ValueError, match="trust-pickle"):
        import_catapro(data, untrusted, tmp_path / "out", test_fold="0", val_fold="1")


def test_missing_feature_id_is_not_row_aligned(tmp_path):
    data, features = source_files(tmp_path)
    with np.load(features) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["sample_ids"] = np.array(["wrong", "c", "b", "a"])
    np.savez(features, **arrays)
    with pytest.raises(ValueError, match="lacks 1"):
        import_catapro(data, features, tmp_path / "out", test_fold="0", val_fold="1")


def test_official_pickle_aligns_column_labels_before_splitting_towers(tmp_path):
    pd = pytest.importorskip("pandas")
    data, _ = source_files(tmp_path)
    matrix = np.concatenate((np.ones((4, 1024)), np.full((4, 768), 2),
                             np.zeros((4, 167))), axis=1)
    frame = pd.DataFrame(matrix, index=["d", "c", "b", "a"])
    swapped = list(range(1024, 1792)) + list(range(1024)) + list(range(1792, 1959))
    cache = tmp_path / "official.pkl"
    frame[swapped].to_pickle(cache)
    output = tmp_path / "out"
    import_catapro(data, cache, output, test_fold="0", val_fold="1", trust_pickle=True)
    with np.load(output / "features.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["protein"], np.ones((4, 1024)))
        np.testing.assert_array_equal(archive["molt5"], np.full((4, 768), 2))


@pytest.mark.parametrize("kind", ["duplicate", "unlabeled"])
def test_official_pickle_rejects_ambiguous_column_schema(tmp_path, kind):
    pd = pytest.importorskip("pandas")
    data, _ = source_files(tmp_path)
    columns = list(range(1959))
    columns[0] = 1 if kind == "duplicate" else "protein_first"
    frame = pd.DataFrame(np.zeros((4, 1959)), index=["a", "b", "c", "d"], columns=columns)
    cache = tmp_path / "bad.pkl"
    frame.to_pickle(cache)
    with pytest.raises(ValueError, match="unique labels"):
        import_catapro(data, cache, tmp_path / "out", test_fold="0", val_fold="1", trust_pickle=True)
