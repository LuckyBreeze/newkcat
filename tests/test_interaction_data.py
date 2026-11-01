"""Cache/data behavior tests with deterministic tiny encoders (no downloads)."""

import csv
import json

import numpy as np
import pytest
import torch

from newkcat.interaction.data import (
    InteractionDataset,
    collate_interactions,
    create_cache,
    file_sha256,
    text_sha256,
)


class ProteinEncoder:
    metadata = {"name": "deterministic-test-protein", "revision": "v1"}

    def __init__(self):
        self.calls = []

    def encode(self, sequence):
        self.calls.append(sequence)
        states = np.arange(len(sequence) * 4, dtype=np.float32).reshape(len(sequence), 4)
        return {"residue_states": states, "metadata": {"raw_text_sha256": text_sha256(sequence)}}


class ReactionEncoder:
    def __init__(self):
        self.calls = []

    # Both a metadata method and a metadata property are supported.
    def metadata(self):
        return {"name": "deterministic-test-reaction", "revision": "v1"}

    def encode(self, text):
        self.calls.append(text)
        if text.startswith("CC"):
            roles = np.array([0, 0, 3, 2], dtype=np.int64)
            atom = np.array([True, True, False, True])
            edge_index = np.array([[0, 1, 3, 0, 1], [0, 1, 3, 1, 0]], dtype=np.int64)
            edge_types = np.array([1, 1, 1, 2, 2], dtype=np.int64)
        else:
            roles = np.array([0, 3, 2], dtype=np.int64)
            atom = np.array([True, False, True])
            edge_index = np.array([[0, 2], [0, 2]], dtype=np.int64)
            edge_types = np.array([1, 1], dtype=np.int64)
        return {
            "reaction_states": np.arange(len(roles) * 3, dtype=np.float32).reshape(len(roles), 3),
            "roles": roles, "atom_mask": atom, "topology_mask": atom.copy(),
            "edge_index": edge_index, "edge_types": edge_types,
            "metadata": {"raw_text_sha256": text_sha256(text)},
        }


def write_manifest(tmp_path, rows=None, name="samples.csv"):
    if rows is None:
        rows = [
            {"sample_id": "t1", "sequence": "ACDE", "smiles": "CCO",
             "reaction_smiles": "CCO>>CC=O", "log10_kcat": "1.25", "split": "train"},
            {"sample_id": "t2", "sequence": "ACDE", "smiles": "C",
             "reaction_smiles": "C>>C", "log10_kcat": "-0.5", "split": "train"},
            {"sample_id": "v1", "sequence": "GG", "smiles": "CCO",
             "reaction_smiles": "CCO>>CC=O", "log10_kcat": "bad-unselected", "split": "val"},
            {"sample_id": "x1", "sequence": "DEF", "smiles": "C",
             "reaction_smiles": "C>>C", "log10_kcat": "", "split": "test"},
        ]
    path = tmp_path / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def build(tmp_path):
    manifest = write_manifest(tmp_path)
    directory = tmp_path / "cache"
    protein, reaction = ProteinEncoder(), ReactionEncoder()
    index = create_cache(manifest, directory, protein, reaction)
    return manifest, directory, index, protein, reaction


def test_cache_preserves_rows_deduplicates_and_resumes_without_reencoding(tmp_path):
    manifest, cache, index, protein, reaction = build(tmp_path)
    assert list(index["samples"]) == ["t1", "t2", "v1", "x1"]
    assert len(protein.calls) == 3
    assert len(reaction.calls) == 2
    assert index["unique_counts"] == {"protein": 3, "reaction": 2}
    assert index["dimensions"] == {"protein": 4, "reaction": 3}
    assert index["samples"]["t1"]["protein"] == index["samples"]["t2"]["protein"]
    assert index["samples"]["t1"]["reaction"] == index["samples"]["v1"]["reaction"]
    assert index["samples"]["t1"]["reaction_sha256"] == text_sha256("CCO>>CC=O")
    for entry in index["samples"].values():
        for kind in ("protein", "reaction"):
            assert file_sha256(cache / entry[kind]) == entry[f"{kind}_file_sha256"]
            with np.load(cache / entry[kind], allow_pickle=False) as archive:
                assert archive["states"].dtype == np.float32
                assert archive["metadata"].dtype.kind == "U"
    again = create_cache(manifest, cache, protein, reaction)
    assert again == index
    assert len(protein.calls) == 3
    assert len(reaction.calls) == 2


def test_only_selected_targets_are_parsed_and_inference_ignores_labels(tmp_path):
    manifest, cache, *_ = build(tmp_path)
    train = InteractionDataset(manifest, cache)
    assert train.sample_ids == ["t1", "t2"]
    assert train.targets == [1.25, -0.5]
    with pytest.raises(ValueError, match="v1.*invalid log10_kcat"):
        InteractionDataset(manifest, cache, splits=["val"])
    inference = InteractionDataset(manifest, cache, splits=["val", "test"], require_targets=False)
    assert inference.sample_ids == ["v1", "x1"]
    assert inference.targets == [None, None]
    batch = collate_interactions([inference[0], inference[1]])
    assert batch["targets"] is None
    assert batch["protein_states"].shape == (2, 3, 4)
    assert batch["protein_mask"].tolist() == [[True, True, False], [True, True, True]]


def test_collate_preserves_tokens_roles_topology_and_mean_all_residues(tmp_path):
    manifest, cache, *_ = build(tmp_path)
    dataset = InteractionDataset(manifest, cache)
    first, second = dataset[0], dataset[1]
    np.testing.assert_equal(first["protein_global"], np.array([6, 7, 8, 9], np.float32))
    assert dataset.protein_global_source == "mean_all_residues"
    batch = collate_interactions([first, second])
    assert batch["sample_ids"] == ["t1", "t2"]
    assert batch["reaction_states"].shape == (2, 4, 3)
    assert batch["roles"].tolist() == [[0, 0, 3, 2], [0, 3, 2, 3]]
    assert batch["reaction_mask"].tolist() == [[True] * 4, [True, True, True, False]]
    assert batch["topology_mask"].tolist() == [[True, True, False, True], [True, False, True, False]]
    assert batch["bond_types"][0, 0, 1].item() == 2
    assert batch["bond_types"][0, 1, 0].item() == 2
    assert batch["bond_types"][0, 3, 3].item() == 1
    assert batch["bond_types"][0, 2].count_nonzero().item() == 0
    assert batch["bond_types"][1, 3].count_nonzero().item() == 0
    torch.testing.assert_close(batch["targets"], torch.tensor([1.25, -0.5]))


def test_external_global_features_join_by_id_and_only_validate_selected_rows(tmp_path):
    manifest, cache, *_ = build(tmp_path)
    global_path = tmp_path / "globals.npz"
    np.savez(global_path, sample_ids=np.array(["v1", "t2", "t1"]),
             protein=np.array([[np.nan] * 4, [20] * 4, [10] * 4], dtype=np.float32))
    dataset = InteractionDataset(manifest, cache, global_features=global_path)
    assert dataset.protein_global_source == "external_id_join"
    assert dataset[0]["protein_global"].tolist() == [10] * 4
    assert dataset[1]["protein_global"].tolist() == [20] * 4
    with pytest.raises(ValueError, match="nonfinite"):
        InteractionDataset(manifest, cache, splits=["val"], global_features=global_path,
                           require_targets=False)


def test_cache_lru_is_bounded_and_dataset_does_not_eagerly_load_states(tmp_path):
    manifest, cache, *_ = build(tmp_path)
    dataset = InteractionDataset(manifest, cache, splits=None, require_targets=False, cache_size=2)
    assert len(dataset._lru) == 0
    for index in range(len(dataset)):
        dataset[index]
        assert len(dataset._lru) <= 2
    no_cache = InteractionDataset(manifest, cache, cache_size=0)
    no_cache[0]
    assert len(no_cache._lru) == 0


def test_changed_input_or_missing_sample_never_silently_reuses_cache(tmp_path):
    manifest, cache, index, *_ = build(tmp_path)
    text = manifest.read_text().replace("CCO>>CC=O", "CCO>>COC", 1)
    changed = tmp_path / "changed.csv"
    changed.write_text(text)
    with pytest.raises(ValueError, match="cached reaction_smiles does not match"):
        InteractionDataset(changed, cache)
    del index["samples"]["t1"]
    (cache / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="t1.*missing"):
        InteractionDataset(manifest, cache)


def test_altered_archive_rejected_before_model_receives_features(tmp_path):
    manifest, cache, index, protein, reaction = build(tmp_path)
    protein_path = cache / index["samples"]["t1"]["protein"]
    with protein_path.open("ab") as handle:
        handle.write(b"altered")
    dataset = InteractionDataset(manifest, cache)
    with pytest.raises(ValueError, match="checksum mismatch"):
        dataset[0]
    with pytest.raises(ValueError, match="checksum mismatch"):
        create_cache(manifest, cache, protein, reaction)


@pytest.mark.parametrize("mode", ["nan", "pooled", "dimension"])
def test_invalid_protein_outputs_fail_without_publishing_partial_index(tmp_path, mode):
    class BadProtein(ProteinEncoder):
        def encode(self, sequence):
            result = super().encode(sequence)
            if mode == "nan":
                result["residue_states"][0, 0] = np.nan
            elif mode == "pooled":
                result["residue_states"] = result["residue_states"].mean(axis=0, keepdims=True)
            elif sequence == "GG":
                result["residue_states"] = np.zeros((len(sequence), 5), np.float32)
            return result

    manifest = write_manifest(tmp_path)
    output = tmp_path / "bad-cache"
    match = {"nan": "nonfinite", "pooled": "one row per", "dimension": "dimension"}[mode]
    with pytest.raises(ValueError, match=match):
        create_cache(manifest, output, BadProtein(), ReactionEncoder())
    assert not (output / "index.json").exists()


def test_explicit_protein_import_reuses_exact_per_sequence_arrays(tmp_path):
    manifest = write_manifest(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    entries = {}
    for sequence in ("ACDE", "GG", "DEF"):
        sha = text_sha256(sequence)
        filename = f"{sha}.npz"
        np.savez(source / filename, residue_features=np.full((len(sequence), 4), len(sequence), np.float32))
        entries[sha] = {"path": filename, "array_key": "residue_features",
                        "sha256": file_sha256(source / filename)}
    import_index = source / "index.json"
    import_index.write_text(json.dumps({
        "version": 1, "encoders": {"protein": ProteinEncoder.metadata}, "sequences": entries,
    }))
    output = tmp_path / "imported-cache"
    create_cache(manifest, output, None, ReactionEncoder(), protein_import_index=import_index)
    dataset = InteractionDataset(manifest, output)
    assert dataset[0]["protein_states"].tolist() == [[4] * 4] * 4
    # An existing interaction cache's samples index is also an explicit import.
    next_output = tmp_path / "next-cache"
    create_cache(manifest, next_output, None, ReactionEncoder(), protein_import_index=output)
    np.testing.assert_array_equal(
        InteractionDataset(manifest, next_output)[0]["protein_states"], dataset[0]["protein_states"]
    )


def test_import_does_not_guess_missing_sequence_or_accept_pooled_features(tmp_path):
    manifest = write_manifest(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    np.savez(source / "global.npz", states=np.zeros((1, 4), dtype=np.float32))
    import_index = source / "index.json"
    import_index.write_text(json.dumps({
        "version": 1, "encoders": {"protein": ProteinEncoder.metadata},
        "sequences": {text_sha256("ACDE"): {"path": "global.npz"}},
    }))
    with pytest.raises(ValueError, match="one row per"):
        create_cache(manifest, tmp_path / "cache", None, ReactionEncoder(),
                     protein_import_index=import_index)


def test_cache_provenance_paths_and_split_checks(tmp_path):
    manifest, cache, index, protein, reaction = build(tmp_path)
    protein.metadata = {"name": "different-model"}
    with pytest.raises(ValueError, match="different encoders"):
        create_cache(manifest, cache, protein, reaction)
    index["samples"]["t1"]["protein"] = "../samples.csv"
    (cache / "index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="inside its index"):
        InteractionDataset(manifest, cache)
    duplicate_sequence = tmp_path / "leaked.csv"
    duplicate_sequence.write_text(manifest.read_text().replace(",GG,", ",ACDE,"))
    with pytest.raises(ValueError, match="Sequence leakage"):
        InteractionDataset(duplicate_sequence, cache)


def test_missing_topology_keeps_tokens_and_does_not_fabricate_edges(tmp_path):
    class NoTopology(ReactionEncoder):
        def encode(self, text):
            result = super().encode(text)
            result["topology_mask"] = np.zeros_like(result["atom_mask"])
            result["edge_index"] = np.empty((2, 0), dtype=np.int64)
            result["edge_types"] = np.empty(0, dtype=np.int64)
            return result

    manifest = write_manifest(tmp_path)
    cache = tmp_path / "cache"
    create_cache(manifest, cache, ProteinEncoder(), NoTopology())
    dataset = InteractionDataset(manifest, cache)
    batch = collate_interactions([dataset[0], dataset[1]])
    assert batch["reaction_mask"].sum().item() == 7
    assert batch["atom_mask"].sum().item() == 5
    assert batch["topology_mask"].count_nonzero().item() == 0
    assert batch["bond_types"].count_nonzero().item() == 0


def test_invalid_directed_topology_rejected(tmp_path):
    class BadTopology(ReactionEncoder):
        def encode(self, text):
            result = super().encode(text)
            # Remove the last reverse edge, preserving the self edges.
            result["edge_index"] = result["edge_index"][:, :-1]
            result["edge_types"] = result["edge_types"][:-1]
            return result

    with pytest.raises(ValueError, match="reverse edges"):
        create_cache(write_manifest(tmp_path), tmp_path / "cache", ProteinEncoder(), BadTopology())
