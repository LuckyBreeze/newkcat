import json

import numpy as np
import pytest
from rdkit import Chem

from newkcat.centers import (
    load_center_records,
    reference_center_from_reaction,
    reference_centers_from_csv,
    validate_center_record,
)
from newkcat.chem import (
    ATOM_DIM,
    BOND_DIM,
    featurize_mol,
    featurize_smiles,
    molecular_identity,
    parse_smiles,
)


def _edge_attributes(graph):
    return {tuple(edge): feature for edge, feature in zip(
        graph.edge_index.T.tolist(), graph.edge_features
    )}


@pytest.mark.parametrize("smiles", [
    "F[C@H](Cl)C/C=C/Br", "C[C@@H](O)[C@H](N)C(=O)O", "c1ccccc1.O", "[Na+]",
])
def test_renumbering_preserves_atom_bond_and_distance_features(smiles):
    mol = parse_smiles(smiles)
    original = featurize_mol(mol)
    permutation = np.random.default_rng(7).permutation(mol.GetNumAtoms()).tolist()
    reordered = featurize_mol(Chem.RenumberAtoms(mol, permutation))
    np.testing.assert_array_equal(original.atom_features[permutation], reordered.atom_features)
    np.testing.assert_array_equal(
        original.distances[np.ix_(permutation, permutation)], reordered.distances
    )
    original_edges = _edge_attributes(original)
    for (i, j), features in _edge_attributes(reordered).items():
        np.testing.assert_array_equal(features, original_edges[permutation[i], permutation[j]])
        np.testing.assert_array_equal(features, _edge_attributes(reordered)[j, i])
    assert reordered.identity == original.identity


def test_atom_map_identifiers_are_never_chemical_features():
    plain = featurize_smiles("F[C@H](Cl)Br")
    mapped = featurize_smiles("[F:88][C@H:2]([Cl:600])[Br:7]")
    np.testing.assert_array_equal(plain.atom_features, mapped.atom_features)
    np.testing.assert_array_equal(plain.edge_features, mapped.edge_features)
    assert mapped.atom_maps == (88, 2, 600, 7)


def test_enantiomers_have_different_absolute_stereo_features():
    left = featurize_smiles("F[C@H](Cl)Br")
    right = featurize_smiles("F[C@@H](Cl)Br")
    assert not np.array_equal(left.atom_features, right.atom_features)
    assert left.identity != right.identity


def test_disconnected_graph_and_single_atom_shapes():
    graph = featurize_smiles("CC.O")
    np.testing.assert_array_equal(graph.distances, [[0, 1, -1], [1, 0, -1], [-1, -1, 0]])
    singleton = featurize_smiles("[Na+]")
    assert singleton.atom_features.shape == (1, ATOM_DIM)
    assert singleton.edge_index.shape == (2, 0)
    assert singleton.edge_features.shape == (0, BOND_DIM)


def _center(**updates):
    result = {
        "sample_id": "s", "mapped_smiles": "[CH3:9][CH2:7][OH:12]",
        "scores": {"12": 0.8, "9": 0.1, "7": 0.2},
        "kind": "predicted", "source": "unit-test", "uses_ec": False,
    }
    result.update(updates)
    return result


def test_scores_follow_map_ids_and_actual_mapped_smiles_order():
    annotation = validate_center_record(_center(), "OCC")
    np.testing.assert_allclose(annotation.scores, [0.1, 0.2, 0.8])
    annotation = validate_center_record(
        _center(mapped_smiles="[OH:12][CH2:7][CH3:9]"), "CCO"
    )
    np.testing.assert_allclose(annotation.scores, [0.8, 0.2, 0.1])


@pytest.mark.parametrize("update", [
    {"scores": {"9": 0.1, "7": 0.2}},
    {"scores": {"09": 0.1, "7": 0.2, "12": 0.3}},
    {"scores": {"9": 0.1, "7": 0.2, "12": float("nan")}},
    {"scores": {"9": 0.1, "7": 0.2, "12": 1.01}},
    {"scores": {"9": 0.1, "7": 0.2, "12": -0.1}},
    {"scores": {"9": 0.1, "7": 0.2, "12": True}},
    {"scores": {"9": 0.1, "7": 0.2, "12": "0.5"}},
    {"mapped_smiles": "[CH3:9][CH2:9][OH:12]"},
    {"mapped_smiles": "C[CH2:7][OH:12]"},
    {"mapped_smiles": "[13CH3:9][CH2:7][OH:12]"},
    {"mapped_smiles": "[CH3:9][CH2:7][O-:12]"},
    {"uses_ec": "false"}, {"source": ""}, {"kind": "reference"},
])
def test_invalid_center_records_fail(update):
    with pytest.raises(ValueError):
        validate_center_record(_center(**update), "CCO")


def test_center_stereo_identity_is_strict():
    center = _center(
        mapped_smiles="[F:1][C@H:2]([Cl:3])[Br:4]",
        scores={str(i): 0.5 for i in range(1, 5)},
    )
    validate_center_record(center, "F[C@H](Cl)Br")
    with pytest.raises(ValueError, match="stereochemistry"):
        validate_center_record(center, "F[C@@H](Cl)Br")
    with pytest.raises(ValueError, match="stereochemistry"):
        validate_center_record(center, "FC(Cl)Br")
    assert molecular_identity("[H]OC") == molecular_identity("CO")
    assert molecular_identity("[2H]OC") != molecular_identity("CO")


@pytest.mark.parametrize("substrate,mapped,scores", [
    ("CO", "[CH3:1][O:2][H:3]", {"1": 0, "2": 1, "3": 0}),
    ("[H]OC", "[OH:2][CH3:1]", {"1": 0, "2": 1}),
])
def test_center_files_cannot_change_explicit_hydrogen_graphs(substrate, mapped, scores):
    assert molecular_identity(substrate) == molecular_identity(mapped)
    with pytest.raises(ValueError, match="explicit-hydrogen graph"):
        validate_center_record(_center(mapped_smiles=mapped, scores=scores), substrate)


@pytest.mark.parametrize("smiles", [
    "C[C@H](F)Cl |o1:1|", "C[C@H](F)Cl |&1:1|", "C[C@H](F)Cl |a:1|",
])
def test_unsupported_enhanced_stereo_cannot_collapse_to_absolute_identity(smiles):
    with pytest.raises(ValueError, match="Enhanced CXSMILES"):
        molecular_identity(smiles)
    with pytest.raises(ValueError, match="Enhanced CXSMILES"):
        molecular_identity(Chem.MolFromSmiles(smiles))


def test_non_tetrahedral_stereo_is_not_silently_dropped():
    with pytest.raises(ValueError, match="Only tetrahedral"):
        parse_smiles("[Pt@SP1](Cl)(F)(Br)I")


def test_duplicate_json_keys_and_duplicate_ids_fail(tmp_path):
    path = tmp_path / "centers.jsonl"
    path.write_text('{"sample_id":"s", "sample_id":"t"}\n')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        load_center_records(path)
    path.write_text((json.dumps(_center()) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate center record"):
        load_center_records(path)
    path.write_text(json.dumps(_center(sample_id=" s ")) + "\n")
    with pytest.raises(ValueError, match="surrounding whitespace"):
        load_center_records(path)


def test_reference_center_tracks_changed_bonds_and_hydrogen_counts():
    record = reference_center_from_reaction(
        "oxidation", "CCO", "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]",
        source="curated-reference:1",
    )
    assert record["scores"] == {"1": 0, "2": 1, "3": 1}
    assert record["kind"] == "reference"
    assert record["source"] == "curated-reference:1"
    assert not record["uses_ec"]


def test_remote_cip_priority_flip_does_not_mark_unchanged_tetrahedral_center():
    reactant = "[F:6][C@H:1]([CH2:2][OH:3])[CH2:4][Br:5]"
    product = "[F:6][C@H:1]([CH2:2][Br:5])[CH2:4][OH:3]"
    substrate = molecular_identity(reactant)
    for _ in range(12):
        # Random SMILES traversal changes CW/CCW tags; the map-relative state
        # must remain fixed, even when distal edits reverse absolute R/S.
        r = Chem.MolToSmiles(parse_smiles(reactant), doRandom=True, canonical=False)
        p = Chem.MolToSmiles(parse_smiles(product), doRandom=True, canonical=False)
        record = reference_center_from_reaction("s", substrate, r + ">>" + p)
        assert record["scores"] == {"1": 0, "2": 1, "3": 1, "4": 1, "5": 1, "6": 0}


def test_reference_center_detects_actual_tetrahedral_inversion():
    reactant = "[F:6][C@H:1]([CH2:2][OH:3])[CH2:4][Br:5]"
    product = reactant.replace("[C@H:1]", "[C@@H:1]")
    record = reference_center_from_reaction("s", molecular_identity(reactant), reactant + ">>" + product)
    assert record["scores"]["1"] == 1
    assert sum(record["scores"].values()) == 1


def test_remote_priority_flip_does_not_mark_unchanged_double_bond():
    reactant = "[OH:1][CH2:2]/[C:3]([CH2:4][Br:5])=[CH:6]/[CH3:7]"
    product = "[Br:5][CH2:2]/[C:3]([CH2:4][OH:1])=[CH:6]/[CH3:7]"
    substrate = molecular_identity(reactant)
    for _ in range(12):
        r = Chem.MolToSmiles(parse_smiles(reactant), doRandom=True, canonical=False)
        p = Chem.MolToSmiles(parse_smiles(product), doRandom=True, canonical=False)
        record = reference_center_from_reaction("s", substrate, r + ">>" + p)
        assert record["scores"] == {"1": 1, "2": 1, "3": 0, "4": 1, "5": 1, "6": 0, "7": 0}


def test_reference_center_detects_actual_cis_trans_change():
    reactant = "[OH:1][CH2:2]/[C:3]([CH2:4][Br:5])=[CH:6]/[CH3:7]"
    product = reactant.replace("=[CH:6]/", "=[CH:6]\\")
    record = reference_center_from_reaction("s", molecular_identity(reactant), reactant + ">>" + product)
    assert record["scores"]["3"] == record["scores"]["6"] == 1
    assert sum(record["scores"].values()) == 2


@pytest.mark.parametrize("reaction,match", [
    ("[CH3:1][CH2:2][OH:3]>>[CH4:1].[CH4:2]", "Incomplete atom correspondence"),
    ("[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][NH2:3]", "changes element"),
    ("[CH3:1][CH2:2][OH:3]>>[CH3:1][CH2:2][OH:3]", "No net structural change"),
    ("[CH3:1][CH:2]=[O:3]>>[CH3:1][CH2:2][OH:3]", "found 0"),
    ("[CH3:1][CH2:2][OH:2]>>[CH3:1][CH:2]=[O:3]", "duplicate atom-map"),
])
def test_reference_extraction_fails_on_incomplete_ambiguous_or_wrong_direction(reaction, match):
    with pytest.raises(ValueError, match=match):
        reference_center_from_reaction("s", "CCO", reaction)


def test_duplicate_substrates_require_explicit_map_selection():
    reaction = (
        "[CH3:1][CH2:2][OH:3].[CH3:4][CH2:5][OH:6]"
        ">>[CH3:1][CH:2]=[O:3].[CH3:4][CH2:5][OH:6]"
    )
    with pytest.raises(ValueError, match="found 2"):
        reference_center_from_reaction("s", "CCO", reaction)
    result = reference_center_from_reaction("s", "CCO", reaction, substrate_maps=[1, 2, 3])
    assert result["scores"] == {"1": 0, "2": 1, "3": 1}
    with pytest.raises(ValueError, match="complete reactant components"):
        reference_center_from_reaction("s", "C", reaction, substrate_maps=[1])


def test_reference_export_is_transactional_and_never_implicitly_overwrites(tmp_path):
    samples = tmp_path / "reactions.csv"
    output = tmp_path / "centers.jsonl"
    samples.write_text(
        "sample_id,smiles,reaction_smiles\n"
        "s1,CCO,[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]\n"
        "s2,CCO,invalid\n"
    )
    with pytest.raises(ValueError, match="line 3"):
        reference_centers_from_csv(samples, output)
    assert not output.exists()
    samples.write_text(samples.read_text().split("s2,")[0])
    assert reference_centers_from_csv(samples, output) == 1
    with pytest.raises(FileExistsError):
        reference_centers_from_csv(samples, output)
