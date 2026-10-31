"""The optional converter must preserve chemistry instead of silently repairing it."""

import importlib.util
import re

import pytest

from newkcat.sequence import chemistry
from newkcat.sequence.chemistry import (
    ReactionPreparationError, assert_map_free, dynamic_spans, normalize_reaction,
    normalize_substrate, prepare_reaction,
)


HAS_SR = importlib.util.find_spec("sr_smiles") is not None
requires_sr = pytest.mark.skipif(not HAS_SR, reason="Optional sr-smiles==0.2.0 is not installed")
OXIDATION = "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]"
WILDCARD = "[*:1][CH2:2][Br:3].[OH-:4]>>[*:1][CH2:2][OH:4].[Br-:3]"
IDENTITY = "[CH3:1][OH:2]>>[CH3:1][OH:2]"


def test_substrate_normalization_allows_dummy_nodes_without_completing_them():
    assert normalize_substrate("[*:9][CH2:4][OH:7]") == "*CO"
    assert normalize_substrate("O.CC") == normalize_substrate("CC.O")
    assert normalize_substrate("[2H]O") != normalize_substrate("[H]O")
    assert normalize_substrate("[H]O") != normalize_substrate("O")
    assert normalize_substrate("[NH4+]") != normalize_substrate("N")
    assert normalize_substrate("F[C@H](Cl)Br") != normalize_substrate("F[C@@H](Cl)Br")


@pytest.mark.parametrize("smiles", ["", None, "C C", "CC ethanol", "C..O", "C(", "CO |a:1|"])
def test_substrate_rejects_bad_or_annotated_text(smiles):
    with pytest.raises(ReactionPreparationError):
        normalize_substrate(smiles)


def test_full_reaction_canonicalization_preserves_direction_and_multiplicity():
    assert normalize_reaction("[OH2:3].[CH4:1]>>[CH3:1][OH:3]") == "C.O>>CO"
    assert normalize_reaction("C.C>>CC") != normalize_reaction("C>>CC")
    assert normalize_reaction("C>>CO") != normalize_reaction("CO>>C")


@pytest.mark.parametrize("reaction", [
    "CCO", "CCO>O>CC=O", ">>O", "C>>", "C>>O>>N", "C >>O",
])
def test_invalid_reaction_format(reaction):
    with pytest.raises(ReactionPreparationError):
        normalize_reaction(reaction)


@pytest.mark.parametrize("reaction, message", [
    ("[CH3:1]O>>[CH3:1][OH:2]", "unmapped"),
    ("[CH3:1][OH:1]>>[CH3:1][OH:2]", "Duplicate"),
    ("[CH4:1]>>[CH3:1][OH:2]", "map sets"),
    ("[CH4:1]>>[NH3:1]", "element or isotope"),
    ("[13CH4:1]>>[CH4:1]", "element or isotope"),
    ("[H][OH:1]>>[H][OH:1]", "unmapped"),
])
def test_invalid_maps_are_rejected_before_converter_loading(monkeypatch, reaction, message):
    def fail_load():
        raise AssertionError("A mapper/converter must not repair this input")
    monkeypatch.setattr(chemistry, "_load_converter", fail_load)
    with pytest.raises(ReactionPreparationError, match=message):
        prepare_reaction(reaction)


def test_dynamic_field_parser_and_ring_closure():
    text = "[C]1{=|-}[C]{~|-}[H]{-|~}[H]{~|-}1"
    spans = dynamic_spans(text)
    assert [text[start:end] for start, end in spans] == ["{=|-}", "{~|-}", "{-|~}", "{~|-}"]
    text = "{[C@@]|[C@]}([H])([CH3]){~|-}{[O-]|[O]}"
    spans = dynamic_spans(text)
    assert [text[start:end] for start, end in spans] == ["{[C@@]|[C@]}", "{~|-}", "{[O-]|[O]}"]
    assert dynamic_spans("[c]1:[cH]:[cH]:[cH]:[cH]:[cH]:1") == []


@pytest.mark.parametrize("text", [
    "C{~|-", "C{~||-}O", "C{{~|-}|=}O", "C{-|-}O", "C{|=}O", "C{C|-}O",
    "C|O", "C}O", "C]O", "[C:7]{-|=}[O]", "{[C:7]|[C:7+]}O", "[C{foo}]",
])
def test_dynamic_parser_rejects_unsupported_or_mapped_fields(text):
    with pytest.raises(ReactionPreparationError):
        dynamic_spans(text)


def test_map_free_check_distinguishes_aromatic_ring_bonds():
    assert_map_free("[c]1:[cH]:[cH]:[cH]:[cH]:[cH]:1")
    for text in ("[C:1]>>[C:1]", "{[C:1^]|[C:1]}O", "[C:1"):
        with pytest.raises(ReactionPreparationError):
            assert_map_free(text)


def test_missing_optional_dependency_is_not_a_recoverable_chemical_error(monkeypatch):
    def absent(_):
        raise chemistry.importlib.metadata.PackageNotFoundError("sr-smiles")
    monkeypatch.setattr(chemistry.importlib.metadata, "version", absent)
    with pytest.raises(ImportError, match="optional"):
        prepare_reaction(IDENTITY)


def test_wrong_converter_version_is_explicit(monkeypatch):
    monkeypatch.setattr(chemistry.importlib.metadata, "version", lambda _: "9.9.9")
    with pytest.raises(RuntimeError, match="0.2.0"):
        prepare_reaction(IDENTITY)


def test_upstream_empty_result_is_failure(monkeypatch):
    monkeypatch.setattr(chemistry, "_load_converter", lambda: (
        lambda *a, **kw: "", lambda *a, **kw: "", "0.2.0"
    ))
    with pytest.raises(ReactionPreparationError, match="empty output"):
        prepare_reaction(IDENTITY)


def test_changed_roundtrip_structure_is_failure(monkeypatch):
    monkeypatch.setattr(chemistry, "_load_converter", lambda: (
        lambda *a, **kw: "[CH3:1][OH:2]",
        lambda *a, **kw: "[CH3:1][F:2]>>[CH3:1][F:2]", "0.2.0"
    ))
    with pytest.raises(ReactionPreparationError, match="Mapped roundtrip"):
        prepare_reaction(IDENTITY)


def test_map_free_roundtrip_is_also_checked(monkeypatch):
    monkeypatch.setattr(chemistry, "_load_converter", lambda: (
        lambda *a, **kw: "[CH3:1][OH:2]",
        lambda s, **kw: IDENTITY if ":" in s else "C.O>>CO", "0.2.0"
    ))
    with pytest.raises(ReactionPreparationError, match="Map-free roundtrip"):
        prepare_reaction(IDENTITY)


@requires_sr
@pytest.mark.parametrize("reaction", [
    OXIDATION, WILDCARD, IDENTITY,
    "[CH2:1]=[CH2:2].[H:3][H:4]>>[CH2:1]([H:3])[CH2:2][H:4]",
    "[F-:6].[Br:1][C@:2]([H:5])([CH3:3])[NH2:4]>>[Br-:1].[CH3:3][C@:2]([H:5])([F:6])[NH2:4]",
    "[13CH3:1][OH:2]>>[13CH3:1][OH:2]",
    "[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[OH:7]>>[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[OH:7]",
])
def test_real_converter_roundtrips_and_preserves_provenance(reaction):
    output = prepare_reaction(reaction)
    assert output["roundtrip_verified"] is True
    assert output["converter_version"] == "0.2.0"
    assert_map_free(output["reaction_smiles"])
    assert_map_free(output["sr_smiles"])
    assert output["reaction_smiles"] == normalize_reaction(reaction)
    assert output["edit_spans"] == dynamic_spans(output["sr_smiles"])
    assert set(map(int, re.findall(r":(\d+)", reaction))) == {
        atom["map_id"] for atom in output["mapping_provenance"]["atom_map_spans"]
    }
    for atom in output["mapping_provenance"]["atom_map_spans"]:
        start, end = atom["span"]
        assert output["sr_smiles"][start:end].startswith("[")


@requires_sr
def test_arbitrary_map_numbers_do_not_become_sequence_features():
    first = prepare_reaction(OXIDATION)
    second = prepare_reaction("[CH3:99][CH2:8][OH:1024]>>[CH3:99][CH:8]=[O:1024]")
    assert first["sr_smiles"] == second["sr_smiles"]
    assert first["reaction_smiles"] == second["reaction_smiles"]
    assert first["edit_spans"] == second["edit_spans"]


@requires_sr
def test_wildcard_is_retained_not_replaced_with_carbon():
    output = prepare_reaction(WILDCARD)
    assert "[*]" in output["sr_smiles"]
    assert "*" in output["reaction_smiles"]


@requires_sr
def test_identity_reaction_has_no_fabricated_center():
    assert prepare_reaction(IDENTITY)["edit_spans"] == []
