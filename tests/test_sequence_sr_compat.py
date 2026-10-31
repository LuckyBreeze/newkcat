"""Reverse-converter defects must be fixed without weakening chemical identity."""

import importlib

import pytest
from rdkit import Chem

pytest.importorskip("sr_smiles", reason="Optional sr-smiles converter not installed")

from newkcat.sequence.chemistry import (  # noqa: E402
    _canonical_mapped, _identity, _reaction_sides, _strip_maps_with_provenance,
    normalize_reaction, prepare_reaction,
)
from newkcat.sequence.sr_compat import (  # noqa: E402
    SR_DECODER_COMPAT, _update_chirality_tags, load_sr_decoder,
)
from sr_smiles import rxn_to_sr, sr_to_rxn  # noqa: E402


RING_OPENING = (
    "[NH2:1][C@H:2]1[CH2:3][CH2:4][O:5][C:6]1=[O:7].[OH2:8]"
    ">>[NH2:1][C@@H:2]([CH2:3][CH2:4][OH:5])[C:6](=[O:7])[OH:8]"
)
SULFUR = (
    "[CH2:1]([C@H:2]([NH2:3])[C:4](=[O:5])[OH:6])[SH:7]"
    ">>[CH3:1][C@H:2]([NH2:3])[C:4](=[O:5])[OH:6].[S:7]"
)


def _sr(reaction):
    sides = _reaction_sides(reaction)
    canonical = ">>".join(_canonical_mapped(mol) for mol in sides)
    return rxn_to_sr(canonical, keep_atom_mapping=True, remove_hydrogens=False,
                     balance_rxn=False, kekulize=False, keep_aromatic_bonds=True)


def _cip(reaction, side, map_num):
    mol = _reaction_sides(reaction)[side]
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return next(atom.GetProp("_CIPCode") for atom in mol.GetAtoms()
                if atom.GetAtomMapNum() == map_num)


def test_chiral_h_ring_opening_retains_absolute_stereochemistry():
    mapped_sr = _sr(RING_OPENING)
    upstream = sr_to_rxn(mapped_sr, add_atom_mapping=True)
    fixed = load_sr_decoder()(mapped_sr, add_atom_mapping=True)
    assert _cip(RING_OPENING, 1, 2) == "S"
    assert _cip(upstream, 1, 2) == "R"  # The 0.2.0 regression being repaired.
    assert _cip(fixed, 1, 2) == "S"
    assert normalize_reaction(fixed) == normalize_reaction(RING_OPENING)
    prepared = prepare_reaction(RING_OPENING)
    assert prepared["roundtrip_verified"] is True
    assert prepared["sr_decoder_compat"] == SR_DECODER_COMPAT


@pytest.mark.parametrize("element", ["C", "13C", "Si", "29Si"])
def test_chiral_replacement_retains_hydrogen_and_isotope_fields(element):
    current = f"[F:1][{element}@H:2]([Cl:3])[Br:4]"
    reference = f"[F:1][{element}@@H:2]([Cl:3])[Br:4]"
    assert _update_chirality_tags(current, reference, [2]) == reference


def test_sulfur_brackets_do_not_turn_zero_hydrogen_s_into_implicit_h_s():
    mapped_sr = _sr(SULFUR)
    mapfree_sr, _ = _strip_maps_with_provenance(mapped_sr)
    upstream = sr_to_rxn(mapfree_sr, add_atom_mapping=False)
    fixed = load_sr_decoder()(mapfree_sr, add_atom_mapping=False)
    assert normalize_reaction(upstream) != normalize_reaction(SULFUR)
    assert normalize_reaction(fixed) == normalize_reaction(SULFUR)
    assert "[S]" in fixed
    assert prepare_reaction(SULFUR)["roundtrip_verified"] is True


def test_adapter_does_not_modify_upstream_functions_or_global_bindings():
    upstream_module = importlib.import_module("sr_smiles.transforms.sr_to_rxn")
    keys = ("sr_to_rxn", "_rebuild_side_from_sr", "update_chirality_tags",
            "remove_redundant_brackets")
    originals = {key: getattr(upstream_module, key) for key in keys}
    mapped_sr = _sr(RING_OPENING)
    old_result = sr_to_rxn(mapped_sr, add_atom_mapping=True)
    decoder1, decoder2 = load_sr_decoder(), load_sr_decoder()
    assert decoder1(mapped_sr, True) == decoder2(mapped_sr, True)
    assert decoder1(mapped_sr, True) != old_result
    assert sr_to_rxn(mapped_sr, True) == old_result
    assert all(getattr(upstream_module, key) is value for key, value in originals.items())


@pytest.mark.parametrize("reaction", [
    "[CH3:1][CH2:2][OH:3]>>[CH3:1][CH:2]=[O:3]",
    "[F:1]/[CH:2]=[CH:3]/[Cl:4]>>[F:1]/[CH:2]=[CH:3]\\[Cl:4]",
    "[*:1][CH2:2][Br:3].[OH-:4]>>[*:1][CH2:2][OH:4].[Br-:3]",
    "[13CH3:1][CH2:2][OH:3]>>[13CH3:1][CH:2]=[O:3]",
])
def test_previously_supported_reactions_remain_strictly_lossless(reaction):
    data = prepare_reaction(reaction)
    restored = load_sr_decoder()(data["mapping_provenance"]["mapped_sr_smiles"], True)
    for original, back in zip(_reaction_sides(reaction), _reaction_sides(restored)):
        assert _identity(original, retain_maps=True) == _identity(back, retain_maps=True)


def test_adapter_refuses_a_different_converter_version(monkeypatch):
    monkeypatch.setattr("newkcat.sequence.sr_compat.importlib.metadata.version", lambda _: "0.3.0")
    with pytest.raises(RuntimeError, match="sr-smiles==0.2.0"):
        load_sr_decoder()
