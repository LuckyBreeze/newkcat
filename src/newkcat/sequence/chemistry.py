"""Conservative, audited conversion to map-free reaction and sr-SMILES text.

Atom mapping is supplied by the caller. This module never invokes a mapper,
adds atoms to balance a reaction, drops explicit hydrogen nodes, or substitutes
a measured substrate. Passing these checks establishes representation fidelity,
not the experimental correctness of the supplied atom correspondence.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import re

from rdkit import Chem, rdBase


SR_SMILES_VERSION = "0.2.0"
_MAP = re.compile(r":([0-9]+)")
_ATOM_FORM = re.compile(r"\[[^\[\]{}|]+\]|(?:Cl|Br|[BCNOPSFIbcnops*])\^*")
_BOND_FORMS = {"-", "=", "#", ":", "~", "/", "\\", "$", "->", "<-"}


class ReactionPreparationError(ValueError):
    """An input or conversion failed the declared chemistry contract."""


def _check_text(text: str, label: str) -> None:
    if not isinstance(text, str) or not text:
        raise ReactionPreparationError(f"{label} must be a nonempty string")
    if any(not 33 <= ord(char) <= 126 for char in text):
        raise ReactionPreparationError(f"{label} must contain printable ASCII without whitespace")


def _parse(smiles: str) -> Chem.Mol:
    _check_text(smiles, "SMILES")
    if any(not part for part in smiles.split(".")):
        raise ReactionPreparationError("SMILES contains an empty disconnected component")
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.parseName = False
    params.allowCXSMILES = False
    molecule = Chem.MolFromSmiles(smiles, params)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ReactionPreparationError("Invalid or empty molecular SMILES")
    if molecule.GetStereoGroups():
        raise ReactionPreparationError("Enhanced stereo groups are not supported by this format")
    return molecule


def _without_maps(molecule: Chem.Mol) -> Chem.Mol:
    output = Chem.Mol(molecule)
    for atom in output.GetAtoms():
        atom.SetAtomMapNum(0)
    return output


def _identity(molecule: Chem.Mol, *, retain_maps: bool = False) -> str:
    """Canonical disconnected-component multiset, retaining explicit H nodes."""
    molecule = Chem.Mol(molecule) if retain_maps else _without_maps(molecule)
    fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
    return ".".join(sorted(
        Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
        for fragment in fragments
    ))


def normalize_substrate(smiles: str) -> str:
    """Canonical map-free substrate text, allowing explicit dummy ``*`` nodes.

    No graph feature object is constructed. This normalizes traversal/order,
    not protonation, tautomerism, salts, stereochemistry, or isotope identity.
    """
    return _identity(_parse(smiles))


def _reaction_sides(reaction: str) -> tuple[Chem.Mol, Chem.Mol]:
    _check_text(reaction, "Reaction SMILES")
    if reaction.count(">>") != 1:
        raise ReactionPreparationError("Reaction SMILES must have exactly one reactants>>products separator")
    reactants, products = reaction.split(">>")
    if not reactants or not products:
        raise ReactionPreparationError("Both reactant and product sides must be nonempty")
    return _parse(reactants), _parse(products)


def normalize_reaction(reaction_smiles: str) -> str:
    """Normalize full reaction text without requiring a mapping or sr package."""
    reactants, products = _reaction_sides(reaction_smiles)
    return f"{_identity(reactants)}>>{_identity(products)}"


def assert_map_free(text: str) -> None:
    """Reject numeric map fields in any bracket atom, not aromatic ring bonds."""
    _check_text(text, "Chemical text")
    index = 0
    while index < len(text):
        if text[index] != "[":
            index += 1
            continue
        end = text.find("]", index + 1)
        if end < 0 or "[" in text[index + 1:end]:
            raise ReactionPreparationError("Malformed atom brackets in chemical text")
        if _MAP.search(text[index:end + 1]):
            raise ReactionPreparationError("Atom-map identifiers must not enter training text")
        index = end + 1


def _mapped_atoms(molecule: Chem.Mol, side: str) -> dict[int, Chem.Atom]:
    output = {}
    for atom in molecule.GetAtoms():
        number = atom.GetAtomMapNum()
        if number <= 0:
            raise ReactionPreparationError(
                f"{side} contains an unmapped atom (including explicit H); no mapping is inferred"
            )
        if number in output:
            raise ReactionPreparationError(f"Duplicate atom map {number} on {side} side")
        output[number] = atom
    return output


def _load_converter():
    try:
        version = importlib.metadata.version("sr-smiles")
    except importlib.metadata.PackageNotFoundError as error:
        raise ImportError(
            "sr-SMILES conversion requires the optional sr-smiles==0.2.0 dependency; "
            "install newkcat[sequence] in a compatible preprocessing environment"
        ) from error
    if version != SR_SMILES_VERSION:
        raise RuntimeError(
            f"Audited conversion requires sr-smiles=={SR_SMILES_VERSION}, found {version}"
        )
    from sr_smiles import rxn_to_sr, sr_to_rxn

    return rxn_to_sr, sr_to_rxn, version


def _strip_maps_with_provenance(mapped_text: str) -> tuple[str, list[dict]]:
    """Remove only atom-map fields inside atom brackets; preserve aromatic ':' bonds."""
    chunks = []
    atom_spans = []
    offset = 0
    index = 0
    while index < len(mapped_text):
        if mapped_text[index] != "[":
            chunks.append(mapped_text[index])
            offset += 1
            index += 1
            continue
        end = mapped_text.find("]", index + 1)
        if end < 0 or "[" in mapped_text[index + 1:end]:
            raise ReactionPreparationError("Malformed bracket atom in converter output")
        atom = mapped_text[index:end + 1]
        maps = list(_MAP.finditer(atom))
        if len(maps) != 1:
            raise ReactionPreparationError("Converter output lost a unique atom-map identifier")
        clean = _MAP.sub("", atom)
        atom_spans.append({
            "map_id": int(maps[0].group(1)),
            "span": [offset, offset + len(clean)],
        })
        chunks.append(clean)
        offset += len(clean)
        index = end + 1
    return "".join(chunks), atom_spans


def dynamic_spans(sr_smiles: str) -> list[list[int]]:
    """Parse sr ``{before|after}`` atom/bond fields into half-open ranges.

    This identifies the serialized changes themselves, including a dynamic
    bond written at a ring closure. It does not claim to identify surrounding
    atoms, graph-hop neighborhoods, or physical transition-state geometry.
    Full graph validity is established separately by the conversion roundtrip.
    """
    _check_text(sr_smiles, "sr-SMILES")
    spans = []
    index = 0
    while index < len(sr_smiles):
        char = sr_smiles[index]
        if char == "[":
            end = sr_smiles.find("]", index + 1)
            if end < 0 or any(c in sr_smiles[index + 1:end] for c in "[]{}|"):
                raise ReactionPreparationError("Malformed atom field in sr-SMILES")
            if _MAP.search(sr_smiles[index:end + 1]):
                raise ReactionPreparationError("Atom-map identifiers must not enter training text")
            index = end + 1
        elif char == "{":
            end = sr_smiles.find("}", index + 1)
            if end < 0:
                raise ReactionPreparationError("Unclosed dynamic field in sr-SMILES")
            body = sr_smiles[index + 1:end]
            if "{" in body or body.count("|") != 1:
                raise ReactionPreparationError("Dynamic fields must be flat {before|after} pairs")
            before, after = body.split("|")
            bond = before in _BOND_FORMS and after in _BOND_FORMS
            atom = bool(_ATOM_FORM.fullmatch(before) and _ATOM_FORM.fullmatch(after))
            if not (bond or atom) or before == after:
                raise ReactionPreparationError("Invalid or unchanged dynamic atom/bond field")
            if atom and (_MAP.search(before) or _MAP.search(after)):
                raise ReactionPreparationError("Atom-map identifiers must not enter training text")
            spans.append([index, end + 1])
            index = end + 1
        elif char in "}|]":
            raise ReactionPreparationError("Unmatched delimiter in sr-SMILES")
        else:
            index += 1
    return spans


def _canonical_mapped(molecule: Chem.Mol) -> str:
    # Map numbers are retained for conversion but do not determine traversal.
    # Rank a map-free copy and traverse the renumbered original. In some RDKit
    # versions ignoreAtomMapNumbers=True restores IDs with a strict <1000
    # setter, although larger IDs in valid input are otherwise supported.
    fragments = Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)
    fragments = sorted(fragments, key=_identity)
    output = []
    for fragment in fragments:
        ranks = list(Chem.CanonicalRankAtoms(
            _without_maps(fragment), breakTies=True, includeChirality=True, includeIsotopes=True
        ))
        order = sorted(range(len(ranks)), key=ranks.__getitem__)
        ordered = Chem.RenumberAtoms(fragment, order)
        output.append(Chem.MolToSmiles(ordered, canonical=False, isomericSmiles=True))
    return ".".join(output)


def prepare_reaction(mapped_rxn: str) -> dict:
    """Return paired R/C representations with an explicit conversion audit.

    All explicit nodes must have one unique, matching map ID on each side;
    corresponding nodes must retain element and isotope identity. Implicit-H
    and charge changes are permitted. Thus this is a mapped-node consistency
    check, not a proof of atom/electron stoichiometric completeness.
    """
    reactants, products = _reaction_sides(mapped_rxn)
    reac_atoms = _mapped_atoms(reactants, "Reactant")
    prod_atoms = _mapped_atoms(products, "Product")
    if reac_atoms.keys() != prod_atoms.keys():
        raise ReactionPreparationError("Reactant/product map sets differ; automatic balancing is disabled")
    for number, atom in reac_atoms.items():
        other = prod_atoms[number]
        if (atom.GetAtomicNum(), atom.GetIsotope()) != (other.GetAtomicNum(), other.GetIsotope()):
            raise ReactionPreparationError(f"Map {number} changes element or isotope identity")
    rxn_to_sr, sr_to_rxn, version = _load_converter()
    canonical_input = f"{_canonical_mapped(reactants)}>>{_canonical_mapped(products)}"
    mapped_sr = rxn_to_sr(
        canonical_input,
        keep_atom_mapping=True,
        remove_hydrogens=False,
        balance_rxn=False,
        kekulize=False,
        keep_aromatic_bonds=True,
    )
    # Upstream catches its own exceptions and returns an empty string.
    if not isinstance(mapped_sr, str) or not mapped_sr:
        raise ReactionPreparationError("sr-SMILES conversion returned empty output")
    sr_smiles, atom_spans = _strip_maps_with_provenance(mapped_sr)
    if {item["map_id"] for item in atom_spans} != set(reac_atoms):
        raise ReactionPreparationError("Converter output changed the explicit mapped-node inventory")
    edit_spans = dynamic_spans(sr_smiles)
    mapped_back = sr_to_rxn(mapped_sr, add_atom_mapping=True)
    if not isinstance(mapped_back, str) or not mapped_back:
        raise ReactionPreparationError("Mapped sr-SMILES reverse conversion returned empty output")
    back_sides = _reaction_sides(mapped_back)
    for name, original, restored in zip(("reactant", "product"), (reactants, products), back_sides):
        if _identity(original, retain_maps=True) != _identity(restored, retain_maps=True):
            raise ReactionPreparationError(
                f"Mapped roundtrip changed {name} structure, hydrogen nodes, isotope, charge, or stereo"
            )
    mapless_back = sr_to_rxn(sr_smiles, add_atom_mapping=False)
    reaction_smiles = f"{_identity(reactants)}>>{_identity(products)}"
    if not isinstance(mapless_back, str) or not mapless_back:
        raise ReactionPreparationError("Map-free sr-SMILES reverse conversion returned empty output")
    if normalize_reaction(mapless_back) != reaction_smiles:
        raise ReactionPreparationError("Map-free roundtrip changed the reaction graph identity")
    return {
        "reaction_smiles": reaction_smiles,
        "sr_smiles": sr_smiles,
        "edit_spans": edit_spans,
        "converter_version": version,
        "rdkit_version": rdBase.rdkitVersion,
        "mapping_provenance": {
            "input_sha256": hashlib.sha256(mapped_rxn.encode("utf-8")).hexdigest(),
            "mapped_sr_smiles": mapped_sr,
            "atom_map_spans": atom_spans,
            "reactant_atom_indices": {str(k): v.GetIdx() for k, v in reac_atoms.items()},
            "product_atom_indices": {str(k): v.GetIdx() for k, v in prod_atoms.items()},
        },
        "roundtrip_verified": True,
        "validation": "mapped_node_identity_and_graph_roundtrip",
        "edit_definition": "serialized_dynamic_atom_and_bond_fields",
    }
