"""Validated atom-map keyed centers and conservative reference-label extraction.

Reference labels are derived from *net* structural changes, not catalytic
mechanisms. They are privileged information when products are unavailable at
inference; downstream training/evaluation must explicitly acknowledge that regime.
"""

import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem

from .chem import molecular_graph_identity, molecular_identity, parse_smiles, without_maps


@dataclass(frozen=True)
class ValidatedCenter:
    mapped_smiles: str
    scores: np.ndarray
    kind: str
    source: str
    uses_ec: bool


def _unique_object(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        output[key] = value
    return output


def load_center_records(path: str | Path | None) -> dict[str, dict]:
    """Read records; chemical validation is deferred to selected samples only."""
    if path is None:
        return {}
    records = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line, object_pairs_hook=_unique_object)
            except (ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid centers JSONL line {line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Center line {line_number} must be a JSON object")
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"Center line {line_number} needs a nonempty sample_id")
            if sample_id != sample_id.strip():
                raise ValueError(f"Center line {line_number}: sample_id has surrounding whitespace")
            if sample_id in records:
                raise ValueError(f"Duplicate center record for sample_id={sample_id!r}")
            records[sample_id] = record
    return records


def _mapped_atoms(mol: Chem.Mol, *, context: str) -> dict[int, Chem.Atom]:
    mapping = {}
    for atom in mol.GetAtoms():
        number = atom.GetAtomMapNum()
        if number <= 0:
            raise ValueError(f"{context}: every atom must have a positive map number")
        if number in mapping:
            raise ValueError(f"{context}: duplicate atom-map number {number}")
        mapping[number] = atom
    return mapping


def validate_center_record(record: dict, expected_smiles: str) -> ValidatedCenter:
    sample_id = record.get("sample_id", "<unknown>")
    for name in ("mapped_smiles", "kind", "source"):
        if not isinstance(record.get(name), str) or not record[name].strip():
            raise ValueError(f"Center {sample_id}: missing/non-string {name}")
    if record["kind"] not in {"predicted", "reference"}:
        raise ValueError(f"Center {sample_id}: kind must be predicted or reference")
    if type(record.get("uses_ec")) is not bool:
        raise ValueError(f"Center {sample_id}: uses_ec must be an explicit JSON boolean")
    mol = parse_smiles(record["mapped_smiles"])
    mapping = _mapped_atoms(mol, context=f"Center {sample_id}")
    if molecular_identity(mol) != molecular_identity(expected_smiles):
        raise ValueError(
            f"Center {sample_id}: mapped molecule differs in structure, charge, isotope, "
            "or stereochemistry from the recorded substrate"
        )
    if molecular_graph_identity(mol) != molecular_graph_identity(expected_smiles):
        raise ValueError(
            f"Center {sample_id}: explicit-hydrogen graph representation must match the "
            "recorded substrate; center annotations may only permute graph atoms"
        )
    scores = record.get("scores")
    if not isinstance(scores, dict):
        raise ValueError(f"Center {sample_id}: scores must map atom-map-number strings to scores")
    if set(scores) != {str(number) for number in mapping}:
        raise ValueError(f"Center {sample_id}: score keys must exactly match every atom map")
    values = []
    for atom in mol.GetAtoms():
        value = scores[str(atom.GetAtomMapNum())]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Center {sample_id}: scores must be numeric, not booleans or strings")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Center {sample_id}: scores must be finite and in [0, 1]")
        values.append(value)
    if record["kind"] == "reference" and any(value not in (0, 1) for value in values):
        raise ValueError(f"Center {sample_id}: reference labels must be binary")
    return ValidatedCenter(
        record["mapped_smiles"], np.asarray(values, dtype=np.float32),
        record["kind"], record["source"], record["uses_ec"],
    )


def _mapped_tetrahedral_state(atom: Chem.Atom) -> tuple:
    """Handedness relative to mapped neighbors, independent of CIP priorities.

    RDKit's tetrahedral tag refers to neighbor iteration order. Correcting its
    sign by that order's parity gives the orientation in ascending map order.
    An implicit H/virtual fourth neighbor occupies the same last slot on both
    sides. Changes in neighbors/H-count are separately detected as center edits.
    """
    tag = atom.GetChiralTag()
    if tag == Chem.ChiralType.CHI_UNSPECIFIED:
        return ("unspecified",)
    if tag not in {Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW}:
        raise ValueError("Reference centers only support tetrahedral atom stereochemistry")
    neighbors = [neighbor.GetAtomMapNum() for neighbor in atom.GetNeighbors()]
    parity = sum(a > b for i, a in enumerate(neighbors) for b in neighbors[i + 1:]) % 2
    clockwise = int(tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    return ("tetrahedral", clockwise ^ parity)


def _mapped_bond_stereo(bond: Chem.Bond) -> tuple:
    """Cis/trans state relative to the minimum-map neighbor at each endpoint.

    E/Z can change when distant substituents change CIP priority, while the
    local geometry remains fixed. Moving the reference to the other substituent
    on one endpoint flips the cis/trans bit; moving both references cancels.
    """
    stereo = bond.GetStereo()
    if stereo in {Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY}:
        return ("unspecified",)
    cis = {Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS}
    trans = {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS}
    if stereo not in cis | trans or len(bond.GetStereoAtoms()) != 2:
        raise ValueError("Unsupported or incomplete reference bond stereochemistry")
    begin, end = bond.GetBeginAtom(), bond.GetEndAtom()
    left = [a.GetAtomMapNum() for a in begin.GetNeighbors() if a.GetIdx() != end.GetIdx()]
    right = [a.GetAtomMapNum() for a in end.GetNeighbors() if a.GetIdx() != begin.GetIdx()]
    if not left or not right or len(left) > 2 or len(right) > 2:
        raise ValueError("Unsupported stereobond neighborhood for reference centers")
    references = [bond.GetOwningMol().GetAtomWithIdx(i).GetAtomMapNum()
                  for i in bond.GetStereoAtoms()]
    if references[0] not in left or references[1] not in right:
        raise ValueError("Stereobond reference atoms do not match bond endpoints")
    parity = int(stereo in trans) ^ int(references[0] != min(left)) ^ int(references[1] != min(right))
    return ("cis-trans", parity)


def _atom_state(atom: Chem.Atom) -> tuple:
    return (
        atom.GetFormalCharge(), atom.GetTotalNumHs(includeNeighbors=True),
        atom.GetNumRadicalElectrons(), _mapped_tetrahedral_state(atom),
    )


def _neighbor_state(atom: Chem.Atom) -> dict[int, tuple]:
    return {
        bond.GetOtherAtom(atom).GetAtomMapNum():
        (str(bond.GetBondType()), bool(bond.GetIsAromatic()), _mapped_bond_stereo(bond))
        for bond in atom.GetBonds()
    }


def _stereo_with_maps(mol: Chem.Mol) -> Chem.Mol:
    """Compute CIP without map labels affecting ranking, then restore identifiers."""
    numbers = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    clean = without_maps(mol)
    for atom, number in zip(clean.GetAtoms(), numbers):
        atom.SetAtomMapNum(number)
    return clean


def reference_center_from_reaction(
    sample_id: str,
    substrate_smiles: str,
    reaction_smiles: str,
    *,
    source: str = "user-supplied-atom-mapped-reaction",
    substrate_maps: list[int] | None = None,
) -> dict:
    """Extract a conservative main-substrate reference center from a mapped reaction.

    Requires complete, unique mapping on both sides and equal atom-map sets.
    Main-substrate selection is exact (including stereochemistry), not Tanimoto
    matching. Duplicate identical reactants require explicit ``substrate_maps``.
    Reactions with missing coproduct atoms fail instead of inventing unchanged labels.
    The optional middle agents field is not treated as an atom-contributing reactant.
    """
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError("A nonempty sample_id is required")
    if sample_id != sample_id.strip():
        raise ValueError("sample_id must not contain surrounding whitespace")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("A nonempty source is required")
    if not isinstance(reaction_smiles, str):
        raise ValueError("Reaction SMILES must be a string")
    parts = reaction_smiles.split(">")
    if len(parts) != 3 or not parts[0].strip() or not parts[2].strip():
        raise ValueError("Reaction SMILES must be reactants>>products or reactants>agents>products")
    reactants = _stereo_with_maps(parse_smiles(parts[0]))
    products = _stereo_with_maps(parse_smiles(parts[2]))
    r_atoms = _mapped_atoms(reactants, context="Reactants")
    p_atoms = _mapped_atoms(products, context="Products")
    if set(r_atoms) != set(p_atoms):
        raise ValueError("Incomplete atom correspondence: both sides must have the same atom maps")
    for number in r_atoms:
        if (r_atoms[number].GetAtomicNum(), r_atoms[number].GetIsotope()) != (
            p_atoms[number].GetAtomicNum(), p_atoms[number].GetIsotope()
        ):
            raise ValueError(f"Atom map {number} changes element or isotope")
    expected_identity = molecular_identity(substrate_smiles)
    if substrate_maps is None:
        matches = [
            fragment for fragment in Chem.GetMolFrags(reactants, asMols=True, sanitizeFrags=True)
            if molecular_identity(fragment) == expected_identity
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one exact main-substrate reactant, found {len(matches)}; "
                "supply explicit substrate_maps to resolve duplicate/disconnected components"
            )
        selected = matches[0]
    else:
        if (
            not substrate_maps or any(type(value) is not int or value <= 0 for value in substrate_maps)
            or len(set(substrate_maps)) != len(substrate_maps)
            or not set(substrate_maps).issubset(r_atoms)
        ):
            raise ValueError("substrate_maps must be distinct positive reactant map numbers")
        selected_edit = Chem.RWMol(reactants)
        retained = set(substrate_maps)
        if any(
            (bond.GetBeginAtom().GetAtomMapNum() in retained)
            != (bond.GetEndAtom().GetAtomMapNum() in retained)
            for bond in reactants.GetBonds()
        ):
            raise ValueError("substrate_maps must select complete reactant components")
        for atom in reversed(list(reactants.GetAtoms())):
            if atom.GetAtomMapNum() not in retained:
                selected_edit.RemoveAtom(atom.GetIdx())
        selected = selected_edit.GetMol()
        Chem.SanitizeMol(selected)
        if molecular_identity(selected) != expected_identity:
            raise ValueError("Explicit substrate_maps do not describe the recorded substrate")
    selected_maps = [atom.GetAtomMapNum() for atom in selected.GetAtoms()]
    scores = {
        str(number): int(
            _atom_state(r_atoms[number]) != _atom_state(p_atoms[number])
            or _neighbor_state(r_atoms[number]) != _neighbor_state(p_atoms[number])
        )
        for number in selected_maps
    }
    if not any(scores.values()):
        raise ValueError("No net structural change touches the requested main substrate")
    record = {
        "sample_id": sample_id,
        "mapped_smiles": Chem.MolToSmiles(selected, canonical=False, isomericSmiles=True),
        "scores": scores,
        "kind": "reference",
        "source": source,
        "uses_ec": False,
        "reaction_smiles": reaction_smiles,
        "label_definition": (
            "bond-neighborhood/order, map-relative tetrahedral or cis/trans stereo, "
            "charge, H-count, or radical change"
        ),
    }
    validate_center_record(record, substrate_smiles)
    return record


def reference_centers_from_csv(
    samples_path: str | Path,
    output_path: str | Path,
    *,
    reaction_column: str = "reaction_smiles",
    overwrite: bool = False,
) -> int:
    """Strict, transactional CSV-to-JSONL conversion; no automatic atom mapping."""
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}")
    records = []
    seen = set()
    with Path(samples_path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "smiles", reaction_column}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Reaction CSV must include columns {sorted(required)}")
        for row_number, row in enumerate(reader, 2):
            if row["sample_id"] in seen:
                raise ValueError(f"Duplicate sample_id on CSV line {row_number}")
            seen.add(row["sample_id"])
            explicit_maps = None
            if row.get("substrate_maps", "").strip():
                explicit_maps = json.loads(row["substrate_maps"])
                if not isinstance(explicit_maps, list):
                    raise ValueError("substrate_maps must be a JSON list of positive integers")
            try:
                records.append(reference_center_from_reaction(
                    row["sample_id"], row["smiles"], row[reaction_column],
                    source=row.get("source") or "user-supplied-atom-mapped-reaction",
                    substrate_maps=explicit_maps,
                ))
            except ValueError as error:
                raise ValueError(f"Reaction CSV line {row_number}: {error}") from error
    if not records:
        raise ValueError("Reaction CSV has no records")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as f:
            temporary = Path(f.name)
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        if output.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {output}")
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return len(records)
