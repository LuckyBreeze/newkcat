"""Order-equivariant 2D molecular features with explicit atom-index bookkeeping.

Atom maps are identifiers, never model features. Absolute CIP labels distinguish
specified tetrahedral stereochemistry without using traversal-dependent CW/CCW
tags. These features do not claim to describe a bound 3D conformation.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np
from rdkit import Chem


ATOM_DIM = 160
BOND_DIM = 16
_HYBRIDIZATIONS = (
    Chem.HybridizationType.UNSPECIFIED,
    Chem.HybridizationType.S,
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
)
_BOND_TYPES = (
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
    Chem.BondType.ZERO,
    Chem.BondType.DATIVE,
)
_BOND_STEREO = (
    Chem.BondStereo.STEREONONE,
    Chem.BondStereo.STEREOANY,
    Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOCIS,
    Chem.BondStereo.STEREOTRANS,
)


@dataclass(frozen=True)
class MolecularGraph:
    atom_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    distances: np.ndarray
    atom_maps: tuple[int, ...]
    smiles: str
    identity: str

    @property
    def num_atoms(self) -> int:
        return int(self.atom_features.shape[0])


def parse_smiles(smiles: str) -> Chem.Mol:
    """Parse without silently deleting explicit (possibly mapped) hydrogens."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a nonempty string")
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.parseName = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid or empty SMILES: {smiles!r}")
    _validate_supported_molecule(mol)
    return mol


def _validate_supported_molecule(mol: Chem.Mol) -> None:
    """Enforce the same supported chemistry for strings and direct RDKit input."""
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        raise ValueError("Wildcard/dummy atoms are not supported as measured substrates")
    if mol.GetStereoGroups():
        raise ValueError("Enhanced CXSMILES stereo groups are not supported")
    supported_atom_stereo = {
        Chem.ChiralType.CHI_UNSPECIFIED,
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    }
    if any(atom.GetChiralTag() not in supported_atom_stereo for atom in mol.GetAtoms()):
        raise ValueError("Only tetrahedral atom stereochemistry is currently supported")
    if any(bond.GetStereo() not in _BOND_STEREO for bond in mol.GetBonds()):
        raise ValueError("Unsupported bond stereochemistry")


def without_maps(mol: Chem.Mol) -> Chem.Mol:
    _validate_supported_molecule(mol)
    result = Chem.Mol(mol)
    for atom in result.GetAtoms():
        atom.SetAtomMapNum(0)
    Chem.AssignStereochemistry(result, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    return result


def molecular_identity(smiles_or_mol: str | Chem.Mol) -> str:
    """Canonical identity preserving isotope, charge and specified stereochemistry.

    Ordinary explicit hydrogen spelling is normalized only for this identity
    comparison. Graph construction itself retains explicitly supplied atoms.
    No tautomer, protonation, salt, or largest-fragment normalization is performed.
    """
    mol = parse_smiles(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
    clean = without_maps(mol)
    clean = Chem.RemoveHs(clean)
    Chem.AssignStereochemistry(clean, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    return Chem.MolToSmiles(clean, canonical=True, isomericSmiles=True)


def molecular_graph_identity(smiles_or_mol: str | Chem.Mol) -> str:
    """Identity of the actual modeled graph, retaining explicit hydrogen nodes.

    Chemical equivalence is insufficient for ablations: introducing explicit H
    nodes through a center file would otherwise change even center-agnostic GNNs.
    """
    mol = parse_smiles(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
    return Chem.MolToSmiles(without_maps(mol), canonical=True, isomericSmiles=True)


def _one_hot(value, vocabulary) -> list[float]:
    return [float(value == candidate) for candidate in vocabulary] + [
        float(value not in vocabulary)
    ]


def absolute_stereo(atom: Chem.Atom) -> str:
    if atom.HasProp("_CIPCode"):
        return atom.GetProp("_CIPCode")
    if atom.HasProp("_ChiralityPossible"):
        return "unassigned"
    return "none"


def _atom_features(atom: Chem.Atom) -> list[float]:
    return (
        _one_hot(atom.GetAtomicNum(), tuple(range(1, 119)))
        + _one_hot(atom.GetFormalCharge(), tuple(range(-3, 4)))
        + _one_hot(atom.GetDegree(), tuple(range(7)))
        + _one_hot(atom.GetTotalNumHs(includeNeighbors=True), tuple(range(5)))
        + _one_hot(atom.GetHybridization(), _HYBRIDIZATIONS)
        + _one_hot(absolute_stereo(atom), ("none", "unassigned", "R", "S", "r", "s"))
        + [
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            float(atom.GetMass() / 200.0),
            float(atom.GetNumRadicalElectrons() / 4.0),
        ]
    )


def _bond_features(bond: Chem.Bond) -> list[float]:
    return (
        _one_hot(bond.GetBondType(), _BOND_TYPES)
        + _one_hot(bond.GetStereo(), _BOND_STEREO)
        + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
    )


def featurize_mol(mol: Chem.Mol, *, smiles: str = "", max_atoms: int = 512) -> MolecularGraph:
    """Featurize in the supplied RDKit atom order; no canonical renumbering."""
    n = mol.GetNumAtoms()
    if not 1 <= n <= max_atoms:
        raise ValueError(f"Molecule has {n} atoms; supported range is 1..{max_atoms}")
    atom_maps = tuple(atom.GetAtomMapNum() for atom in mol.GetAtoms())
    clean = without_maps(mol)
    atoms = np.asarray([_atom_features(atom) for atom in clean.GetAtoms()], dtype=np.float32)
    edges: list[tuple[int, int]] = []
    attributes: list[list[float]] = []
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for bond in clean.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feature = _bond_features(bond)
        edges.extend(((i, j), (j, i)))
        attributes.extend((feature, feature))
        adjacency[i].append(j)
        adjacency[j].append(i)
    edge_index = np.asarray(edges, dtype=np.int64).reshape(-1, 2).T.copy()
    edge_features = np.asarray(attributes, dtype=np.float32).reshape(-1, BOND_DIM)
    distances = np.full((n, n), -1, dtype=np.int64)
    for root in range(n):
        distances[root, root] = 0
        queue = deque([root])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if distances[root, neighbor] < 0:
                    distances[root, neighbor] = distances[root, node] + 1
                    queue.append(neighbor)
    if atoms.shape != (n, ATOM_DIM):
        raise RuntimeError("Internal atom-feature schema does not match ATOM_DIM")
    return MolecularGraph(
        atoms, edge_index, edge_features, distances, atom_maps,
        smiles or Chem.MolToSmiles(mol, canonical=False, isomericSmiles=True),
        molecular_identity(mol),
    )


def featurize_smiles(smiles: str, *, max_atoms: int = 512) -> MolecularGraph:
    return featurize_mol(parse_smiles(smiles), smiles=smiles, max_atoms=max_atoms)
