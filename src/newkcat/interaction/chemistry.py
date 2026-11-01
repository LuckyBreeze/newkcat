"""Lossless RXNFP lexical alignment and optional within-component topology.

This is molecular topology on each side of a reaction, not atom mapping or a
reaction-centre annotation. Atom-map numbers are removed from model tokens.
Every original character remains covered by a token span, including characters
outside RXNFP's vocabulary or regular expression. An unparseable component only
loses its topology; its sequence tokens are retained.

RXNFP's tokenizer expression comes from the MIT-licensed implementation:
https://github.com/rxn4chemistry/rxnfp/blob/master/rxnfp/tokenization.py
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any

import numpy as np
from rdkit import Chem


RXNFP_TOKEN_PATTERN = (
    r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|"
    r"\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
)
_TOKEN_RE = re.compile(RXNFP_TOKEN_PATTERN)
_ATOM_MAP_RE = re.compile(r":([0-9]+)(?=\]$)")
_BARE_ATOMS = frozenset(
    ("B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I", "b", "c", "n", "o", "p", "s", "*")
)

ROLE_REACTANT = 0
ROLE_AGENT = 1
ROLE_PRODUCT = 2
ROLE_SEPARATOR = 3

BOND_NONE = 0
BOND_SELF = 1
BOND_SINGLE = 2
BOND_DOUBLE = 3
BOND_TRIPLE = 4
BOND_AROMATIC = 5
BOND_OTHER = 6

_BOND_TYPES = {
    Chem.BondType.SINGLE: BOND_SINGLE,
    Chem.BondType.DOUBLE: BOND_DOUBLE,
    Chem.BondType.TRIPLE: BOND_TRIPLE,
    Chem.BondType.AROMATIC: BOND_AROMATIC,
}


@dataclass(frozen=True)
class TokenizedReaction:
    """Arrays use chemical-token indices; neither CLS nor SEP is included."""

    tokens: tuple[str, ...]
    spans: np.ndarray
    roles: np.ndarray
    atom_mask: np.ndarray
    topology_mask: np.ndarray
    edge_index: np.ndarray
    edge_types: np.ndarray
    metadata: dict[str, Any]


def _lex(text: str) -> tuple[list[str], list[tuple[int, int]], list[bool], int]:
    tokens: list[str] = []
    spans: list[tuple[int, int]] = []
    recognised: list[bool] = []
    cursor = 0
    removed_maps = 0
    for match in _TOKEN_RE.finditer(text):
        # findall alone silently drops unmatched characters. Keep each of them.
        for index in range(cursor, match.start()):
            tokens.append(text[index])
            spans.append((index, index + 1))
            recognised.append(False)
        raw = match.group(0)
        token, n_maps = _ATOM_MAP_RE.subn("", raw) if raw.startswith("[") else (raw, 0)
        # The official vocabulary has '*' rather than the equivalent '[*]'.
        if token == "[*]":
            token = "*"
        removed_maps += n_maps
        tokens.append(token)
        spans.append((match.start(), match.end()))
        recognised.append(True)
        cursor = match.end()
    for index in range(cursor, len(text)):
        tokens.append(text[index])
        spans.append((index, index + 1))
        recognised.append(False)
    return tokens, spans, recognised, removed_maps


def _reaction_separator(
    index: int, tokens: list[str], spans: list[tuple[int, int]], recognised: list[bool]
) -> bool:
    if not recognised[index] or tokens[index] not in (">", ">>"):
        return False
    # A dative SMILES bond N->[Cu] contains '>', but is not a reaction arrow.
    return not (
        tokens[index] == ">" and index > 0 and tokens[index - 1] == "-"
        and spans[index - 1][1] == spans[index][0]
    )


def tokenize_reaction(text: str) -> TokenizedReaction:
    """Tokenize a reaction without truncation or row/component deletion.

    ``A>>B`` and ``A>agents>B`` receive the same three role definitions. A dot
    separates molecular components but keeps its side's role. Malformed reaction
    separators are reported in metadata, not silently interpreted as a centre.
    RDKit parses the very same component traversal used by the lexical tokens;
    atom order is therefore input order, including explicit hydrogens and '*'.
    """
    if not isinstance(text, str) or not text:
        raise ValueError("Reaction text must be a nonempty string")
    tokens, spans, recognised, removed_maps = _lex(text)
    n_tokens = len(tokens)
    roles = np.empty(n_tokens, dtype=np.int64)
    atom_mask = np.zeros(n_tokens, dtype=np.bool_)
    topology_mask = np.zeros(n_tokens, dtype=np.bool_)
    separator_count = 0
    for i, (token, known) in enumerate(zip(tokens, recognised)):
        if _reaction_separator(i, tokens, spans, recognised):
            roles[i] = ROLE_SEPARATOR
            separator_count += len(token)
        else:
            roles[i] = min(separator_count, ROLE_PRODUCT)
            atom_mask[i] = known and (token in _BARE_ATOMS or token.startswith("["))

    edges: list[tuple[int, int]] = []
    edge_types: list[int] = []
    components: list[dict[str, Any]] = []

    def add_component(start: int, stop: int) -> None:
        if start == stop:
            return
        atom_tokens = [i for i in range(start, stop) if atom_mask[i]]
        record: dict[str, Any] = {
            "token_start": start,
            "token_stop": stop,
            "char_start": spans[start][0],
            "char_stop": spans[stop - 1][1],
            "role": int(roles[start]),
            "atom_count": len(atom_tokens),
        }
        components.append(record)
        # Parse the original spelling, so normalization cannot accidentally repair
        # an invalid component. Map numbers are ignored when extracting edges.
        component_text = text[spans[start][0]:spans[stop - 1][1]]
        params = Chem.SmilesParserParams()
        params.removeHs = False
        params.parseName = False
        params.allowCXSMILES = False
        try:
            mol = Chem.MolFromSmiles(component_text, params)
        except (RuntimeError, ValueError):
            mol = None
        if mol is None or not atom_tokens:
            record["topology_status"] = "unparseable"
            return
        if mol.GetNumAtoms() != len(atom_tokens):
            record["topology_status"] = "atom_token_count_mismatch"
            record["rdkit_atom_count"] = mol.GetNumAtoms()
            return
        record["topology_status"] = "available"
        topology_mask[atom_tokens] = True
        for index in atom_tokens:
            edges.append((index, index))
            edge_types.append(BOND_SELF)
        for bond in mol.GetBonds():
            left = atom_tokens[bond.GetBeginAtomIdx()]
            right = atom_tokens[bond.GetEndAtomIdx()]
            kind = _BOND_TYPES.get(bond.GetBondType(), BOND_OTHER)
            edges.extend(((left, right), (right, left)))
            edge_types.extend((kind, kind))

    component_start = 0
    for i, token in enumerate(tokens):
        if (recognised[i] and token == ".") or _reaction_separator(
            i, tokens, spans, recognised
        ):
            add_component(component_start, i)
            component_start = i + 1
    add_component(component_start, n_tokens)
    metadata = {
        "raw_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "token_count": n_tokens,
        "token_spans": [[int(a), int(b)] for a, b in spans],
        "atom_map_numbers_removed": removed_maps,
        "unmatched_character_count": sum(not value for value in recognised),
        "unmatched_token_indices": [i for i, value in enumerate(recognised) if not value],
        "reaction_separator_status": "valid" if separator_count == 2 else "malformed",
        "reaction_separator_count": separator_count,
        "components": components,
        "topology_definition": "within_component_2d_bonds_not_reaction_centres",
    }
    return TokenizedReaction(
        tokens=tuple(tokens),
        spans=np.asarray(spans, dtype=np.int64).reshape(-1, 2),
        roles=roles,
        atom_mask=atom_mask,
        topology_mask=topology_mask,
        edge_index=np.asarray(edges, dtype=np.int64).reshape(-1, 2).T.copy(),
        edge_types=np.asarray(edge_types, dtype=np.int64),
        metadata=metadata,
    )
