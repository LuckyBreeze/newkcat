"""Version-specific, local fixes for sr-smiles 0.2.0 reverse conversion.

The chirality helper below is adapted from sr-smiles' update_chirality_tags:
https://github.com/heid-lab/sr-smiles
Copyright (c) 2025 E. Heid, MIT License; see THIRD_PARTY_NOTICES.md.

Two upstream decoder behaviours can change otherwise valid inputs: its chiral
replacement regex misses atoms containing H counts/isotopes, and map-free bracket
cleanup turns e.g. [S] into S, introducing implicit hydrogen. Fixing these decoder
operations does not relax the caller's mapped or map-free roundtrip checks.
The original installed module and its process-global functions remain untouched.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re
from types import FunctionType


SR_DECODER_COMPAT = "0.2.0-chiral-h-and-brackets-v1"


def _update_chirality_tags(smiles: str, sr_scaffold: str,
                           chiral_center_map_nums: list[int]) -> str:
    """Keep upstream neighbour-parity logic, replacing the complete atom token."""
    from sr_smiles.chem_utils.list_utils import (
        common_elements_preserving_order, is_num_permutations_even,
    )
    from sr_smiles.chem_utils.smiles_utils import (
        TokenType, _tokenize, extract_chiral_tag_by_atom_map_num,
        get_atom_map_adjacency_list_from_smiles,
    )

    current_adj = get_atom_map_adjacency_list_from_smiles(smiles)
    scaffold_adj = get_atom_map_adjacency_list_from_smiles(sr_scaffold)
    tokens = [[kind, token] for kind, _, token in _tokenize(smiles)]
    for index, (kind, token) in enumerate(tokens):
        if kind != TokenType.ATOM:
            continue
        map_match = re.search(r":(\d+)", token)
        if map_match is None:
            raise ValueError("Chirality repair requires an atom-mapped decoder scaffold")
        map_num = int(map_match.group(1))
        if map_num not in chiral_center_map_nums:
            continue
        current, reference = common_elements_preserving_order(
            current_adj[map_num], scaffold_adj[map_num]
        )
        tag = extract_chiral_tag_by_atom_map_num(sr_scaffold, map_num)
        if tag not in {"@", "@@"}:
            raise ValueError("Missing tetrahedral reference tag during sr decoding")
        if not is_num_permutations_even(current, reference):
            tag = "@@" if tag == "@" else "@"
        # H counts and isotope prefixes must survive an @/@@ replacement. The
        # original 0.2.0 regex matches [C@:1], but silently misses [C@H:1].
        pattern = (
            rf"(\[(?:\d+)?(?:[A-Z][a-z]?|se|as|[bcnops]|\*))"
            rf"(@{{1,2}})?((?:H\d*)?(?:[+-]\d*|[+-]+)?:{map_num}\])"
        )
        atom_match = re.fullmatch(pattern, str(token))
        if atom_match is None:
            raise ValueError(f"Unsupported chiral atom token during sr decoding: {token}")
        tokens[index][1] = atom_match.group(1) + tag + atom_match.group(3)
    return "".join(str(token) for _, token in tokens)


def _preserve_brackets(text: str) -> str:
    # A bracket is not necessarily redundant: [S] and S have different H/radical
    # states. Keeping all brackets is lossless and valid molecular SMILES.
    return text


def _clone(function, namespace):
    clone = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    return clone


def load_sr_decoder():
    """Return an isolated compatible decoder for the exact audited version.

    Private upstream helper bindings are copied into a private namespace. No
    file, source text, installed function or module-global binding is modified.
    """
    if importlib.metadata.version("sr-smiles") != "0.2.0":
        raise RuntimeError("The sr decoder compatibility adapter requires sr-smiles==0.2.0")
    upstream = importlib.import_module("sr_smiles.transforms.sr_to_rxn")
    namespace = dict(vars(upstream))
    namespace["update_chirality_tags"] = _update_chirality_tags
    namespace["remove_redundant_brackets"] = _preserve_brackets
    namespace["_rebuild_side_from_sr"] = _clone(upstream._rebuild_side_from_sr, namespace)
    return _clone(upstream.sr_to_rxn, namespace)
