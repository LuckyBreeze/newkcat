"""Tests of symbol fidelity, positional masks, and the fixed vocabulary contract."""

import numpy as np
import pytest

from newkcat.sequence.tokenizer import (
    ALPHABET, BOS_ID, EOS_ID, MASK_ID, PAD_ID, VOCAB_SIZE, ReactionTokenizer,
)


@pytest.mark.parametrize("text", [
    "C*", "[2H][C@@H](F)C(=O)[O-]", "[Na+].[Cl-]>>[Na+].[Cl-]",
    "[CH3]{[CH2]|[CH]}{-|=}{[OH]|[O]}", "C1{~|-}CC{-|~}1", "C[->=]O",
    ALPHABET,
])
def test_fixed_alphabet_roundtrip(text):
    tokenizer = ReactionTokenizer()
    encoded = tokenizer.encode(text)
    assert tokenizer.decode(encoded["input_ids"]) == text
    assert len(encoded["input_ids"]) == len(text) + 2
    assert encoded["input_ids"][0] == BOS_ID
    assert encoded["input_ids"][-1] == EOS_ID
    assert all(4 <= value < VOCAB_SIZE for value in encoded["input_ids"][1:-1])
    assert not any(encoded["edit_mask"])


def test_dynamic_mask_offsets_include_bos_correctly():
    text = "[C]{-|=}[O]"
    encoded = ReactionTokenizer().encode(text, [[3, 8]])
    assert encoded["edit_mask"] == [False] * 4 + [True] * 5 + [False] * 4
    assert "".join(char for char, active in zip(text, encoded["edit_mask"][1:-1]) if active) == "{-|=}"


def test_adjacent_edit_fields_are_allowed():
    text = "{C|N}{-|=}O"
    encoded = ReactionTokenizer().encode(text, [[0, 5], [5, 10]])
    assert sum(encoded["edit_mask"]) == 10


@pytest.mark.parametrize("text", ["", None, "C C", "C\nO", "ＣCO", "CCé", "CC\x7f"])
def test_invalid_text_is_not_silently_dropped_or_replaced(text):
    with pytest.raises(ValueError):
        ReactionTokenizer().encode(text)


def test_length_limit_includes_boundary_tokens_without_truncation():
    tokenizer = ReactionTokenizer()
    assert len(tokenizer.encode("CCO", max_length=5)["input_ids"]) == 5
    with pytest.raises(ValueError, match="silent truncation"):
        tokenizer.encode("CCO", max_length=4)


@pytest.mark.parametrize("limit", [True, 2, 0, -1, 3.0, None])
def test_invalid_length_limit(limit):
    with pytest.raises(ValueError):
        ReactionTokenizer().encode("C", max_length=limit)


@pytest.mark.parametrize("spans", [
    None, "0:1", [[0]], [[0, 4]], [[-1, 1]], [[1, 1]], [[0.0, 1]],
    [[False, 1]], [[1, 2], [0, 1]], [[0, 2], [1, 3]], [[0, 1, 2]],
])
def test_invalid_span_coordinates_fail(spans):
    with pytest.raises(ValueError):
        ReactionTokenizer().encode("CCO", edit_spans=spans)


def test_metadata_is_fixed_and_not_mutable_through_return_value():
    tokenizer = ReactionTokenizer()
    assert (PAD_ID, BOS_ID, EOS_ID, MASK_ID, VOCAB_SIZE) == (0, 1, 2, 3, 98)
    assert ReactionTokenizer.from_state_dict(tokenizer.state_dict()).fingerprint == tokenizer.fingerprint
    metadata = tokenizer.metadata()
    metadata["special_ids"]["pad"] = 7
    assert tokenizer.metadata()["special_ids"]["pad"] == 0
    with pytest.raises(ValueError, match="metadata"):
        ReactionTokenizer.from_state_dict(metadata)


def test_decode_numpy_ids_and_special_display():
    tokenizer = ReactionTokenizer()
    ids = np.asarray(tokenizer.encode("CO")["input_ids"] + [PAD_ID])
    assert tokenizer.decode(ids) == "CO"
    assert tokenizer.decode([BOS_ID, MASK_ID, EOS_ID], skip_special_tokens=False) == "<bos><mask><eos>"


@pytest.mark.parametrize("ids", [[-1], [98], [True], [4.0], ["4"]])
def test_decode_rejects_invalid_ids(ids):
    with pytest.raises(ValueError):
        ReactionTokenizer().decode(ids)
