"""A fixed, lossless character vocabulary for controlled reaction comparisons.

This is deliberately not the tokenizer or pretrained model from the sr-SMILES
paper. All representations share the same printable-ASCII vocabulary. No test
data are used to fit it, no chemical symbols are converted to UNK, and long
strings are rejected instead of being silently truncated.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from numbers import Integral


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
MASK_ID = 3
ALPHABET = "".join(chr(code) for code in range(33, 127))
VOCAB_SIZE = 4 + len(ALPHABET)
_SCHEMA = {
    "name": "printable_ascii_char",
    "version": 1,
    "alphabet": ALPHABET,
    "special_ids": {"pad": PAD_ID, "bos": BOS_ID, "eos": EOS_ID, "mask": MASK_ID},
}
TOKENIZER_FINGERPRINT = hashlib.sha256(
    json.dumps(_SCHEMA, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


class ReactionTokenizer:
    """Map each chemical-text character to exactly one stable token.

    Edit spans are half-open Python character ranges in the *map-free* input
    string. BOS/EOS are never edits. The optional chemistry preprocessing step,
    not this lexical tokenizer, establishes the chemical validity of a string.
    """

    PAD_ID = PAD_ID
    BOS_ID = BOS_ID
    EOS_ID = EOS_ID
    MASK_ID = MASK_ID
    VOCAB_SIZE = VOCAB_SIZE
    pad_id = PAD_ID
    bos_id = BOS_ID
    eos_id = EOS_ID
    mask_id = MASK_ID
    vocab_size = VOCAB_SIZE
    fingerprint = TOKENIZER_FINGERPRINT

    def metadata(self) -> dict:
        """Return a JSON-serializable vocabulary contract for checkpoints."""
        return {
            **_SCHEMA,
            "special_ids": dict(_SCHEMA["special_ids"]),
            "vocab_size": VOCAB_SIZE,
            "fingerprint": self.fingerprint,
        }

    def state_dict(self) -> dict:
        return self.metadata()

    @classmethod
    def from_state_dict(cls, state: dict) -> ReactionTokenizer:
        """Reject checkpoints made with a different token-to-ID contract."""
        if not isinstance(state, dict) or state != cls().metadata():
            raise ValueError("Tokenizer metadata does not match the fixed vocabulary")
        return cls()

    def encode(
        self,
        text: str,
        edit_spans: Sequence[Sequence[int]] = (),
        max_length: int = 2048,
    ) -> dict[str, list]:
        if not isinstance(text, str) or not text:
            raise ValueError("Chemical text must be a nonempty string")
        if type(max_length) is not int or max_length < 3:
            raise ValueError("max_length must be an integer >= 3, including BOS/EOS")
        invalid = [index for index, char in enumerate(text) if not 33 <= ord(char) <= 126]
        if invalid:
            raise ValueError(
                "Chemical text contains whitespace or non-ASCII characters at positions "
                f"{invalid[:5]}"
            )
        if len(text) + 2 > max_length:
            raise ValueError(
                f"Sequence length {len(text) + 2} exceeds max_length={max_length}; "
                "silent truncation is forbidden"
            )
        if not isinstance(edit_spans, (list, tuple)):
            raise ValueError("edit_spans must be a list/tuple of [start, end] ranges")
        mask = [False] * len(text)
        previous_end = 0
        for span in edit_spans:
            if (
                not isinstance(span, (list, tuple))
                or len(span) != 2
                or any(type(value) is not int for value in span)
            ):
                raise ValueError("Each edit span must contain two integer character offsets")
            start, end = span
            if not 0 <= start < end <= len(text):
                raise ValueError("Edit span outside the input string or empty")
            if start < previous_end:
                raise ValueError("Edit spans must be sorted and non-overlapping")
            mask[start:end] = [True] * (end - start)
            previous_end = end
        return {
            "input_ids": [BOS_ID, *(ord(char) - 33 + 4 for char in text), EOS_ID],
            "edit_mask": [False, *mask, False],
        }

    def decode(self, input_ids: Iterable[int], *, skip_special_tokens: bool = True) -> str:
        """Decode text; special tokens may be shown as bracketed names."""
        if type(skip_special_tokens) is not bool:
            raise ValueError("skip_special_tokens must be a boolean")
        specials = {PAD_ID: "<pad>", BOS_ID: "<bos>", EOS_ID: "<eos>", MASK_ID: "<mask>"}
        pieces = []
        for value in input_ids:
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError("Token IDs must be integers")
            value = int(value)
            if not 0 <= value < VOCAB_SIZE:
                raise ValueError("Token ID outside the fixed vocabulary")
            if value < 4:
                if not skip_special_tokens:
                    pieces.append(specials[value])
            else:
                pieces.append(chr(value - 4 + 33))
        return "".join(pieces)
