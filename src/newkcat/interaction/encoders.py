"""Frozen, pretrained token encoders with complete overlapping-window coverage.

Transformers is optional and imported only when an encoder is instantiated.
RXNFP model files are loaded from a user-selected local checkpoint directory;
this module never substitutes a randomly initialised backbone. It does not
depend on the old rxnfp package. Protein global features remain an independent
input: a residue encoder never fabricates/replaces those existing features.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from .chemistry import RXNFP_TOKEN_PATTERN, tokenize_reaction


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _transformers():
    try:
        import transformers
    except ImportError as error:
        raise ImportError(
            "Pretrained encoders require optional dependencies transformers and sentencepiece; "
            "run 'pip install -r requirements-interaction.txt' in the feature-generation environment."
        ) from error
    return transformers


def window_slices(length: int, window_size: int, overlap: int) -> Iterator[tuple[int, int]]:
    """Cover every item, with no padding/truncation or repeated terminal window."""
    if length < 1:
        raise ValueError("Cannot encode an empty sequence")
    if window_size < 1 or overlap < 0 or overlap >= window_size:
        raise ValueError("Require window_size > overlap >= 0")
    start = 0
    while True:
        stop = min(length, start + window_size)
        yield start, stop
        if stop == length:
            break
        start = stop - overlap


def _freeze(model: torch.nn.Module, device: torch.device) -> None:
    model.to(device)
    model.requires_grad_(False)
    model.eval()


def _check_loaded_backbone(info: dict[str, Any], prefixes: tuple[str, ...]) -> None:
    """Reject incomplete/incompatible loads; unused pretraining heads are allowed."""
    missing = [str(key) for key in info.get("missing_keys", ())]
    critical = [key for key in missing if key.startswith(prefixes)]
    if critical or info.get("mismatched_keys") or info.get("error_msgs"):
        raise ValueError(
            "Checkpoint did not load a complete pretrained backbone: "
            f"missing={critical}, mismatched={info.get('mismatched_keys', [])}, "
            f"errors={info.get('error_msgs', [])}"
        )


def _local_file_provenance(directory: Path) -> dict[str, str]:
    patterns = (
        "config.json", "vocab.txt", "spiece.model", "tokenizer*.json",
        "special_tokens_map.json", "pytorch_model*.bin", "model*.safetensors",
        "*.index.json",
    )
    files = {path for pattern in patterns for path in directory.glob(pattern) if path.is_file()}
    return {path.name: _sha256(path) for path in sorted(files)}


class ReactionTokenEncoder:
    """RXNFP BERT states for every token, with molecular topology kept separately.

    ``checkpoint`` must contain config.json, vocab.txt and pytorch_model.bin (or
    model.safetensors). Both official bert_pretrained and bert_ft are accepted.
    No pretraining objective is inferred from an arbitrary directory's name.
    Long inputs use overlapping windows of max_position_embeddings - 2 tokens;
    the two positions reserved for CLS/SEP never enter the returned features.
    Window states for the same original token are averaged. Cross-window global
    context is left to the trainable interaction model, not claimed as pretrained.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "cpu",
        overlap: int = 128,
        *,
        local_files_only: bool = True,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        required = (self.checkpoint / "config.json", self.checkpoint / "vocab.txt")
        if not self.checkpoint.is_dir() or not all(path.is_file() for path in required):
            raise ValueError("RXNFP checkpoint must be a local directory with config.json/vocab.txt")
        if not any((self.checkpoint / name).is_file() for name in (
            "pytorch_model.bin", "model.safetensors"
        )):
            raise ValueError("RXNFP checkpoint has no pretrained weight file")
        self.vocab = (self.checkpoint / "vocab.txt").read_text(encoding="utf-8").splitlines()
        if len(set(self.vocab)) != len(self.vocab) or any(not token for token in self.vocab):
            raise ValueError("RXNFP vocabulary must have unique nonempty tokens")
        self.token_to_id = {token: index for index, token in enumerate(self.vocab)}
        for token in ("[CLS]", "[SEP]", "[UNK]", "[PAD]"):
            if token not in self.token_to_id:
                raise ValueError(f"RXNFP vocabulary is missing {token}")
        config_data = json.loads((self.checkpoint / "config.json").read_text(encoding="utf-8"))
        if int(config_data.get("vocab_size", -1)) != len(self.vocab):
            raise ValueError("Checkpoint vocabulary size and vocab.txt disagree")
        self.device = torch.device(device)
        self.overlap = int(overlap)
        self.payload_size = int(config_data.get("max_position_embeddings", 0)) - 2
        # Validate window parameters before importing/loading optional models.
        list(window_slices(1, self.payload_size, self.overlap))
        transformers = _transformers()
        self.model, loading_info = transformers.BertModel.from_pretrained(
            str(self.checkpoint),
            local_files_only=local_files_only,
            add_pooling_layer=False,
            output_loading_info=True,
        )
        _check_loaded_backbone(loading_info, ("embeddings.", "encoder.", "bert."))
        _freeze(self.model, self.device)
        self.hidden_size = int(self.model.config.hidden_size)
        if self.model.get_input_embeddings().num_embeddings != len(self.vocab):
            raise ValueError("Loaded embedding table and checkpoint vocabulary disagree")
        self.metadata: dict[str, Any] = {
            "encoder": "RXNFP_BertModel",
            "checkpoint": str(self.checkpoint),
            "checkpoint_files_sha256": _local_file_provenance(self.checkpoint),
            "config": config_data,
            "transformers_version": transformers.__version__,
            "tokenizer": "RXNFP_regex_with_lossless_unmatched_characters",
            "tokenizer_source": (
                "https://github.com/rxn4chemistry/rxnfp/blob/master/rxnfp/tokenization.py"
            ),
            "tokenizer_pattern": RXNFP_TOKEN_PATTERN,
            "tokenizer_pattern_sha256": hashlib.sha256(RXNFP_TOKEN_PATTERN.encode()).hexdigest(),
            "atom_maps": "stripped_from_bracket_tokens_before_vocab_lookup",
            "hidden_size": self.hidden_size,
            "max_position_embeddings": self.payload_size + 2,
            "window_payload": self.payload_size,
            "window_overlap": self.overlap,
            "window_merge": "arithmetic_mean_per_original_token",
            "local_files_only": bool(local_files_only),
            "frozen": True,
            "unused_checkpoint_keys": [str(key) for key in loading_info.get("unexpected_keys", ())],
        }

    def encode(self, text: str) -> dict[str, Any]:
        reaction = tokenize_reaction(text)
        unknown = self.token_to_id["[UNK]"]
        unmatched = set(reaction.metadata["unmatched_token_indices"])
        ids = np.asarray([
            unknown if i in unmatched else self.token_to_id.get(token, unknown)
            for i, token in enumerate(reaction.tokens)
        ], dtype=np.int64)
        n_tokens = len(ids)
        states = np.zeros((n_tokens, self.hidden_size), dtype=np.float32)
        counts = np.zeros(n_tokens, dtype=np.int64)
        windows = list(window_slices(n_tokens, self.payload_size, self.overlap))
        # External callers cannot accidentally enable dropout while populating a cache.
        self.model.eval()
        with torch.no_grad():
            for start, stop in windows:
                window_ids = [self.token_to_id["[CLS]"], *ids[start:stop].tolist(),
                              self.token_to_id["[SEP]"]]
                input_ids = torch.tensor([window_ids], dtype=torch.long, device=self.device)
                output = self.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
                token_states = output.last_hidden_state[0, 1:-1]
                if tuple(token_states.shape) != (stop - start, self.hidden_size):
                    raise ValueError("RXNFP returned a token shape inconsistent with its input")
                states[start:stop] += token_states.detach().float().cpu().numpy()
                counts[start:stop] += 1
        if np.any(counts == 0):
            raise RuntimeError("Windowing left reaction tokens uncovered")
        states /= counts[:, None]
        if not np.isfinite(states).all():
            raise ValueError("RXNFP returned non-finite states")
        metadata = copy.deepcopy(self.metadata)
        metadata.update(reaction.metadata)
        metadata.update({
            "window_specs": [[start, stop] for start, stop in windows],
            "window_count": len(windows),
            "oov_token_count": int(np.count_nonzero(ids == unknown)),
            "oov_token_indices": np.flatnonzero(ids == unknown).tolist(),
            "tokens": list(reaction.tokens),
        })
        return {
            "reaction_states": states,
            "roles": reaction.roles,
            "atom_mask": reaction.atom_mask,
            "topology_mask": reaction.topology_mask,
            "edge_index": reaction.edge_index,
            "edge_types": reaction.edge_types,
            "metadata": metadata,
        }


class ProteinResidueEncoder:
    """Frozen ProtT5 encoder returning one 1024-dimensional state per residue.

    Input must be an unspaced amino-acid sequence. U/Z/O/B become X, as in the
    ProtTrans feature extraction recipe; positions are retained and recorded.
    Long proteins use overlapping windows (default 1000 residues, overlap 128),
    followed by arithmetic averaging at each original residue. This preserves
    all residues but does not claim full-protein attention across windows.
    ``local_files_only=False`` explicitly allows the usual HF checkpoint download.
    """

    def __init__(
        self,
        model_name_or_path: str | Path,
        device: str = "cpu",
        window_size: int = 1000,
        overlap: int = 128,
        *,
        local_files_only: bool = True,
        revision: str | None = None,
    ) -> None:
        self.window_size = int(window_size)
        self.overlap = int(overlap)
        list(window_slices(1, self.window_size, self.overlap))
        self.device = torch.device(device)
        source = str(model_name_or_path)
        kwargs: dict[str, Any] = {"local_files_only": local_files_only}
        if revision is not None:
            kwargs["revision"] = revision
        transformers = _transformers()
        self.tokenizer = transformers.T5Tokenizer.from_pretrained(
            source, do_lower_case=False, **kwargs,
        )
        self.model, loading_info = transformers.T5EncoderModel.from_pretrained(
            source, output_loading_info=True, **kwargs,
        )
        _check_loaded_backbone(loading_info, ("shared.", "encoder."))
        self.hidden_size = int(self.model.config.d_model)
        if self.hidden_size != 1024:
            raise ValueError(f"Expected ProtT5 residue width 1024, found {self.hidden_size}")
        _freeze(self.model, self.device)
        local_source = Path(source).expanduser()
        self.metadata: dict[str, Any] = {
            "encoder": "ProtT5_T5EncoderModel",
            "checkpoint": str(local_source.resolve()) if local_source.is_dir() else source,
            "checkpoint_files_sha256": (
                _local_file_provenance(local_source) if local_source.is_dir() else {}
            ),
            "config": self.model.config.to_dict(),
            "tokenizer_class": type(self.tokenizer).__name__,
            "tokenizer_name_or_path": str(getattr(self.tokenizer, "name_or_path", source)),
            "tokenizer_vocab_size": int(len(self.tokenizer)),
            "requested_revision": revision,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "transformers_version": transformers.__version__,
            "hidden_size": self.hidden_size,
            "window_payload": self.window_size,
            "window_overlap": self.overlap,
            "window_merge": "arithmetic_mean_per_original_residue",
            "local_files_only": bool(local_files_only),
            "frozen": True,
        }

    def encode(self, sequence: str) -> dict[str, Any]:
        if not isinstance(sequence, str) or not sequence or not sequence.isascii():
            raise ValueError("Protein input must be a nonempty ASCII amino-acid sequence")
        if not sequence.isalpha():
            raise ValueError("Protein input must contain letters only, without spaces or FASTA headers")
        normalised = sequence.upper()
        changed = [i for i, residue in enumerate(normalised) if residue in "UZOB"]
        normalised = normalised.translate(str.maketrans({residue: "X" for residue in "UZOB"}))
        n_residues = len(normalised)
        states = np.zeros((n_residues, self.hidden_size), dtype=np.float32)
        counts = np.zeros(n_residues, dtype=np.int64)
        unknown_positions: set[int] = set()
        windows = list(window_slices(n_residues, self.window_size, self.overlap))
        self.model.eval()
        with torch.no_grad():
            for start, stop in windows:
                encoded = self.tokenizer(
                    " ".join(normalised[start:stop]),
                    add_special_tokens=True,
                    return_special_tokens_mask=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                )
                if "special_tokens_mask" not in encoded:
                    raise ValueError("ProtT5 tokenizer must identify its special tokens")
                special = encoded.pop("special_tokens_mask")[0].to(dtype=torch.bool)
                keep = ~special
                if "attention_mask" in encoded:
                    keep &= encoded["attention_mask"][0].to(dtype=torch.bool)
                if int(keep.sum()) != stop - start:
                    raise ValueError("ProtT5 tokenizer does not yield exactly one token per residue")
                token_ids = encoded["input_ids"][0][keep]
                unk_id = getattr(self.tokenizer, "unk_token_id", None)
                if unk_id is not None:
                    unknown_positions.update(start + i for i in torch.where(token_ids == unk_id)[0].tolist())
                model_inputs = {key: value.to(self.device) for key, value in encoded.items()}
                output = self.model(**model_inputs)
                residue_states = output.last_hidden_state[0, keep.to(self.device)]
                if tuple(residue_states.shape) != (stop - start, self.hidden_size):
                    raise ValueError("ProtT5 returned a residue shape inconsistent with its input")
                states[start:stop] += residue_states.detach().float().cpu().numpy()
                counts[start:stop] += 1
        if np.any(counts == 0):
            raise RuntimeError("Windowing left protein residues uncovered")
        states /= counts[:, None]
        if not np.isfinite(states).all():
            raise ValueError("ProtT5 returned non-finite states")
        metadata = copy.deepcopy(self.metadata)
        metadata.update({
            "raw_text_sha256": hashlib.sha256(sequence.encode("utf-8")).hexdigest(),
            "normalised_text_sha256": hashlib.sha256(normalised.encode("utf-8")).hexdigest(),
            "residue_count": n_residues,
            "uzob_to_x_positions": changed,
            "uppercase_normalised": sequence != sequence.upper(),
            "unknown_residue_positions": sorted(unknown_positions),
            "window_specs": [[start, stop] for start, stop in windows],
            "window_count": len(windows),
        })
        return {"residue_states": states, "metadata": metadata}
