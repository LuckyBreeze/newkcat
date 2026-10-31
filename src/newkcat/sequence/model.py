"""Protein + reaction-sequence regression, without the original chemical towers.

The sequence encoder is deliberately small and does not imply pretrained chemical
knowledge. Absolute sinusoidal positions describe string order, not molecular
distance. Edit-conditioned pooling reads only grammar-derived changed-token spans;
it is not atom--residue alignment or a transition-state model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch
from torch import Tensor, nn


PAD_ID, BOS_ID, EOS_ID, MASK_ID = 0, 1, 2, 3
VOCAB_SIZE = 98  # Four special IDs plus printable non-whitespace ASCII 33..126.
ENCODER_FORMAT_VERSION = 1


@dataclass(frozen=True)
class SequenceModelConfig:
    vocab_size: int = VOCAB_SIZE
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_length: int = 2048
    pooling: str = "mean"
    protein_dim: int = 1024
    head_dim: int = 256

    def __post_init__(self) -> None:
        for name in ("vocab_size", "d_model", "nhead", "num_layers", "dim_feedforward",
                     "max_length", "protein_dim", "head_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"model.{name} must be a positive integer")
        if self.vocab_size != VOCAB_SIZE:
            raise ValueError(f"The fixed reaction tokenizer requires vocab_size={VOCAB_SIZE}")
        if self.d_model % self.nhead:
            raise ValueError("model.d_model must be divisible by model.nhead")
        if self.max_length < 3:
            raise ValueError("model.max_length must allow BOS, a chemical character, and EOS")
        if (isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float))
                or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1):
            raise ValueError("model.dropout must be finite and lie in [0, 1)")
        if self.pooling not in {"mean", "edit_conditioned"}:
            raise ValueError("model.pooling must be mean or edit_conditioned")

    def encoder_config(self) -> dict:
        """The fields that determine encoder state and forward semantics."""
        return {name: value for name, value in asdict(self).items()
                if name not in {"pooling", "protein_dim", "head_dim"}}

    def encoder_fingerprint(self) -> str:
        encoded = json.dumps(self.encoder_config(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def chemical_mask(input_ids: Tensor, attention_mask: Tensor) -> Tensor:
    """Chemical characters and MLM masks, excluding BOS/EOS/PAD."""
    return attention_mask & (input_ids != PAD_ID) & (input_ids != BOS_ID) & (input_ids != EOS_ID)


def _masked_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    weights = logits.masked_fill(~mask, torch.finfo(logits.dtype).min).softmax(dim=-1)
    weights = weights.masked_fill(~mask, 0)
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class ReactionEncoder(nn.Module):
    """Bidirectional character Transformer shared by regression and MLM.

    Padding is masked in attention and zeroed at the output. Encoding a reaction
    alone or with extra *right* padding gives the same valid-token representations
    in evaluation mode, up to floating-point roundoff. Different serializations
    of the same reaction are not guaranteed to have identical embeddings.
    """

    def __init__(self, config: SequenceModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=PAD_ID)
        positions = torch.arange(config.max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, config.d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / config.d_model)
        )
        positional = torch.zeros(config.max_length, config.d_model)
        positional[:, 0::2] = torch.sin(positions * frequencies)
        positional[:, 1::2] = torch.cos(positions * frequencies[:config.d_model // 2])
        self.register_buffer("positions", positional, persistent=True)
        self.input_dropout = nn.Dropout(config.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.nhead,
            dim_feedforward=config.dim_feedforward, dropout=config.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=config.num_layers, norm=nn.LayerNorm(config.d_model),
            enable_nested_tensor=False,
        )
        # TransformerEncoder clones its prototype. Initialize each layer separately
        # so layers start with distinct attention and feed-forward parameters.
        for block in self.transformer.layers:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            nn.init.zeros_(block.self_attn.in_proj_bias)
            for linear in (block.self_attn.out_proj, block.linear1, block.linear2):
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)

    def _validate(self, input_ids: Tensor, attention_mask: Tensor) -> None:
        if (not isinstance(input_ids, Tensor) or input_ids.dtype != torch.int64
                or input_ids.ndim != 2 or input_ids.shape[0] < 1 or input_ids.shape[1] < 1):
            raise ValueError("input_ids must be int64 [B, L] with B and L >= 1")
        if (not isinstance(attention_mask, Tensor) or attention_mask.dtype != torch.bool
                or attention_mask.shape != input_ids.shape):
            raise ValueError("attention_mask must be boolean [B, L]")
        if input_ids.device != attention_mask.device:
            raise ValueError("Token IDs and attention mask must share a device")
        if input_ids.device != self.token_embedding.weight.device:
            raise ValueError("Token IDs and encoder parameters must share a device")
        if input_ids.shape[1] > self.config.max_length:
            raise ValueError("Sequence exceeds model.max_length; silent truncation is forbidden")
        if bool(((input_ids < 0) | (input_ids >= self.config.vocab_size)).any()):
            raise ValueError("Token ID outside the fixed tokenizer vocabulary")
        if bool(((input_ids == PAD_ID) & attention_mask).any()):
            raise ValueError("PAD tokens must not be visible to attention")
        if not bool(chemical_mask(input_ids, attention_mask).any(dim=1).all()):
            raise ValueError("Every sequence must contain at least one chemical character")

    def encode(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        self._validate(input_ids, attention_mask)
        # Replace masked IDs, so arbitrary valid IDs in padded positions cannot leak.
        masked_ids = input_ids.masked_fill(~attention_mask, PAD_ID)
        states = self.token_embedding(masked_ids) * math.sqrt(self.config.d_model)
        states = states + self.positions[:input_ids.shape[1]].to(states.dtype)[None, :, :]
        states = self.input_dropout(states).masked_fill(~attention_mask[:, :, None], 0)
        states = self.transformer(states, src_key_padding_mask=~attention_mask)
        return states.masked_fill(~attention_mask[:, :, None], 0)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        return self.encode(input_ids, attention_mask)

    def export_encoder(self, tokenizer_fingerprint: str) -> dict:
        """A CPU-only tensor payload suitable for torch.save / weights_only load."""
        if not isinstance(tokenizer_fingerprint, str) or not tokenizer_fingerprint:
            raise ValueError("A nonempty tokenizer fingerprint is required")
        return {
            "format_version": ENCODER_FORMAT_VERSION,
            "encoder_config": self.config.encoder_config(),
            "encoder_fingerprint": self.config.encoder_fingerprint(),
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "encoder_state": {key: value.detach().cpu().clone()
                              for key, value in self.state_dict().items()},
        }

    def load_encoder(self, payload: Mapping, tokenizer_fingerprint: str) -> None:
        """Validate all metadata and tensors before mutating the current encoder."""
        required = {"format_version", "encoder_config", "encoder_fingerprint",
                    "tokenizer_fingerprint", "encoder_state"}
        if not isinstance(payload, Mapping) or not required.issubset(payload):
            raise ValueError("Not a complete exported reaction encoder")
        if (type(payload["format_version"]) is not int
                or payload["format_version"] != ENCODER_FORMAT_VERSION):
            raise ValueError("Unsupported reaction encoder format version")
        if (payload["encoder_config"] != self.config.encoder_config()
                or payload["encoder_fingerprint"] != self.config.encoder_fingerprint()):
            raise ValueError("Pretrained encoder configuration does not match the current encoder")
        if (not isinstance(tokenizer_fingerprint, str) or not tokenizer_fingerprint
                or payload["tokenizer_fingerprint"] != tokenizer_fingerprint):
            raise ValueError("Pretrained tokenizer fingerprint does not match")
        state = payload["encoder_state"]
        destination = self.state_dict()
        if not isinstance(state, Mapping) or state.keys() != destination.keys():
            raise ValueError("Pretrained encoder state has missing or unexpected keys")
        for key, value in state.items():
            if not isinstance(value, Tensor) or value.shape != destination[key].shape:
                raise ValueError(f"Invalid pretrained tensor shape for {key}")
            if value.dtype != destination[key].dtype:
                raise ValueError(f"Invalid pretrained tensor dtype for {key}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"Nonfinite pretrained tensor {key}")
        self.load_state_dict(state, strict=True)


class ReactionSequenceModel(nn.Module):
    """Regress log10(kcat / s^-1) from a protein cache and one reaction encoder."""

    def __init__(self, config: SequenceModelConfig):
        super().__init__()
        self.config = config
        self.encoder = ReactionEncoder(config)
        self._encoder_frozen = False
        self.protein_norm = nn.LayerNorm(config.protein_dim)
        self.protein_projection = nn.Sequential(
            nn.Linear(config.protein_dim, config.head_dim), nn.GELU(),
            nn.LayerNorm(config.head_dim),
        )
        chemical_width = config.d_model
        if config.pooling == "edit_conditioned":
            self.enzyme_query = nn.Linear(config.head_dim, config.d_model, bias=False)
            self.edit_key = nn.Linear(config.d_model, config.d_model, bias=False)
            self.edit_value = nn.Linear(config.d_model, config.d_model, bias=False)
            self.local_projection = nn.Sequential(
                nn.Linear(config.d_model, config.d_model), nn.GELU(),
                nn.LayerNorm(config.d_model),
            )
            chemical_width *= 2
        self.chemical_projection = nn.Sequential(
            nn.Linear(chemical_width, config.head_dim), nn.GELU(),
            nn.LayerNorm(config.head_dim),
        )
        self.regressor = nn.Sequential(
            nn.Linear(2 * config.head_dim, config.head_dim), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(config.head_dim, 1),
        )

    def set_encoder_frozen(self, frozen: bool = True) -> None:
        if type(frozen) is not bool:
            raise ValueError("frozen must be a boolean")
        self._encoder_frozen = frozen
        self.encoder.requires_grad_(not frozen)
        self.encoder.train(self.training and not frozen)

    def train(self, mode: bool = True):
        super().train(mode)
        if self._encoder_frozen:
            self.encoder.eval()
        return self

    def _validate_batch(self, batch) -> None:
        protein = batch.protein
        if (not isinstance(protein, Tensor) or protein.ndim != 2
                or protein.shape != (batch.input_ids.shape[0], self.config.protein_dim)
                or not protein.is_floating_point()):
            raise ValueError(f"protein must be a floating [B, {self.config.protein_dim}] tensor")
        if not bool(torch.isfinite(protein).all()):
            raise ValueError("Protein features must be finite")
        if (protein.device != batch.input_ids.device
                or protein.device != self.protein_projection[0].weight.device):
            raise ValueError("Protein features, sequence tokens and model must share a device")
        if protein.dtype != self.protein_projection[0].weight.dtype:
            raise ValueError("Protein features and regression parameters must share a dtype")
        edits = batch.edit_mask
        if (not isinstance(edits, Tensor) or edits.shape != batch.input_ids.shape
                or edits.dtype != torch.bool or edits.device != batch.input_ids.device):
            raise ValueError("edit_mask must be boolean [B, L] on the token device")
        allowed = chemical_mask(batch.input_ids, batch.attention_mask)
        if bool((edits & ~allowed).any()):
            raise ValueError("Edit spans cannot include special tokens or padding")
        target = getattr(batch, "target", None)
        if target is not None:
            if (not isinstance(target, Tensor) or target.shape != (protein.shape[0],)
                    or not target.is_floating_point() or not bool(torch.isfinite(target).all())):
                raise ValueError("target must be a finite floating [B] vector")
            if target.device != protein.device:
                raise ValueError("Target and inputs must share a device")

    def forward(self, batch, return_diagnostics: bool = False):
        self.encoder._validate(batch.input_ids, batch.attention_mask)
        self._validate_batch(batch)
        states = self.encoder.encode(batch.input_ids, batch.attention_mask)
        protein = self.protein_projection(self.protein_norm(batch.protein))
        mask = chemical_mask(batch.input_ids, batch.attention_mask)
        pooled = states.masked_fill(~mask[:, :, None], 0).sum(dim=1)
        pooled = pooled / mask.sum(dim=1, keepdim=True).to(states.dtype)
        edit_weights = states.new_zeros(batch.input_ids.shape)
        local = torch.zeros_like(pooled)
        active = torch.zeros(mask.shape[0], dtype=torch.bool, device=mask.device)
        if self.config.pooling == "edit_conditioned":
            active = batch.edit_mask.any(dim=1)
            query = self.enzyme_query(protein)
            logits = torch.einsum("bd,bld->bl", query, self.edit_key(states))
            edit_weights = _masked_softmax(logits / math.sqrt(self.config.d_model), batch.edit_mask)
            selected = (self.edit_value(states) * edit_weights[:, :, None]).sum(dim=1)
            local = self.local_projection(selected).masked_fill(~active[:, None], 0)
            chemical = self.chemical_projection(torch.cat((pooled, local), dim=-1))
        else:
            chemical = self.chemical_projection(pooled)
        prediction = self.regressor(torch.cat((protein, chemical), dim=-1)).squeeze(-1)
        if prediction.shape != (batch.input_ids.shape[0],):
            raise RuntimeError("Internal error: kcat prediction must be a [B] vector")
        if return_diagnostics:
            return prediction, {"pooled": pooled, "local": local, "edit_weights": edit_weights,
                                "chemical_mask": mask, "edit_mask": batch.edit_mask,
                                "edit_active": active, "chemical": chemical}
        return prediction


class ReactionMLM(nn.Module):
    """Masked character prediction for optional self-supervised pretraining.

    The trainer controls masking and applies loss only to the selected character
    positions. No kcat/protein values enter this module. The MLM head is discarded
    when exporting the shared encoder to the regression model.
    """

    def __init__(self, config: SequenceModelConfig):
        super().__init__()
        self.config = config
        self.encoder = ReactionEncoder(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        states = self.encoder.encode(input_ids, attention_mask)
        logits = self.lm_head(states)
        return logits.masked_fill(~attention_mask[:, :, None], 0)
