"""Controlled readout ablations on the original reaction Transformer.

The default projected/mean/concat path preserves the original R model's state
keys, parameter initialization and arithmetic. ``full`` retains the normalized
protein vector through fusion. Learned-query slots are fixed-size summaries of
all chemical characters; they are not atom, residue or reaction-center labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..sequence.model import (
    ReactionEncoder, SequenceModelConfig, VOCAB_SIZE, chemical_mask,
)


@dataclass(frozen=True)
class ReadoutModelConfig:
    vocab_size: int = VOCAB_SIZE
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_length: int = 2048
    protein_dim: int = 1024
    head_dim: int = 256
    chemical_dim: int | None = 256
    protein_readout: str = "projected"
    readout: str = "mean"
    num_queries: int = 4
    condition_queries: bool = False
    fusion: str = "concat"
    fusion_num_layers: int = 1
    max_flatten_parameters: int = 20_000_000

    def __post_init__(self) -> None:
        # Keep the encoder validation and exported checkpoint contract identical.
        self.to_sequence_config()
        for name in ("num_queries", "fusion_num_layers", "max_flatten_parameters"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"model.{name} must be a positive integer")
        if (self.chemical_dim is not None
                and (type(self.chemical_dim) is not int or self.chemical_dim < 1)):
            raise ValueError("model.chemical_dim must be a positive integer or null")
        if self.protein_readout not in {"projected", "full"}:
            raise ValueError("protein_readout must be projected or full")
        if self.readout not in {"mean", "mean_max", "queries", "flatten"}:
            raise ValueError("readout must be mean, mean_max, queries or flatten")
        if type(self.condition_queries) is not bool:
            raise ValueError("condition_queries must be a boolean")
        if self.condition_queries and self.readout != "queries":
            raise ValueError("condition_queries requires readout=queries")
        if self.fusion not in {"concat", "transformer"}:
            raise ValueError("fusion must be concat or transformer")
        if self.fusion == "transformer" and self.readout != "queries":
            raise ValueError("transformer fusion requires readout=queries")
        parameters = self.flatten_dense_parameters()
        if parameters > self.max_flatten_parameters:
            raise ValueError(
                f"Flatten's first dense reducer needs {parameters:,} parameters, exceeding "
                f"max_flatten_parameters={self.max_flatten_parameters:,}; choose a smaller "
                "declared max_length/width or explicitly raise the finite budget"
            )

    def to_sequence_config(self) -> SequenceModelConfig:
        return SequenceModelConfig(
            vocab_size=self.vocab_size, d_model=self.d_model, nhead=self.nhead,
            num_layers=self.num_layers, dim_feedforward=self.dim_feedforward,
            dropout=self.dropout, max_length=self.max_length, pooling="mean",
            protein_dim=self.protein_dim, head_dim=self.head_dim,
        )

    def encoder_config(self) -> dict:
        return self.to_sequence_config().encoder_config()

    def encoder_fingerprint(self) -> str:
        return self.to_sequence_config().encoder_fingerprint()

    @property
    def protein_width(self) -> int:
        return self.protein_dim if self.protein_readout == "full" else self.head_dim

    @property
    def readout_width(self) -> int:
        multiplier = {"mean": 1, "mean_max": 2, "queries": self.num_queries,
                      "flatten": self.max_length}[self.readout]
        return multiplier * self.d_model

    @property
    def chemical_width(self) -> int:
        return self.readout_width if self.chemical_dim is None else self.chemical_dim

    def flatten_dense_parameters(self) -> int:
        """The actual first dense reduction after flattening, including its bias."""
        if self.readout != "flatten":
            return 0
        if self.chemical_dim is not None:
            return self.readout_width * self.chemical_dim + self.chemical_dim
        return (self.readout_width + self.protein_width) * self.head_dim + self.head_dim


class ReactionReadoutModel(nn.Module):
    """Protein cache + one reaction encoder + a configurable, explicit readout."""

    def __init__(self, config: ReadoutModelConfig):
        super().__init__()
        if not isinstance(config, ReadoutModelConfig):
            raise TypeError("config must be a ReadoutModelConfig")
        self.config = config
        # Repeat the size gate before any allocation, even for a mutated config.
        if config.flatten_dense_parameters() > config.max_flatten_parameters:
            raise ValueError("Flatten dense allocation exceeds max_flatten_parameters")
        self.encoder = ReactionEncoder(config.to_sequence_config())
        self._encoder_frozen = False
        self.protein_norm = nn.LayerNorm(config.protein_dim)
        self.protein_projection = (
            nn.Sequential(
                nn.Linear(config.protein_dim, config.head_dim), nn.GELU(),
                nn.LayerNorm(config.head_dim),
            ) if config.protein_readout == "projected" else nn.Identity()
        )
        if config.readout == "queries":
            self.readout_queries = nn.Parameter(torch.empty(config.num_queries, config.d_model))
            nn.init.normal_(self.readout_queries, std=0.02)
            self.query_key = nn.Linear(config.d_model, config.d_model, bias=False)
            self.query_value = nn.Linear(config.d_model, config.d_model, bias=False)
            if config.condition_queries:
                self.enzyme_query = nn.Linear(config.protein_width, config.d_model, bias=False)
        if config.fusion == "transformer":
            self.fusion_protein = nn.Linear(config.protein_width, config.d_model)
            self.fusion_types = nn.Embedding(2, config.d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=config.d_model, nhead=config.nhead,
                dim_feedforward=config.dim_feedforward, dropout=config.dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.fusion_transformer = nn.TransformerEncoder(
                layer, num_layers=config.fusion_num_layers,
                norm=nn.LayerNorm(config.d_model), enable_nested_tensor=False,
            )
            for block in self.fusion_transformer.layers:
                nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
                nn.init.zeros_(block.self_attn.in_proj_bias)
                for linear in (block.self_attn.out_proj, block.linear1, block.linear2):
                    nn.init.xavier_uniform_(linear.weight)
                    nn.init.zeros_(linear.bias)
        self.chemical_projection = (
            nn.Sequential(
                nn.Linear(config.readout_width, config.chemical_dim), nn.GELU(),
                nn.LayerNorm(config.chemical_dim),
            ) if config.chemical_dim is not None else nn.Identity()
        )
        self.regressor = nn.Sequential(
            nn.Linear(config.protein_width + config.chemical_width, config.head_dim),
            nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.head_dim, 1),
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
        self.encoder._validate(batch.input_ids, batch.attention_mask)
        protein = batch.protein
        weight = self.protein_norm.weight
        if (not isinstance(protein, Tensor) or protein.ndim != 2
                or protein.shape != (batch.input_ids.shape[0], self.config.protein_dim)
                or not protein.is_floating_point()):
            raise ValueError(f"protein must be floating [B, {self.config.protein_dim}]")
        if not bool(torch.isfinite(protein).all()):
            raise ValueError("Protein features must be finite")
        if protein.device != weight.device or protein.device != batch.input_ids.device:
            raise ValueError("Protein, token inputs and model must share a device")
        if protein.dtype != weight.dtype:
            raise ValueError("Protein and model must share a floating dtype")
        edits = batch.edit_mask
        if (not isinstance(edits, Tensor) or edits.shape != batch.input_ids.shape
                or edits.dtype != torch.bool or edits.device != batch.input_ids.device):
            raise ValueError("edit_mask must be boolean [B, L] on the token device")
        mask = chemical_mask(batch.input_ids, batch.attention_mask)
        if bool((edits & ~mask).any()):
            raise ValueError("Edit masks cannot include special tokens or padding")
        target = getattr(batch, "target", None)
        if target is not None:
            if (not isinstance(target, Tensor) or target.shape != (protein.shape[0],)
                    or not target.is_floating_point() or not bool(torch.isfinite(target).all())):
                raise ValueError("target must be a finite floating [B] vector")
            if target.device != protein.device:
                raise ValueError("Target and inputs must share a device")

    def forward(self, batch, return_diagnostics: bool = False):
        self._validate_batch(batch)
        states = self.encoder.encode(batch.input_ids, batch.attention_mask)
        protein = self.protein_projection(self.protein_norm(batch.protein))
        mask = chemical_mask(batch.input_ids, batch.attention_mask)
        masked = states.masked_fill(~mask[:, :, None], 0)
        mean = masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).to(states.dtype)
        weights = states.new_zeros((states.shape[0], 0, states.shape[1]))
        slots = states.new_zeros((states.shape[0], 0, self.config.d_model))
        fusion_tokens = slots
        if self.config.readout == "mean":
            pooled = mean
        elif self.config.readout == "mean_max":
            maximum = states.masked_fill(~mask[:, :, None], -torch.inf).amax(dim=1)
            pooled = torch.cat((mean, maximum), dim=-1)
        elif self.config.readout == "flatten":
            # The declared fixed width is independent of the current batch's length.
            fixed = F.pad(masked, (0, 0, 0, self.config.max_length - masked.shape[1]))
            pooled = fixed.reshape(fixed.shape[0], -1)
        else:
            queries = self.readout_queries[None, :, :].expand(states.shape[0], -1, -1)
            if self.config.condition_queries:
                queries = queries + self.enzyme_query(protein)[:, None, :]
            logits = torch.einsum("bkd,bld->bkl", queries, self.query_key(states))
            logits = logits / math.sqrt(self.config.d_model)
            weights = logits.masked_fill(~mask[:, None, :], -torch.inf).softmax(dim=-1)
            weights = weights.masked_fill(~mask[:, None, :], 0)
            slots = weights.matmul(self.query_value(states).masked_fill(~mask[:, :, None], 0))
            if self.config.fusion == "transformer":
                protein_token = self.fusion_protein(protein)[:, None, :]
                protein_token = protein_token + self.fusion_types.weight[0][None, None, :]
                chemical_tokens = slots + self.fusion_types.weight[1][None, None, :]
                fusion_tokens = self.fusion_transformer(
                    torch.cat((protein_token, chemical_tokens), dim=1)
                )
                # Keep the full direct protein path; read fused chemical slots only.
                slots = fusion_tokens[:, 1:, :]
            pooled = slots.reshape(slots.shape[0], -1)
        chemical = self.chemical_projection(pooled)
        prediction = self.regressor(torch.cat((protein, chemical), dim=-1)).squeeze(-1)
        if prediction.shape != (batch.input_ids.shape[0],):
            raise RuntimeError("Internal error: kcat prediction must be a [B] vector")
        if not bool(torch.isfinite(prediction).all()):
            raise ValueError("Model produced nonfinite predictions")
        if return_diagnostics:
            return prediction, {
                "protein": protein, "chemical": chemical, "pooled": pooled,
                "global_mean": mean, "chemical_mask": mask,
                "query_weights": weights, "slots": slots, "fusion_tokens": fusion_tokens,
            }
        return prediction
