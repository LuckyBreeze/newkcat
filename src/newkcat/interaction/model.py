"""Interact with complete frozen reaction-token and protein-residue states.

This is one chemical tower: bond-local attention augments the reaction token
states; it does not create a separate pooled GNN feature. Both directions of
cross-attention operate before readout and have multiple keys. Topology is a
molecular-neighborhood constraint, not a reaction-center annotation or an atom
mapping between the two reaction sides.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


NUM_BOND_TYPES = 7
REACTION_READOUT_NAMES = ("overall", "reactants", "products", "agents")


@dataclass(frozen=True)
class InteractionModelConfig:
    protein_dim: int = 1024
    reaction_dim: int = 256
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 512
    head_hidden_dim: int = 512
    head_bottleneck_dim: int = 128
    dropout: float = 0.1
    use_topology: bool = True
    use_cross_attention: bool = True
    residual_scale_init: float = 0.1

    def __post_init__(self) -> None:
        for name in ("protein_dim", "reaction_dim", "d_model", "nhead", "num_layers",
                     "dim_feedforward", "head_hidden_dim", "head_bottleneck_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.nhead:
            raise ValueError("d_model must be divisible by nhead")
        for name in ("use_topology", "use_cross_attention"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if (isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float))
                or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1):
            raise ValueError("dropout must be finite and lie in [0, 1)")
        if (isinstance(self.residual_scale_init, bool)
                or not isinstance(self.residual_scale_init, (int, float))
                or not math.isfinite(self.residual_scale_init)
                or not 0 < self.residual_scale_init <= 1):
            raise ValueError("residual_scale_init must be finite and lie in (0, 1]")

    @property
    def head_input_dim(self) -> int:
        # Four chemical summaries and one residue summary, plus the intact cache.
        return self.protein_dim + 5 * self.d_model


ModelConfig = InteractionModelConfig


def _masked_softmax(logits: Tensor, allowed: Tensor, dim: int = -1) -> Tensor:
    """An all-masked row is exactly zero, including in low precision."""
    # A finite sentinel avoids the undefined softmax([-inf, ... , -inf]).
    weights = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
    weights = weights.softmax(dim=dim).masked_fill(~allowed, 0)
    denominator = weights.sum(dim=dim, keepdim=True)
    return weights / denominator.clamp_min(torch.finfo(weights.dtype).tiny)


class MaskedMultiheadAttention(nn.Module):
    """Attention with explicit query/key masks and optional molecular edges."""

    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query_states: Tensor, key_states: Tensor,
                query_mask: Tensor, key_mask: Tensor,
                pair_mask: Tensor | None = None, bias: Tensor | None = None,
                return_weights: bool = False) -> tuple[Tensor, Tensor | None]:
        batch_size, query_length, width = query_states.shape
        key_length = key_states.shape[1]

        def split_heads(states: Tensor, projection: nn.Linear) -> Tensor:
            return projection(states).reshape(
                batch_size, states.shape[1], self.nhead, self.head_dim
            ).transpose(1, 2)

        query = split_heads(query_states, self.query)
        key = split_heads(key_states, self.key)
        value = split_heads(key_states, self.value)
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if bias is not None:
            logits = logits + bias
        allowed = query_mask[:, None, :, None] & key_mask[:, None, None, :]
        if pair_mask is not None:
            allowed = allowed & pair_mask[:, None, :, :]
        weights = _masked_softmax(logits, allowed)
        result = torch.matmul(self.dropout(weights), value)
        result = result.transpose(1, 2).reshape(batch_size, query_length, width)
        result = self.output(result)
        # output.bias must not resurrect empty local neighborhoods or padding.
        has_keys = allowed.any(dim=-1).squeeze(1)
        result = result.masked_fill(~has_keys[:, :, None], 0)
        if key_length == 0:  # Batch validation disallows this; documents the contract.
            raise ValueError("attention key dimension cannot be empty")
        return result, weights if return_weights else None


class InteractionBlock(nn.Module):
    """Local/global chemical updates, simultaneous bidirectional interaction, FFNs."""

    def __init__(self, config: InteractionModelConfig):
        super().__init__()
        width = config.d_model
        self.use_topology = config.use_topology
        self.use_cross_attention = config.use_cross_attention
        self.chemical_norm = nn.LayerNorm(width)
        self.global_attention = MaskedMultiheadAttention(width, config.nhead, config.dropout)
        self.global_scale = nn.Parameter(torch.full((width,), config.residual_scale_init))
        if self.use_topology:
            self.local_attention = MaskedMultiheadAttention(width, config.nhead, config.dropout)
            self.bond_bias = nn.Embedding(NUM_BOND_TYPES, config.nhead, padding_idx=0)
            nn.init.normal_(self.bond_bias.weight, std=0.02)
            with torch.no_grad():
                self.bond_bias.weight[0].zero_()
            self.local_scale = nn.Parameter(torch.full((width,), config.residual_scale_init))
        if self.use_cross_attention:
            self.reaction_cross_norm = nn.LayerNorm(width)
            self.protein_cross_norm = nn.LayerNorm(width)
            self.reaction_to_protein = MaskedMultiheadAttention(
                width, config.nhead, config.dropout
            )
            self.protein_to_reaction = MaskedMultiheadAttention(
                width, config.nhead, config.dropout
            )
            self.reaction_cross_scale = nn.Parameter(
                torch.full((width,), config.residual_scale_init)
            )
            self.protein_cross_scale = nn.Parameter(
                torch.full((width,), config.residual_scale_init)
            )
        self.reaction_ff_norm = nn.LayerNorm(width)
        self.protein_ff_norm = nn.LayerNorm(width)

        def feedforward() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(width, config.dim_feedforward), nn.GELU(),
                nn.Dropout(config.dropout), nn.Linear(config.dim_feedforward, width),
                nn.Dropout(config.dropout),
            )

        self.reaction_ff = feedforward()
        self.protein_ff = feedforward()
        self.reaction_ff_scale = nn.Parameter(torch.full((width,), config.residual_scale_init))
        self.protein_ff_scale = nn.Parameter(torch.full((width,), config.residual_scale_init))

    def forward(self, reaction: Tensor, protein: Tensor, reaction_mask: Tensor,
                protein_mask: Tensor, reliable_atoms: Tensor, bond_types: Tensor,
                return_diagnostics: bool = False) -> tuple[Tensor, Tensor, dict]:
        normalized_reaction = self.chemical_norm(reaction)
        global_update, _ = self.global_attention(
            normalized_reaction, normalized_reaction, reaction_mask, reaction_mask
        )
        local_update = torch.zeros_like(reaction)
        local_weights = None
        if self.use_topology:
            length = reaction.shape[1]
            diagonal = torch.eye(length, dtype=torch.bool, device=reaction.device)[None]
            # Self loops are generated only for reliable atomic tokens. Unknown or
            # unparsed tokens have a global path but no manufactured local edges.
            effective_types = bond_types.masked_fill(diagonal, 1)
            pairs = effective_types.gt(0)
            bias = self.bond_bias(effective_types).permute(0, 3, 1, 2)
            local_update, local_weights = self.local_attention(
                normalized_reaction, normalized_reaction,
                reliable_atoms, reliable_atoms, pair_mask=pairs, bias=bias,
                return_weights=return_diagnostics,
            )
            reaction = reaction + self.local_scale * local_update
        reaction = reaction + self.global_scale * global_update
        reaction = reaction.masked_fill(~reaction_mask[:, :, None], 0)

        reaction_cross_weights = protein_cross_weights = None
        if self.use_cross_attention:
            # Both directions see the same pre-cross-attention states. In
            # particular protein queries have T reaction keys, not one CLS key.
            reaction_source = self.reaction_cross_norm(reaction)
            protein_source = self.protein_cross_norm(protein)
            reaction_update, reaction_cross_weights = self.reaction_to_protein(
                reaction_source, protein_source, reaction_mask, protein_mask,
                return_weights=return_diagnostics,
            )
            protein_update, protein_cross_weights = self.protein_to_reaction(
                protein_source, reaction_source, protein_mask, reaction_mask,
                return_weights=return_diagnostics,
            )
            reaction = reaction + self.reaction_cross_scale * reaction_update
            protein = protein + self.protein_cross_scale * protein_update

        reaction = reaction + self.reaction_ff_scale * self.reaction_ff(
            self.reaction_ff_norm(reaction)
        )
        protein = protein + self.protein_ff_scale * self.protein_ff(self.protein_ff_norm(protein))
        reaction = reaction.masked_fill(~reaction_mask[:, :, None], 0)
        protein = protein.masked_fill(~protein_mask[:, :, None], 0)
        diagnostics = {}
        if return_diagnostics:
            diagnostics = {
                "local_update": local_update,
                "local_attention": local_weights,
                "reaction_to_protein_attention": reaction_cross_weights,
                "protein_to_reaction_attention": protein_cross_weights,
            }
        return reaction, protein, diagnostics


class MaskedQueryPool(nn.Module):
    """Read named sets; an absent role has an exact zero summary."""

    def __init__(self, d_model: int, num_queries: int):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(num_queries, d_model))
        nn.init.normal_(self.queries, std=0.02)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)

    def forward(self, states: Tensor, masks: Tensor) -> tuple[Tensor, Tensor]:
        scores = torch.einsum("qd,btd->bqt", self.queries, self.key(states))
        scores = scores / math.sqrt(states.shape[-1])
        weights = _masked_softmax(scores, masks)
        return torch.matmul(weights, self.value(states)), weights


class ReactionInteractionModel(nn.Module):
    """Predict a scalar log10(kcat) from cached full token and residue states.

    ``protein_global`` is concatenated unchanged into the final regressor.
    Reducing residue states to d_model is only the interaction pathway, not the
    removal of the original 1024-dimensional global path. Frozen pretrained
    encoders live outside this module; this module learns lightweight adapters.
    """

    def __init__(self, config: InteractionModelConfig):
        super().__init__()
        if not isinstance(config, InteractionModelConfig):
            raise TypeError("config must be an InteractionModelConfig")
        self.config = config
        self.protein_norm = nn.LayerNorm(config.protein_dim)
        self.protein_projection = nn.Linear(config.protein_dim, config.d_model)
        self.reaction_norm = nn.LayerNorm(config.reaction_dim)
        self.reaction_projection = (
            nn.Identity() if config.reaction_dim == config.d_model
            else nn.Linear(config.reaction_dim, config.d_model)
        )
        self.role_embedding = nn.Embedding(4, config.d_model)
        nn.init.normal_(self.role_embedding.weight, std=0.02)
        self.blocks = nn.ModuleList(InteractionBlock(config) for _ in range(config.num_layers))
        self.reaction_pool = MaskedQueryPool(config.d_model, len(REACTION_READOUT_NAMES))
        self.protein_pool = MaskedQueryPool(config.d_model, 1)
        self.regressor = nn.Sequential(
            nn.Linear(config.head_input_dim, config.head_hidden_dim), nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden_dim, config.head_bottleneck_dim), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(config.head_bottleneck_dim, 1),
        )

    def _validate_batch(self, batch: Mapping) -> None:
        if not isinstance(batch, Mapping):
            raise TypeError("batch must be a mapping of named tensors")
        required = ("protein_states", "protein_mask", "protein_global", "reaction_states",
                    "reaction_mask", "roles", "atom_mask", "topology_mask", "bond_types")
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(f"batch is missing required fields: {', '.join(missing)}")
        for name in required:
            if not isinstance(batch[name], Tensor):
                raise TypeError(f"{name} must be a tensor")
        reaction, protein = batch["reaction_states"], batch["protein_states"]
        if (reaction.ndim != 3 or reaction.shape[0] < 1 or reaction.shape[1] < 1
                or reaction.shape[2] != self.config.reaction_dim):
            raise ValueError("reaction_states must have shape [B, T, reaction_dim], B,T >= 1")
        batch_size, reaction_length = reaction.shape[:2]
        if (protein.ndim != 3 or protein.shape[0] != batch_size or protein.shape[1] < 1
                or protein.shape[2] != self.config.protein_dim):
            raise ValueError("protein_states must have shape [B, L, protein_dim], L >= 1")
        if batch["protein_global"].shape != (batch_size, self.config.protein_dim):
            raise ValueError("protein_global must have shape [B, protein_dim]")
        weight = self.protein_norm.weight
        for name in ("protein_states", "reaction_states", "protein_global"):
            value = batch[name]
            if not value.is_floating_point() or value.dtype != weight.dtype:
                raise ValueError(f"{name} must use the model's floating dtype")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains nonfinite values")
        for name in required:
            if batch[name].device != weight.device:
                raise ValueError(f"{name} must be on the same device as the model")
        for name in ("reaction_mask", "atom_mask", "topology_mask"):
            if batch[name].dtype != torch.bool or batch[name].shape != reaction.shape[:2]:
                raise ValueError(f"{name} must be boolean [B, T]")
        if (batch["protein_mask"].dtype != torch.bool
                or batch["protein_mask"].shape != protein.shape[:2]):
            raise ValueError("protein_mask must be boolean [B, L]")
        roles = batch["roles"]
        if roles.dtype != torch.int64 or roles.shape != reaction.shape[:2]:
            raise ValueError("roles must be int64 [B, T]")
        if bool(((roles < 0) | (roles > 3)).any()):
            raise ValueError("roles must lie in 0..3")
        bonds = batch["bond_types"]
        if (bonds.dtype != torch.int64
                or bonds.shape != (batch_size, reaction_length, reaction_length)):
            raise ValueError("bond_types must be int64 [B, T, T]")
        if bool(((bonds < 0) | (bonds >= NUM_BOND_TYPES)).any()):
            raise ValueError("bond_types must lie in 0..6")
        if bool((batch["atom_mask"] & ~batch["reaction_mask"]).any()):
            raise ValueError("atom_mask cannot include padding")
        if bool((batch["topology_mask"] & ~batch["atom_mask"]).any()):
            raise ValueError("topology_mask must be a subset of atom_mask")

    def forward(self, batch: Mapping, return_diagnostics: bool = False):
        self._validate_batch(batch)
        reaction_mask, protein_mask = batch["reaction_mask"], batch["protein_mask"]
        reaction = batch["reaction_states"].masked_fill(~reaction_mask[:, :, None], 0)
        protein = batch["protein_states"].masked_fill(~protein_mask[:, :, None], 0)
        reaction = self.reaction_projection(self.reaction_norm(reaction))
        reaction = reaction + self.role_embedding(batch["roles"])
        reaction = reaction.masked_fill(~reaction_mask[:, :, None], 0)
        protein = self.protein_projection(self.protein_norm(protein))
        protein = protein.masked_fill(~protein_mask[:, :, None], 0)
        reliable_atoms = batch["topology_mask"] & batch["atom_mask"] & reaction_mask
        block_diagnostics = []
        for block in self.blocks:
            reaction, protein, details = block(
                reaction, protein, reaction_mask, protein_mask,
                reliable_atoms, batch["bond_types"], return_diagnostics,
            )
            if return_diagnostics:
                block_diagnostics.append(details)

        roles = batch["roles"]
        role_masks = torch.stack((reaction_mask, reaction_mask & roles.eq(0),
                                  reaction_mask & roles.eq(2), reaction_mask & roles.eq(1)), dim=1)
        reaction_summaries, reaction_weights = self.reaction_pool(reaction, role_masks)
        protein_summary, protein_weights = self.protein_pool(protein, protein_mask[:, None, :])
        head_input = torch.cat((batch["protein_global"], reaction_summaries.flatten(1),
                                protein_summary[:, 0]), dim=-1)
        prediction = self.regressor(head_input).squeeze(-1)
        if not return_diagnostics:
            return prediction
        return prediction, {
            "reaction_states": reaction,
            "protein_states": protein,
            "reaction_summaries": reaction_summaries,
            "protein_summary": protein_summary[:, 0],
            "reaction_pool_weights": reaction_weights,
            "protein_pool_weights": protein_weights[:, 0],
            "head_input": head_input,
            "blocks": block_diagnostics,
        }
