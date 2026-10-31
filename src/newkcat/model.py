"""CataPro-compatible regression with a controlled center-relative graph branch."""

from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .position import NUM_DISTANCE_BINS, center_relative_profiles, distance_shell_pool
from .types import GraphBatch

MODES = ("baseline", "global", "center", "shell", "conditioned", "learned")
CENTER_MODES = frozenset(("center", "shell", "conditioned"))
GLOBAL_FEATURE_DIM = 1024 + 768 + 167


@dataclass(frozen=True)
class ModelConfig:
    atom_dim: int
    bond_dim: int
    mode: str = "conditioned"
    hidden_dim: int = 128
    num_layers: int = 3
    local_dim: int = 64
    dropout: float = 0.0
    use_ec2: bool = False
    ec_vocab_size: int = 1
    ec_dim: int = 16

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        for name in ("atom_dim", "bond_dim", "hidden_dim", "num_layers", "local_dim",
                     "ec_vocab_size", "ec_dim"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float))
                or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1):
            raise ValueError("dropout must lie in [0, 1)")
        if not isinstance(self.use_ec2, bool):
            raise ValueError("use_ec2 must be a boolean")


def _masked_softmax(logits: Tensor, mask: Tensor, dim: int) -> Tensor:
    """Masked softmax with exactly zero output when every entry is masked."""
    weights = logits.masked_fill(~mask, torch.finfo(logits.dtype).min).softmax(dim=dim)
    weights = weights.masked_fill(~mask, 0)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-12)


class EdgeMessageLayer(nn.Module):
    """Edge-conditioned mean aggregation; no dependency on atom/edge order."""

    def __init__(self, hidden_dim: int, bond_dim: int, dropout: float):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim + bond_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, states: Tensor, edges: Tensor, edge_features: Tensor,
                atom_mask: Tensor) -> Tensor:
        shape = states.shape
        flat = states.reshape(-1, shape[-1])
        aggregate = torch.zeros_like(flat)
        degree = flat.new_zeros((flat.shape[0], 1))
        if edges.shape[1]:
            source, destination = edges
            messages = self.message(torch.cat((flat[source], edge_features), dim=-1))
            aggregate.index_add_(0, destination, messages)
            degree.index_add_(0, destination, flat.new_ones((destination.numel(), 1)))
        aggregate = aggregate / degree.clamp_min(1)
        updated = self.norm(flat + self.update(torch.cat((flat, aggregate), dim=-1)))
        return updated.reshape(shape).masked_fill(~atom_mask[:, :, None], 0)


class NewKcat(nn.Module):
    """Predict log10(kcat / s^-1), returning one scalar per sample.

    All graph ablations instantiate the same graph/readout modules. Inactive
    modules do not receive gradients in the corresponding ablation; parameter
    counts alone therefore do not establish equivalent effective capacity.
    The baseline contains only the original global-feature regressor (plus
    optional EC embedding for an equal-information comparison).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.prot_norm = nn.BatchNorm1d(1024)
        self.molt5_norm = nn.BatchNorm1d(768)
        extra_dim = 0 if config.mode == "baseline" else config.local_dim
        if config.use_ec2:
            self.ec_embedding = nn.Embedding(config.ec_vocab_size, config.ec_dim, padding_idx=0)
            extra_dim += config.ec_dim
        self.decoder = nn.Sequential(
            nn.Linear(GLOBAL_FEATURE_DIM + extra_dim, 256), nn.BatchNorm1d(256),
            nn.Dropout(config.dropout), nn.ReLU(),
        )
        # Sequential preserves the official CataPro 'out.0.*' state-dict keys.
        self.out = nn.Sequential(nn.Linear(256, 1))
        if config.mode == "baseline":
            return
        hidden = config.hidden_dim
        self.atom_encoder = nn.Linear(config.atom_dim, hidden)
        self.position_encoder = nn.Linear(NUM_DISTANCE_BINS, hidden, bias=False)
        self.graph_layers = nn.ModuleList(
            EdgeMessageLayer(hidden, config.bond_dim, config.dropout)
            for _ in range(config.num_layers)
        )
        self.slot_encoder = nn.Sequential(
            nn.Linear(hidden + 1, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.slot_embedding = nn.Parameter(torch.empty(NUM_DISTANCE_BINS, hidden))
        self.enzyme_query = nn.Linear(1024, hidden, bias=False)
        self.slot_key = nn.Linear(hidden, hidden, bias=False)
        self.learned_queries = nn.Parameter(torch.empty(NUM_DISTANCE_BINS, hidden))
        self.atom_key = nn.Linear(hidden, hidden, bias=False)
        self.atom_value = nn.Linear(hidden, hidden, bias=False)
        self.local_projection = nn.Sequential(
            nn.Linear(NUM_DISTANCE_BINS * hidden + 3, hidden), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(hidden, config.local_dim),
            nn.LayerNorm(config.local_dim),
        )
        nn.init.normal_(self.slot_embedding, std=0.02)
        nn.init.normal_(self.learned_queries, std=0.02)

    def _validate_global(self, batch: GraphBatch) -> None:
        if batch.protein.ndim != 2 or batch.protein.shape[0] < 1:
            raise ValueError("protein must have shape [B, 1024] with B >= 1")
        batch_size = batch.protein.shape[0]
        for name, width in (("protein", 1024), ("molt5", 768), ("maccs", 167)):
            value = getattr(batch, name)
            if value.shape != (batch_size, width) or not value.is_floating_point():
                raise ValueError(f"{name} must be a floating [{batch_size}, {width}] tensor")
            if value.device != batch.protein.device or value.dtype != batch.protein.dtype:
                raise ValueError("All global feature tensors must share device and dtype")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains nonfinite features")
        if self.training and batch_size < 2:
            raise ValueError(
                "CataPro BatchNorm requires at least two training samples per batch; "
                "merge a final singleton into the previous batch or use eval() for inference"
            )
        if batch.target is not None:
            if batch.target.shape != (batch_size,):
                raise ValueError("target must have shape [B], not [B, 1] (broadcasting risk)")
            if not bool(torch.isfinite(batch.target).all()):
                raise ValueError("target contains nonfinite values")
        if self.config.use_ec2:
            if batch.ec2.shape != (batch_size,) or batch.ec2.dtype != torch.int64:
                raise ValueError("ec2 must be int64 with shape [B]")
            if batch.ec2.device != batch.protein.device:
                raise ValueError("ec2 and global features must share a device")
            if bool(((batch.ec2 < 0) | (batch.ec2 >= self.config.ec_vocab_size)).any()):
                raise ValueError("EC index outside the training vocabulary; use index 0 for unknown")

    def _validate_graph(self, batch: GraphBatch) -> None:
        x, mask, edges = batch.atom_features, batch.atom_mask, batch.edge_index
        if x.ndim != 3 or x.shape[0] != batch.batch_size or x.shape[2] != self.config.atom_dim:
            raise ValueError("atom_features must have shape [B, N, atom_dim]")
        if x.shape[1] < 1 or mask.shape != x.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("atom_mask must be boolean [B, N] with N >= 1")
        if not bool(mask.any(dim=1).all()):
            raise ValueError("Every graph must contain a real atom")
        if not x.is_floating_point() or x.dtype != batch.protein.dtype:
            raise ValueError("Atom and global features must have the same floating dtype")
        if x.device != batch.protein.device or mask.device != x.device:
            raise ValueError("Graph and global features must share a device")
        if not bool(torch.isfinite(x[mask]).all()):
            raise ValueError("Real atom features must be finite")
        if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != torch.int64:
            raise ValueError("edge_index must be int64 [2, E]")
        if edges.device != x.device:
            raise ValueError("Edges and atom features must share a device")
        if batch.edge_features.shape != (edges.shape[1], self.config.bond_dim):
            raise ValueError("edge_features must have shape [E, bond_dim]")
        if batch.edge_features.dtype != x.dtype or batch.edge_features.device != x.device:
            raise ValueError("Bond and atom features must have the same dtype and device")
        if not bool(torch.isfinite(batch.edge_features).all()):
            raise ValueError("Bond features must be finite")
        if batch.distances.shape != (x.shape[0], x.shape[1], x.shape[1]):
            raise ValueError("distances must have shape [B, N, N]")
        if edges.shape[1]:
            total_nodes, width = x.shape[0] * x.shape[1], x.shape[1]
            if bool(((edges < 0) | (edges >= total_nodes)).any()):
                raise ValueError("edge_index outside the padded batch")
            if not bool(mask.reshape(-1)[edges].all()):
                raise ValueError("Edges must not touch padded atoms")
            if bool((edges[0].div(width, rounding_mode="floor") !=
                     edges[1].div(width, rounding_mode="floor")).any()):
                raise ValueError("Edges must not connect different graphs")
            if bool((edges[0] == edges[1]).any()):
                raise ValueError("Chemical graph edges must not contain self-loops")
            hashes = edges[0] * total_nodes + edges[1]
            if hashes.unique().numel() != hashes.numel():
                raise ValueError("Duplicate directed chemical edges are not allowed")
            reverse = edges[1] * total_nodes + edges[0]
            if not bool((hashes.sort().values == reverse.sort().values).all()):
                raise ValueError("Each undirected chemical bond needs both directed edges")
            forward_order = hashes.argsort()
            reverse_order = reverse.argsort()
            if not torch.equal(batch.edge_features[forward_order],
                               batch.edge_features[reverse_order]):
                raise ValueError("The two orientations of each chemical bond must share features")

    def _local_branch(self, batch: GraphBatch, protein: Tensor) -> tuple[Tensor, dict]:
        self._validate_graph(batch)
        cfg = self.config
        mask = batch.atom_mask
        batch_size, width = mask.shape
        positions = None
        if cfg.mode in CENTER_MODES:
            if batch.center_scores.dtype != batch.atom_features.dtype:
                raise ValueError("Center scores and atom features must share a floating dtype")
            positions = center_relative_profiles(
                batch.distances, batch.center_scores, mask, batch.center_available
            )
            active = positions.active
            score_statistics = positions.statistics
        else:
            # Information-matched controls really ignore scores and source flags.
            active = torch.ones(batch_size, dtype=torch.bool, device=mask.device)
            score_statistics = batch.atom_features.new_zeros((batch_size, 3))
        states = self.atom_encoder(batch.atom_features.masked_fill(~mask[:, :, None], 0))
        if cfg.mode in ("shell", "conditioned"):
            states = states + self.position_encoder(positions.profiles)
        states = states.masked_fill(~mask[:, :, None], 0)
        for layer in self.graph_layers:
            states = layer(states, batch.edge_index, batch.edge_features, mask)

        if cfg.mode in ("shell", "conditioned"):
            tokens, masses, slots = distance_shell_pool(states, positions.profiles, mask)
        elif cfg.mode == "learned":
            logits = torch.einsum("kh,bnh->bkn", self.learned_queries,
                                  self.atom_key(states)) / math.sqrt(cfg.hidden_dim)
            attention = _masked_softmax(logits, mask[:, None, :], dim=-1)
            tokens = attention.matmul(self.atom_value(states).masked_fill(~mask[:, :, None], 0))
            masses = states.new_ones((batch_size, NUM_DISTANCE_BINS))
            slots = torch.ones_like(masses, dtype=torch.bool)
        else:
            if cfg.mode == "center":
                weights = batch.center_scores.masked_fill(~mask | ~active[:, None], 0)
            else:
                weights = mask.to(states.dtype)
            mass = weights.sum(dim=1)
            pooled = (states * weights[:, :, None]).sum(dim=1) / mass.clamp_min(1e-8)[:, None]
            zeros = states.new_zeros((batch_size, NUM_DISTANCE_BINS - 1, cfg.hidden_dim))
            tokens = torch.cat((pooled[:, None, :], zeros), dim=1)
            masses = torch.cat((mass[:, None], states.new_zeros((batch_size, 4))), dim=1)
            slots = masses > 1e-8

        tokens = self.slot_encoder(torch.cat((tokens, torch.log1p(masses)[:, :, None]), dim=-1))
        tokens = (tokens + self.slot_embedding[None, :, :]).masked_fill(~slots[:, :, None], 0)
        if cfg.mode in ("conditioned", "learned"):
            query = self.enzyme_query(protein)
            logits = torch.einsum("bh,bkh->bk", query, self.slot_key(tokens))
            weights = _masked_softmax(logits / math.sqrt(cfg.hidden_dim), slots, dim=-1)
        else:
            weights = slots.to(states.dtype) / slots.sum(dim=1, keepdim=True).clamp_min(1)
        weighted = (tokens * weights[:, :, None]).reshape(batch_size, -1)
        local = self.local_projection(torch.cat((weighted, score_statistics), dim=-1))
        # Apply after all affine layers/LayerNorm: missing centers cannot leak a learned bias.
        local = local.masked_fill(~active[:, None], 0)
        diagnostics = {
            "local": local,
            "branch_active": active,
            "slot_weights": weights,
            "slot_mass": masses,
            "slot_mask": slots,
            "center_statistics": score_statistics,
        }
        if positions is not None:
            diagnostics["center_profiles"] = positions.profiles
        return local, diagnostics

    def forward(self, batch: GraphBatch, return_diagnostics: bool = False):
        self._validate_global(batch)
        protein = self.prot_norm(batch.protein)
        parts = [protein, self.molt5_norm(batch.molt5), batch.maccs]
        diagnostics = {}
        if self.config.mode != "baseline":
            local, diagnostics = self._local_branch(batch, protein)
            parts.append(local)
        if self.config.use_ec2:
            parts.append(self.ec_embedding(batch.ec2))
        prediction = self.out(self.decoder(torch.cat(parts, dim=1))).squeeze(-1)
        if prediction.shape != (batch.batch_size,):
            raise RuntimeError("Internal error: prediction must be a vector of length B")
        return (prediction, diagnostics) if return_diagnostics else prediction

    def load_catapro_weights(self, state_dict: Mapping[str, Tensor]) -> None:
        """Initialize shared layers from the official KcatModel state dictionary.

        New input columns are exactly zero, so predictions in eval mode equal
        the original model before optimization. This does not freeze the model.
        Pass raw weights, not an arbitrary pickled model/checkpoint object.
        Validation is completed before any parameters are modified.
        """
        if not isinstance(state_dict, Mapping) or not state_dict:
            raise ValueError("Expected a nonempty CataPro state dictionary")
        if not all(isinstance(key, str) for key in state_dict):
            raise ValueError("CataPro state dictionary keys must be strings")
        if all(key.startswith("module.") for key in state_dict):
            state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        destination = self.state_dict()
        prefixes = ("prot_norm.", "molt5_norm.", "decoder.", "out.")
        expected = {key for key in destination if key.startswith(prefixes)}
        missing, unexpected = expected - state_dict.keys(), state_dict.keys() - expected
        if missing or unexpected:
            raise ValueError(f"Not an official CataPro state dict: missing={sorted(missing)}, "
                             f"unexpected={sorted(unexpected)}")
        for key in expected:
            value = state_dict[key]
            if not isinstance(value, Tensor):
                raise ValueError(f"Checkpoint entry {key!r} is not a tensor")
            shape = (256, GLOBAL_FEATURE_DIM) if key == "decoder.0.weight" else destination[key].shape
            if tuple(value.shape) != tuple(shape):
                raise ValueError(f"CataPro checkpoint shape mismatch for {key}: "
                                 f"expected {tuple(shape)}, got {tuple(value.shape)}")
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"Nonfinite checkpoint values in {key}")
            if value.is_floating_point() != destination[key].is_floating_point():
                raise ValueError(f"Incompatible checkpoint tensor dtype for {key}")
            if not value.is_floating_point() and value.dtype != destination[key].dtype:
                raise ValueError(f"Incompatible checkpoint tensor dtype for {key}")
            if key.endswith(("running_var", "num_batches_tracked")) and bool((value < 0).any()):
                raise ValueError(f"BatchNorm checkpoint statistic {key} cannot be negative")
        with torch.no_grad():
            for key in expected:
                value = state_dict[key].to(destination[key])
                if key == "decoder.0.weight":
                    destination[key].zero_()
                    destination[key][:, :GLOBAL_FEATURE_DIM].copy_(value)
                else:
                    destination[key].copy_(value)
