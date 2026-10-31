"""Shared typed tensor contract; indices refer to padded graph batches."""

from dataclasses import dataclass, fields, replace

import torch
from torch import Tensor


@dataclass
class GraphBatch:
    """A batch of independently padded molecular graphs and cached global features.

    ``edge_index`` indexes ``atom_features.reshape(B * N, F)``; N is the
    *batch padded width*, not the size of each individual molecule. Distances
    are exact graph hop counts, with -1 for disconnected or padded pairs.
    Map numbers/sample identifiers are never numerical model features.
    """

    atom_features: Tensor       # [B, N, atom_dim]
    edge_index: Tensor          # [2, E], directed edges, offset by graph_idx * N
    edge_features: Tensor       # [E, bond_dim]
    distances: Tensor           # [B, N, N], long; -1 means disconnected/padding
    atom_mask: Tensor           # [B, N], bool
    center_scores: Tensor       # [B, N], finite in [0, 1], zero on padding
    center_available: Tensor    # [B], bool, false for missing/ambiguous annotations
    protein: Tensor            # [B, 1024]
    molt5: Tensor               # [B, 768]
    maccs: Tensor               # [B, 167]
    ec2: Tensor                # [B], long; 0=unknown; vocab fit on train only
    target: Tensor | None = None  # [B], log10(kcat / s^-1)
    sample_ids: tuple[str, ...] = ()
    center_kinds: tuple[str, ...] = ()  # predicted/reference/missing, audit only

    def to(self, device: str | torch.device) -> "GraphBatch":
        updates = {
            field.name: getattr(self, field.name).to(device)
            for field in fields(self)
            if isinstance(getattr(self, field.name), Tensor)
        }
        return replace(self, **updates)

    @property
    def batch_size(self) -> int:
        return int(self.protein.shape[0])
