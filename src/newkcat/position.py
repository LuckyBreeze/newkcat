"""Permutation-equivariant reaction-center-relative graph positions.

The five channels are distances 0, 1, 2, >=3 and disconnected. They describe
distance to a center drawn from normalized input scores, NOT the probability
of distance to the nearest center. Scores are not calibrated confidence.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor

NUM_DISTANCE_BINS = 5
DISTANCE_BIN_NAMES = ("center", "one_hop", "two_hops", "far", "disconnected")


@dataclass
class CenterProfiles:
    profiles: Tensor       # [B, N, 5]
    active: Tensor         # [B]
    mass: Tensor           # [B], sum of the original (available) scores
    max_score: Tensor      # [B]
    mean_score: Tensor     # [B], excludes padded atoms

    @property
    def statistics(self) -> Tensor:
        """Absolute score summaries, separate from normalized positions."""
        return torch.stack((torch.log1p(self.mass), self.max_score, self.mean_score), dim=-1)


def center_relative_profiles(
    distances: Tensor,
    center_scores: Tensor,
    atom_mask: Tensor,
    center_available: Tensor | None = None,
    eps: float = 1e-8,
) -> CenterProfiles:
    """Compute soft center-relative positions without choosing an arbitrary root.

    Invalid values are errors, not silently repaired annotations. Missing
    centers and zero score mass explicitly produce zero profiles. Padding is
    excluded from both source centers and destination atoms. A real atom in
    another connected component contributes to the disconnected channel.
    """
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if atom_mask.ndim != 2 or atom_mask.dtype != torch.bool:
        raise ValueError("atom_mask must be a boolean [B, N] tensor")
    batch_size, width = atom_mask.shape
    if batch_size < 1 or width < 1 or not bool(atom_mask.any(dim=1).all()):
        raise ValueError("Every graph must contain at least one real atom")
    if center_scores.shape != (batch_size, width) or not center_scores.is_floating_point():
        raise ValueError("center_scores must be a floating [B, N] tensor")
    if distances.shape != (batch_size, width, width):
        raise ValueError("distances must have shape [B, N, N]")
    if distances.dtype not in (torch.int32, torch.int64):
        raise ValueError("distances must contain integer graph-hop counts")
    if not (distances.device == center_scores.device == atom_mask.device):
        raise ValueError("Position inputs must be on the same device")
    if not bool(torch.isfinite(center_scores).all()):
        raise ValueError("center_scores must be finite")
    if bool(((center_scores < 0) | (center_scores > 1)).any()):
        raise ValueError("center_scores must lie in [0, 1]")
    if bool((center_scores.masked_select(~atom_mask) != 0).any()):
        raise ValueError("Padded atoms must have zero center scores")
    if bool((distances < -1).any()):
        raise ValueError("Only -1 is allowed for disconnected/padded distances")
    if not bool((distances == distances.transpose(1, 2)).all()):
        raise ValueError("Undirected molecular graph distances must be symmetric")
    if bool((distances.diagonal(dim1=1, dim2=2).masked_select(atom_mask) != 0).any()):
        raise ValueError("The distance of a real atom to itself must be zero")
    pair_mask = atom_mask.unsqueeze(2) & atom_mask.unsqueeze(1)
    off_diagonal = ~torch.eye(width, dtype=torch.bool, device=distances.device)
    if bool(((distances == 0) & pair_mask & off_diagonal).any()):
        raise ValueError("Distinct atoms cannot have zero graph distance")
    if bool((distances.masked_select(~pair_mask) != -1).any()):
        raise ValueError("Distances involving padded atoms must be -1")
    if center_available is None:
        center_available = torch.ones(batch_size, dtype=torch.bool, device=atom_mask.device)
    if center_available.shape != (batch_size,) or center_available.dtype != torch.bool:
        raise ValueError("center_available must be boolean with shape [B]")
    if center_available.device != atom_mask.device:
        raise ValueError("center_available must be on the same device as the graph")

    scores = center_scores.masked_fill(~atom_mask | ~center_available[:, None], 0)
    mass = scores.sum(dim=1)
    active = center_available & (mass > eps)
    alpha = scores / mass.clamp_min(eps)[:, None]
    bins = distances.clamp(min=0, max=3)
    bins = torch.where(distances == -1, 4, bins)
    profiles = torch.stack(
        [
            ((bins == index) & pair_mask).to(alpha.dtype).matmul(alpha.unsqueeze(-1)).squeeze(-1)
            for index in range(NUM_DISTANCE_BINS)
        ],
        dim=-1,
    )
    profiles = profiles.masked_fill(~atom_mask[:, :, None] | ~active[:, None, None], 0)
    mass = mass.masked_fill(~active, 0)
    max_score = scores.max(dim=1).values.masked_fill(~active, 0)
    mean_score = (mass / atom_mask.sum(dim=1)).masked_fill(~active, 0)
    return CenterProfiles(profiles, active, mass, max_score, mean_score)


def distance_shell_pool(
    atom_states: Tensor, profiles: Tensor, atom_mask: Tensor, eps: float = 1e-8
) -> tuple[Tensor, Tensor, Tensor]:
    """Return five weighted means, their effective masses and valid-slot masks.

    The masses are retained so a normalized mean cannot silently discard all
    information about region size. Soft memberships may overlap between slots.
    """
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if atom_states.ndim != 3:
        raise ValueError("atom_states must have shape [B, N, H]")
    if profiles.shape != (*atom_states.shape[:2], NUM_DISTANCE_BINS):
        raise ValueError("profiles must have shape [B, N, 5]")
    if atom_mask.shape != atom_states.shape[:2] or atom_mask.dtype != torch.bool:
        raise ValueError("atom_mask must be boolean with shape [B, N]")
    if not atom_states.is_floating_point() or profiles.dtype != atom_states.dtype:
        raise ValueError("Profiles and atom states must share a floating dtype")
    if not (atom_states.device == profiles.device == atom_mask.device):
        raise ValueError("Shell inputs must share a device")
    if not bool(torch.isfinite(atom_states[atom_mask]).all()):
        raise ValueError("Real atom states must be finite")
    if not bool(torch.isfinite(profiles).all()) or bool((profiles < 0).any()):
        raise ValueError("profiles must be finite nonnegative memberships")
    states = atom_states.masked_fill(~atom_mask[:, :, None], 0)
    memberships = profiles.masked_fill(~atom_mask[:, :, None], 0)
    masses = memberships.sum(dim=1)
    valid_slots = masses > eps
    pooled = memberships.transpose(1, 2).matmul(states) / masses.clamp_min(eps)[:, :, None]
    pooled = pooled.masked_fill(~valid_slots[:, :, None], 0)
    return pooled, masses, valid_slots
