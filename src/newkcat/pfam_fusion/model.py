"""Four Pfam × CataPro fusion variants; original newkcat modules remain untouched."""

from __future__ import annotations

import torch
from torch import nn

from newkcat.model import GLOBAL_FEATURE_DIM, ModelConfig, NewKcat

PFAM_OUT_DIM = 256


class PfamEncoder(nn.Module):
    """100-D random or DeepBGC lookup -> projection -> Transformer -> masked flatten.

    For ordered=True, tokens are arranged by start coordinates and receive
    learned positional embeddings. For ordered=False, preprocessing deduplicates
    and alphabetically sorts IDs; positional embeddings are omitted.
    """

    def __init__(self, vocab_size: int, nmax: int, ordered: bool,
                 pretrained: torch.Tensor | None = None, dropout: float = 0.1,
                 layers: int = 1, heads: int = 4):
        super().__init__()
        if vocab_size < 3 or nmax < 1 or 256 % heads or layers < 1:
            raise ValueError("Invalid vocab size, Nmax, number of heads or layers")
        self.vocab_size = vocab_size
        self.nmax = nmax
        self.ordered = ordered
        self.embedding = nn.Embedding(vocab_size, 100, padding_idx=0)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embedding.weight[0].zero_()
            if pretrained is not None:
                if pretrained.shape != (vocab_size, 100):
                    raise ValueError("pretrained lookup must have shape [vocab_size,100]")
                if not torch.isfinite(pretrained).all():
                    raise ValueError("pretrained lookup contains nonfinite values")
                self.embedding.weight[2:].copy_(pretrained[2:])
        self.projection = nn.Linear(100, PFAM_OUT_DIM)
        self.position = nn.Embedding(nmax, PFAM_OUT_DIM) if ordered else None
        if self.position is not None:
            nn.init.normal_(self.position.weight, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=PFAM_OUT_DIM, nhead=heads, dim_feedforward=512,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers,
                                             enable_nested_tensor=False)
        self.out = nn.Sequential(nn.Linear(nmax * PFAM_OUT_DIM, PFAM_OUT_DIM),
                                 nn.LayerNorm(PFAM_OUT_DIM), nn.ReLU())

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim != 2 or indices.shape[1] != self.nmax:
            raise ValueError(f"Pfam tokens must be [B,{self.nmax}]")
        if indices.dtype != torch.long:
            raise ValueError("Pfam tokens must have dtype torch.long")
        if bool(((indices < 0) | (indices >= self.vocab_size)).any()):
            raise ValueError("Pfam token index out of vocabulary")
        mask = indices != 0
        any_valid = mask.any(dim=1)
        # Never use an untrained random UNK vector at inference.
        embedded = self.embedding(indices)
        unknown = indices == 1
        if bool(unknown.any()):
            mean_embedding = self.embedding.weight[2:].mean(dim=0)
            embedded = torch.where(unknown.unsqueeze(-1), mean_embedding, embedded)
        x = self.projection(embedded)
        if self.position is not None:
            x = x + self.position(torch.arange(self.nmax, device=indices.device))[None]
        x = x.masked_fill(~mask.unsqueeze(-1), 0)
        # PyTorch attention cannot softmax an all-masked row.
        safe_mask = mask.clone()
        safe_mask[~any_valid, 0] = True
        hidden = self.encoder(x, src_key_padding_mask=~safe_mask)
        # Padded query states must be zeroed explicitly after attention.
        hidden = hidden.masked_fill(~mask.unsqueeze(-1), 0)
        output = self.out(hidden.flatten(start_dim=1))
        return output * any_valid.unsqueeze(-1).to(output.dtype)


class PfamCataPro(nn.Module):
    """Exact existing ProtT5 BN, MolT5 BN, MACCS and decoder; +256 Pfam columns."""

    def __init__(self, vocab_size: int, nmax: int, ordered: bool,
                 pretrained: torch.Tensor | None = None, dropout: float = 0.1,
                 layers: int = 1):
        super().__init__()
        self.pfam = PfamEncoder(vocab_size, nmax, ordered, pretrained=pretrained,
                                dropout=dropout, layers=layers)
        self.catapro = NewKcat(ModelConfig(atom_dim=1, bond_dim=1,
                                            mode="baseline", dropout=0.0))
        # Existing modules are unmodified: only this model's fusion width changes.
        self.catapro.decoder[0] = nn.Linear(GLOBAL_FEATURE_DIM + PFAM_OUT_DIM, 256)

    def initialize_catapro(self, original_state: dict) -> None:
        """Copy official CataPro parameters; new input columns start at zero."""
        self.catapro.load_catapro_weights(original_state)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        protein, molt5, maccs, pfam = (
            batch["protein"], batch["molt5"], batch["maccs"], batch["pfam_ids"]
        )
        b = protein.shape[0]
        if (protein.shape != (b, 1024) or molt5.shape != (b, 768)
                or maccs.shape != (b, 167) or pfam.shape[0] != b):
            raise ValueError("Expected ProtT5[B,1024], MolT5[B,768], MACCS[B,167]")
        if self.training and b < 2:
            raise ValueError("CataPro BatchNorm requires batch_size >= 2 during training")
        p = self.catapro.prot_norm(protein)
        m = self.catapro.molt5_norm(molt5)
        f = self.pfam(pfam)
        z = torch.cat((p, m, maccs, f), dim=-1)
        y = self.catapro.out(self.catapro.decoder(z)).squeeze(-1)
        if y.shape != (b,):
            raise RuntimeError("Unexpected prediction shape")
        return y
