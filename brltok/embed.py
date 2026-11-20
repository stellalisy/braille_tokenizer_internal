"""Input embeddings for braille tokens (PyTorch).

E1  TableEmbedding    one free vector per token
E3  DotEmbedding      a cell vector is a linear map of its dot bits, plus an
                      optional learned residual per cell
"""
from __future__ import annotations

import torch
from torch import nn


class TableEmbedding(nn.Embedding):
    def __init__(self, vocab_size: int, dim: int, pad_id: int = 0):
        super().__init__(vocab_size, dim, padding_idx=pad_id)
        nn.init.normal_(self.weight, std=dim ** -0.5)
        with torch.no_grad():
            self.weight[pad_id].zero_()


class DotEmbedding(nn.Module):
    """Embedding for a cell tokenizer that shares parameters through the dots.

    masks[i] is the dot bitmask of token id i, or -1 for a special token.
    Dots are coded as +1 (raised) and -1 (flat), so two cells that differ in
    one dot start out as close neighbors. With residual=True a zero-initialized
    per-cell vector is added, which keeps the layer as expressive as a table.
    """

    def __init__(self, masks, dim: int, n_dots: int = 6, residual: bool = True, mlp: bool = False):
        super().__init__()
        bits = torch.zeros(len(masks), n_dots)
        special = torch.zeros(len(masks), dtype=torch.bool)
        for i, m in enumerate(masks):
            if m < 0:
                special[i] = True
            else:
                bits[i] = torch.tensor([1.0 if m >> d & 1 else -1.0 for d in range(n_dots)])
        self.register_buffer("bits", bits)
        self.register_buffer("special", special)
        if mlp:
            self.proj = nn.Sequential(nn.Linear(n_dots, dim), nn.GELU(), nn.Linear(dim, dim))
        else:
            self.proj = nn.Linear(n_dots, dim)
        self.special_table = nn.Embedding(len(masks), dim)
        nn.init.zeros_(self.special_table.weight)
        self.residual = nn.Embedding(len(masks), dim) if residual else None
        if residual:
            nn.init.zeros_(self.residual.weight)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.proj(self.bits[ids])
        x = torch.where(self.special[ids].unsqueeze(-1), self.special_table(ids), x)
        if self.residual is not None:
            x = x + self.residual(ids)
        return x
