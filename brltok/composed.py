"""E3 for any braille tokenizer: a token's vector is composed from its cells.

A cell vector is a linear map of its six dot bits (+1 raised, -1 flat) plus a learned per-cell
residual (zero at start) plus a position vector. A token averages its cell vectors, adds its
word-start and word-end flags, and a two-layer MLP maps the result to the model dimension.
Tokens without cells (pad, unk, the T1 space) get their own vectors. Two tokens whose cells
differ in one dot start out close, which is the point of E3 (robustness to dot errors).

fit() regresses the module onto a target table (the E2 vectors) before task training.
"""
from __future__ import annotations

import torch
from torch import nn

from .cells import mask
from .expansion import id_tokens, token_cells
from .tokenizers import MARK, BrailleBPE


def token_features(btok) -> tuple[list[str], list[tuple[bool, bool]]]:
    """Cells and (starts_word, ends_word) for every id of a T1/T2/T3 tokenizer."""
    toks = id_tokens(btok, list(range(btok.vocab_size)))
    cells, flags = [], []
    for t in toks:
        c = token_cells(t)
        ok = c.strip() != "" and all(0x2800 <= ord(ch) <= 0x283F for ch in c)
        cells.append(c if ok else "")
        flags.append((t.startswith(MARK), t.endswith(MARK) or t.endswith(BrailleBPE.EOW)))
    return cells, flags


class ComposedEmbedding(nn.Module):
    def __init__(self, cells: list[str], flags, dim: int, hidden: int = 512, n_dots: int = 6):
        super().__init__()
        V, L = len(cells), max(1, max(len(c) for c in cells))
        bits = torch.zeros(V, L, n_dots)
        cid = torch.zeros(V, L, dtype=torch.long)
        m = torch.zeros(V, L)
        for i, c in enumerate(cells):
            for j, ch in enumerate(c):
                b = mask(ch)
                bits[i, j] = torch.tensor([1.0 if b >> d & 1 else -1.0 for d in range(n_dots)])
                cid[i, j], m[i, j] = b, 1.0
        self.register_buffer("bits", bits)
        self.register_buffer("cid", cid)
        self.register_buffer("m", m)
        self.register_buffer("flags", torch.tensor(flags, dtype=torch.float32))
        self.register_buffer("special", m.sum(1) == 0)
        self.dot = nn.Linear(n_dots, hidden)
        self.cell_res = nn.Embedding(1 << n_dots, hidden)
        nn.init.zeros_(self.cell_res.weight)
        self.pos = nn.Embedding(L, hidden)
        nn.init.normal_(self.pos.weight, std=0.02)
        self.flag = nn.Linear(2, hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, dim))
        self.special_table = nn.Embedding(V, dim)
        nn.init.zeros_(self.special_table.weight)

    def table(self) -> torch.Tensor:
        L = self.bits.shape[1]
        x = self.dot(self.bits) + self.cell_res(self.cid) + self.pos.weight[:L]
        x = (x * self.m.unsqueeze(-1)).sum(1) / self.m.sum(1, keepdim=True).clamp(min=1)
        x = self.mlp(x + self.flag(self.flags))
        return torch.where(self.special.unsqueeze(-1), self.special_table.weight, x)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.table()[ids]

    def fit(self, target: torch.Tensor, rows: torch.Tensor, steps: int = 3000, lr: float = 1e-3,
            seed: int = 0) -> dict:
        """Regress onto target[rows] (MSE); special rows copy the target. Returns fit stats."""
        torch.manual_seed(seed)
        target = target.to(self.bits.device).float()
        with torch.no_grad():
            self.special_table.weight.copy_(target)
        opt = torch.optim.Adam([p for n, p in self.named_parameters() if not n.startswith("special_table")], lr=lr)
        rows = rows.to(self.bits.device) & ~self.special
        scale = target[rows].pow(2).mean()
        for _ in range(steps):
            loss = (self.table()[rows] - target[rows]).pow(2).mean() / scale
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            t = self.table()[rows]
            cos = nn.functional.cosine_similarity(t, target[rows], dim=-1).mean().item()
        return {"rel_mse": round(loss.item(), 4), "mean_cos": round(cos, 4), "rows": int(rows.sum())}
