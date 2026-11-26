"""Braille prior P(c): a causal transformer language model over braille cells
(docs/design_noisy_channel.md). Symbols: the 64 six-dot patterns, 64 = blank (word break),
65 = anything else, 66 = sentence boundary (BOS and EOS)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cells import is_cell, mask

SPACE, UNK, BOUND, N_SYM = 64, 65, 66, 67


def cell_seq(braille: str) -> list[int]:
    out = []
    for ch in braille:
        if ch == " ":
            out.append(SPACE)
        elif is_cell(ch) and mask(ch) < 64:
            out.append(SPACE if mask(ch) == 0 else mask(ch))
        else:
            out.append(UNK)
    return out


class CellLM(nn.Module):
    def __init__(self, d: int = 512, layers: int = 6, heads: int = 8, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(N_SYM, d)
        self.pos = nn.Embedding(max_len, d)
        el = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(el, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, N_SYM)

    @property
    def device(self):
        return self.emb.weight.device

    def _batch(self, texts: list[str]):
        seqs = [[BOUND] + cell_seq(t)[: self.max_len - 2] + [BOUND] for t in texts]
        L = max(len(s) for s in seqs)
        X = torch.full((len(seqs), L), BOUND, dtype=torch.long)
        lab = torch.full((len(seqs), L), -100, dtype=torch.long)
        for i, s in enumerate(seqs):
            X[i, : len(s)] = torch.tensor(s)
            lab[i, : len(s) - 1] = torch.tensor(s[1:])
        return X.to(self.device), lab.to(self.device)

    def forward(self, X):
        L = X.shape[1]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=X.device), 1)
        h = self.emb(X) + self.pos(torch.arange(L, device=X.device))[None]
        return self.head(self.norm(self.tr(h, mask=causal, is_causal=True)))

    def loss(self, texts: list[str]):
        X, lab = self._batch(texts)
        lg = self(X).float()
        return F.cross_entropy(lg.reshape(-1, N_SYM), lab.reshape(-1), ignore_index=-100)

    @torch.no_grad()
    def score(self, texts: list[str], batch: int = 256) -> list[float]:
        """Natural-log probability of each braille string, end boundary included."""
        out = []
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        res = [0.0] * len(texts)
        for j in range(0, len(order), batch):
            idx = order[j: j + batch]
            X, lab = self._batch([texts[i] for i in idx])
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=X.is_cuda):
                lg = self(X)
            lp = F.log_softmax(lg.float(), -1)
            s = (lp.gather(-1, lab.clamp(min=0)[..., None])[..., 0] * (lab >= 0)).sum(-1).tolist()
            for i, v in zip(idx, s):
                res[i] = v
        out = res
        return out


def load(path: str, device="cuda") -> CellLM:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = CellLM(**ck["config"])
    m.load_state_dict(ck["model"])
    return m.to(device).eval()
