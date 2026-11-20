"""A small non-autoregressive back-translation model (braille tokens -> print characters).

A Transformer encoder reads the braille tokens, and each token then emits a
fixed number of output frames that are trained with CTC against the print
string. Back-translation is monotone, so CTC fits, and the model trains and
decodes quickly enough to compare tokenizers on a CPU.
"""
from __future__ import annotations

import torch
from torch import nn


class BackTranslator(nn.Module):
    def __init__(self, embed: nn.Module, dim: int, n_out: int, frames: int,
                 n_layers: int = 4, n_heads: int = 4, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embed, self.frames, self.n_out = embed, frames, n_out
        self.pos = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(dim, n_heads, 4 * dim, dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, frames * n_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, ids: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """ids, pad: (B, L). Returns log-probabilities of shape (B, L * frames, n_out)."""
        B, L = ids.shape
        x = self.embed(ids) + self.pos(torch.arange(L, device=ids.device))[None]
        x = self.encoder(self.drop(x), src_key_padding_mask=pad)
        logits = self.head(self.norm(x)).view(B, L * self.frames, self.n_out)
        return logits.log_softmax(-1)


def greedy_decode(logp: torch.Tensor, lengths: torch.Tensor, chars: list[str]) -> list[str]:
    """Collapse repeats and drop blanks (id 0) from the best path."""
    best = logp.argmax(-1).tolist()
    out = []
    for row, n in zip(best, lengths.tolist()):
        prev, s = 0, []
        for i in row[:n]:
            if i != prev and i != 0:
                s.append(chars[i])
            prev = i
        out.append("".join(s))
    return out
