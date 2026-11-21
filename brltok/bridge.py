"""Braille input path into a frozen pretrained causal LM.

Braille goes through its own tokenizer and embedding module and enters the base model as
`inputs_embeds`, between the print embeddings of the surrounding prompt. The base model's
tokenizer, embeddings, body and output head are untouched; `param_hash` proves it.

E1  mean init: every braille token starts at the mean input embedding, plus small noise
    scaled by the embedding standard deviation (Hewitt, 2021).
E2  print-aligned init: the count-weighted mean, over a token's print expansions, of the mean
    base embedding of each tokenized expansion, with a leading space when the braille token
    starts a word. Tokens without a non-empty expansion fall back to E1.
"""
from __future__ import annotations

import hashlib

import torch
from torch import nn


def param_hash(model: nn.Module) -> str:
    """SHA-256 over every parameter's name and bytes, in a fixed order."""
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters(), key=lambda x: x[0]):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def freeze(model: nn.Module) -> nn.Module:
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


@torch.no_grad()
def mean_init(base_emb: torch.Tensor, vocab_size: int, seed: int = 0, noise: float = 0.02) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    e = base_emb.float()
    mu, sd = e.mean(0), e.std(0)
    return mu + noise * sd * torch.randn(vocab_size, e.shape[1], generator=g)


@torch.no_grad()
def print_aligned_init(counts, base_tokenizer, base_emb: torch.Tensor, vocab_size: int,
                       seed: int = 0) -> tuple[torch.Tensor, int]:
    """E2 table. counts[id] is a Counter of (print text, starts_word). Returns (table, n_fallback)."""
    table = mean_init(base_emb, vocab_size, seed)
    e = base_emb.float()
    cache, fallback = {}, 0
    for i in range(vocab_size):
        acc, tot = None, 0
        for (text, sw), c in counts.get(i, {}).items():
            if not text.strip():
                continue
            s = (" " + text) if sw and not text.startswith(" ") else text
            if s not in cache:
                ids = base_tokenizer(s, add_special_tokens=False)["input_ids"]
                cache[s] = e[ids].mean(0) if ids else None
            v = cache[s]
            if v is None:
                continue
            acc = v * c if acc is None else acc + v * c
            tot += c
        if tot:
            table[i] = acc / tot
        else:
            fallback += 1
    return table, fallback


class BrailleInput(nn.Module):
    """Wraps a braille embedding module and splices its output into a base model's input."""

    def __init__(self, embed: nn.Module):
        super().__init__()
        self.embed = embed

    def splice(self, base_model, base_tokenizer, prefix: str, braille_ids: list[int], suffix: str,
               device=None) -> tuple[torch.Tensor, int, int]:
        """inputs_embeds [1, T, d] for prefix + braille + suffix, and the braille span (start, end)."""
        emb = base_model.get_input_embeddings()
        device = device or emb.weight.device
        pre = base_tokenizer(prefix, add_special_tokens=False)["input_ids"]
        suf = base_tokenizer(suffix, add_special_tokens=False)["input_ids"]
        pe = emb(torch.tensor([pre], device=device)) if pre else None
        se = emb(torch.tensor([suf], device=device)) if suf else None
        be = self.embed(torch.tensor([braille_ids], device=device)).to(emb.weight.dtype)
        parts = [x for x in (pe, be, se) if x is not None]
        return torch.cat(parts, dim=1), len(pre), len(pre) + len(braille_ids)
