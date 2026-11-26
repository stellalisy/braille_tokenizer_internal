"""Per-word symbol-to-symbol transformers for the noisy-channel reader (docs/design_noisy_channel.md).

One class serves three tasks:
  reader      braille cells of one word -> print characters (the proposal reader, P(y | b))
  translator  print characters of one word -> braille cells (P(c | y))
  wordprior   a constant input -> print characters (P_words(y), the print-word prior the reader
              learns from its training mix)
The encoder reads the input symbols of one word (cells get their six dot bits as extra features); the
decoder emits output symbols, then END. `nbest` is beam search per word; `score` is the teacher-forced
log probability of given outputs, END included.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cells import is_cell, mask

CELL_UNK = 65
N_CELL_IN = 66      # 64 patterns, 64 = blank (unused inside a word), 65 = not a cell


def cell_input(word: str) -> list[int]:
    return [mask(ch) & 63 if is_cell(ch) and mask(ch) < 64 else CELL_UNK for ch in word]


def cells_to_str(ids) -> str:
    return "".join(chr(0x2800 + i) for i in ids)


class CharVocab:
    def __init__(self, chars):
        self.chars = list(chars)
        self.idx = {c: i for i, c in enumerate(self.chars)}
        self.unk = len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.idx.get(c, self.unk) for c in s]

    def decode(self, ids) -> str:
        return "".join(self.chars[i] if i < len(self.chars) else "�" for i in ids)


class WordSeq2Seq(nn.Module):
    def __init__(self, n_in: int, n_out: int, d: int = 512, enc_layers: int = 4, dec_layers: int = 2,
                 heads: int = 8, max_in: int = 64, max_out: int = 40, dropout: float = 0.1,
                 cell_bits: bool = False):
        super().__init__()
        self.n_in, self.n_out, self.max_in, self.max_out = n_in, n_out, max_in, max_out
        self.cell_bits = cell_bits
        self.inp = nn.Embedding(n_in + 1, d)             # n_in = PAD
        self.bits = nn.Linear(6, d, bias=False) if cell_bits else None
        self.pos = nn.Embedding(max_in, d)
        el = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(el, enc_layers, enable_nested_tensor=False)
        self.enc_norm = nn.LayerNorm(d)
        self.out_in = nn.Embedding(n_out + 1, d)          # n_out = BOS
        self.tpos = nn.Embedding(max_out + 2, d)
        dl = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(dl, dec_layers)
        self.dec_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_out + 1)               # n_out = END

    @property
    def device(self):
        return self.inp.weight.device

    def encode(self, inputs: list[list[int]]):
        dev = self.device
        L = max(1, max(min(len(x), self.max_in) for x in inputs))
        X = torch.full((len(inputs), L), self.n_in, dtype=torch.long, device=dev)
        pad = torch.ones((len(inputs), L), dtype=torch.bool, device=dev)
        for i, x in enumerate(inputs):
            x = x[: self.max_in]
            if x:
                X[i, : len(x)] = torch.tensor(x, device=dev)
                pad[i, : len(x)] = False
        pad[:, 0] = False          # an empty input attends to one PAD slot
        h = self.inp(X) + self.pos(torch.arange(L, device=dev))[None]
        if self.bits is not None:
            bits = ((X.clamp(max=63).unsqueeze(-1) >> torch.arange(6, device=dev)) & 1).float()
            h = h + self.bits(bits * (X < 64).unsqueeze(-1))
        h = self.enc(h, src_key_padding_mask=pad)
        return self.enc_norm(h), pad

    def logits(self, mem, mpad, prev):
        """prev [W, K] output ids -> scores [W, K + 1, n_out + 1] (END last)."""
        W, K = prev.shape
        bos = torch.full((W, 1), self.n_out, dtype=torch.long, device=prev.device)
        x = self.out_in(torch.cat([bos, prev], 1)) + self.tpos(torch.arange(K + 1, device=prev.device))[None]
        causal = torch.triu(torch.ones(K + 1, K + 1, dtype=torch.bool, device=x.device), 1)
        h = self.dec(x, mem, tgt_mask=causal, memory_key_padding_mask=mpad)
        return self.head(self.dec_norm(h))

    def _targets(self, targets):
        dev = self.device
        K = max(1, max(min(len(t), self.max_out) for t in targets))
        prev = torch.zeros((len(targets), K), dtype=torch.long, device=dev)
        lab = torch.full((len(targets), K + 1), -100, dtype=torch.long, device=dev)
        for i, t in enumerate(targets):
            t = t[: self.max_out]
            if t:
                prev[i, : len(t)] = torch.tensor(t, device=dev)
                lab[i, : len(t)] = torch.tensor(t, device=dev)
            lab[i, len(t)] = self.n_out
        return prev, lab

    def loss(self, inputs, targets, smoothing: float = 0.05):
        mem, mpad = self.encode(inputs)
        prev, lab = self._targets(targets)
        lg = self.logits(mem, mpad, prev).float()
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), lab.reshape(-1), ignore_index=-100,
                             label_smoothing=smoothing)
        with torch.no_grad():
            acc = ((lg.argmax(-1) == lab) | (lab == -100)).all(1).float().mean()
        return ce, acc

    @torch.no_grad()
    def score(self, inputs, targets, batch: int = 4096) -> list[float]:
        """log P(target + END | input) per pair."""
        out = []
        for j in range(0, len(inputs), batch):
            mem, mpad = self.encode(inputs[j: j + batch])
            prev, lab = self._targets(targets[j: j + batch])
            lp = F.log_softmax(self.logits(mem, mpad, prev).float(), -1)
            tl = lab.clamp(min=0)
            out.extend((lp.gather(-1, tl[..., None])[..., 0] * (lab >= 0)).sum(-1).tolist())
        return out

    @torch.no_grad()
    def nbest(self, inputs, k: int = 16, batch: int = 512):
        """Beam search. Per input: up to k (output ids, log prob incl. END), best first."""
        res = []
        for j in range(0, len(inputs), batch):
            res.extend(self._nbest(inputs[j: j + batch], k))
        return res

    def _nbest(self, inputs, k):
        mem, mpad = self.encode(inputs)
        W, dev, END = len(inputs), self.device, self.n_out
        memk, mpadk = mem.repeat_interleave(k, 0), mpad.repeat_interleave(k, 0)
        prev = torch.zeros((W * k, 0), dtype=torch.long, device=dev)
        score = torch.full((W, k), float("-inf"), device=dev)
        score[:, 0] = 0.0
        fin = [[] for _ in range(W)]
        for step in range(self.max_out + 1):
            lp = F.log_softmax(self.logits(memk, mpadk, prev)[:, -1].float(), -1).view(W, k, -1)
            if step > 0:
                ends = (score + lp[:, :, END]).cpu()
                pv = prev.view(W, k, -1).cpu()
                for w, b in torch.nonzero(torch.isfinite(ends)).tolist():
                    fin[w].append((pv[w, b].tolist(), float(ends[w, b])))
            if step == self.max_out:
                break
            cand = (score[:, :, None] + lp[:, :, :END]).reshape(W, -1)
            score, idx = cand.topk(k, -1)
            beam, tok = idx // END, idx % END
            kth = torch.tensor([sorted((x[1] for x in f), reverse=True)[k - 1] if len(f) >= k else float("-inf")
                                for f in fin], device=dev)
            score = torch.where(score > kth[:, None], score, torch.full_like(score, float("-inf")))
            if not torch.isfinite(score).any():
                break
            rows = (torch.arange(W, device=dev)[:, None] * k + beam).view(-1)
            prev = torch.cat([prev[rows], tok.reshape(-1, 1)], 1)
        return [sorted(f, key=lambda x: -x[1])[:k] for f in fin]


def save(path: str, model: WordSeq2Seq, config: dict, meta: dict):
    torch.save({"model": model.state_dict(), "config": config, "meta": meta}, path)


def load(path: str, device="cuda"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = WordSeq2Seq(**ck["config"])
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck["meta"]
