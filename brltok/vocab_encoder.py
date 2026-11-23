"""Braille read into the model's own print vocabulary (docs/design_print_vocab_encoder.md).

A cell-level transformer encodes a braille span. For each braille word (a run of non-blank cells),
a small decoder attends to that word's encoder states and emits the word's print tokens, as the base
model's tokenizer splits " " + word, then an end-of-word symbol. Output scores are dot products with
the base model's frozen input embeddings of the allowed print tokens, so predictions live in the
model's vocabulary. The braille input to the model is, per predicted token,

    x = sum_v p(v) E[v]

a convex mix of real token embeddings (soft), or E[argmax] (hard).

With local=True (design note revision 2) the reader is word-local: encoder attention is masked so a
cell attends only to cells of its own word, and positions count from the start of each word. The
reading of a word then depends on its cells alone, never on the sentence around it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cells import is_cell, mask, normalize

SPACE, UNK = 64, 65
N_IN = 66


def span_words(braille: str) -> list[tuple[int, int]]:
    """(start, end) character spans of the words in a normalized braille string (blank = ' ')."""
    return [m.span() for m in re.finditer(r"[^ ]+", braille)]


def cell_ids(braille: str) -> list[int]:
    out = []
    for ch in braille:
        if ch == " ":
            out.append(SPACE)
        elif is_cell(ch):
            out.append(mask(ch))
        else:
            out.append(UNK)
    return out


def word_targets(base_tok, text: str) -> list[list[int]]:
    """Print token ids of each whitespace-separated word, each with its leading space."""
    return [base_tok(" " + w, add_special_tokens=False)["input_ids"] for w in text.split()]


class VocabEncoder(nn.Module):
    def __init__(self, allowed_ids: torch.Tensor, emb: torch.Tensor, d: int = 512, enc_layers: int = 4,
                 dec_layers: int = 2, heads: int = 8, max_len: int = 4096, max_tokens: int = 24,
                 dropout: float = 0.1, local: bool = False):
        super().__init__()
        V, D = emb.shape
        self.local = local
        self.heads = heads
        self.max_tokens = max_tokens
        self.register_buffer("allowed", allowed_ids.long().clone())
        to_idx = torch.full((V,), -1, dtype=torch.long)
        to_idx[self.allowed] = torch.arange(len(self.allowed))
        self.register_buffer("to_idx", to_idx)
        self.register_buffer("E", emb[self.allowed].detach().float().clone(), persistent=False)
        self.cell = nn.Embedding(N_IN, d)
        self.dots = nn.Linear(6, d, bias=False)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, enc_layers, enable_nested_tensor=False)
        self.enc_norm = nn.LayerNorm(d)
        self.tok_in = nn.Linear(D, d, bias=False)
        self.bos = nn.Parameter(torch.randn(d) * 0.02)
        self.tpos = nn.Embedding(max_tokens + 1, d)
        dl = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(dl, dec_layers)
        self.dec_norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, D, bias=False)
        self.end = nn.Linear(d, 1)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def set_embeddings(self, emb: torch.Tensor):
        """Re-attach the frozen base embeddings (not saved in the state dict)."""
        self.E = emb[self.allowed].detach().float().to(self.allowed.device)

    # ---- encoder -------------------------------------------------------------------------------
    def encode(self, spans: list[str]):
        dev = self.allowed.device
        ids = [cell_ids(s) for s in spans]
        L = max(1, max(len(x) for x in ids))
        X = torch.full((len(ids), L), SPACE, dtype=torch.long, device=dev)
        pad = torch.ones((len(ids), L), dtype=torch.bool, device=dev)
        for i, x in enumerate(ids):
            if x:
                X[i, : len(x)] = torch.tensor(x, device=dev)
                pad[i, : len(x)] = False
        bits = ((X.clamp(max=63).unsqueeze(-1) >> torch.arange(6, device=dev)) & 1).float()
        bits = bits * (X < 64).unsqueeze(-1)
        if not self.local:
            h = self.cell(X) + self.dots(bits) + self.pos(torch.arange(L, device=dev))[None]
            h = self.enc(h, src_key_padding_mask=pad)
            return self.enc_norm(h)
        # word-local: group id per position (each blank or pad cell is its own group), offset in word
        grp = torch.zeros((len(ids), L), dtype=torch.long, device=dev)
        off = torch.zeros((len(ids), L), dtype=torch.long, device=dev)
        for i, x in enumerate(ids):
            g, o, prev_blank = 0, 0, True
            gl, ol = [], []
            for c in x + [SPACE] * (L - len(x)):
                if c == SPACE:
                    g += 1
                    gl.append(g)
                    ol.append(0)
                    g += 1
                    prev_blank = True
                else:
                    if prev_blank:
                        o = 0
                    gl.append(g)
                    ol.append(o)
                    o += 1
                    prev_blank = False
            grp[i] = torch.tensor(gl, device=dev)
            off[i] = torch.tensor(ol, device=dev)
        h = self.cell(X) + self.dots(bits) + self.pos(off)
        block = grp[:, :, None] != grp[:, None, :]                      # True = may not attend
        block = block.repeat_interleave(self.heads, 0)
        h = self.enc(h, mask=block)
        return self.enc_norm(h)

    def word_memory(self, H, spans: list[str]):
        """Encoder states of every word, padded: memory [W, Lw, d], key padding mask [W, Lw]."""
        words = [(i, s, e) for i, sp in enumerate(spans) for s, e in span_words(sp)]
        Lw = max(1, max((e - s for _, s, e in words), default=1))
        mem = H.new_zeros((len(words), Lw, H.shape[-1]))
        mpad = torch.ones((len(words), Lw), dtype=torch.bool, device=H.device)
        for k, (i, s, e) in enumerate(words):
            mem[k, : e - s] = H[i, s:e]
            mpad[k, : e - s] = False
        return mem, mpad, words

    # ---- decoder -------------------------------------------------------------------------------
    def _dec_inputs(self, prev: torch.Tensor):
        """prev [W, K] allowed-vocab indices (-1 = pad) -> decoder inputs [W, K+1, d] (BOS first)."""
        W, K = prev.shape
        emb = self.tok_in(self.E[prev.clamp(min=0)])
        x = torch.cat([self.bos.expand(W, 1, -1), emb], 1)
        return x + self.tpos(torch.arange(K + 1, device=prev.device))[None]

    def logits(self, mem, mpad, prev):
        """Scores over allowed tokens plus END (last column), [W, K+1, V'+1]."""
        x = self._dec_inputs(prev)
        T = x.shape[1]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        h = self.dec(x, mem, tgt_mask=causal, memory_key_padding_mask=mpad)
        h = self.dec_norm(h)
        tok = (self.out(h) @ self.E.T.to(h.dtype)) * self.logit_scale / (self.E.shape[1] ** 0.5)
        return torch.cat([tok, self.end(h)], -1)

    def targets(self, word_ids: list[list[int]]):
        """Gold print ids per word -> (prev [W, K] allowed indices, labels [W, K+1] with END, -100 pad)."""
        dev = self.allowed.device
        K = min(self.max_tokens, max(1, max(len(w) for w in word_ids)))
        prev = torch.full((len(word_ids), K), -1, dtype=torch.long, device=dev)
        lab = torch.full((len(word_ids), K + 1), -100, dtype=torch.long, device=dev)
        endc = len(self.allowed)
        for k, w in enumerate(word_ids):
            w = w[: self.max_tokens]
            idx = self.to_idx[torch.tensor(w, dtype=torch.long, device=dev)] if w else torch.zeros(0, dtype=torch.long, device=dev)
            idx = torch.where(idx < 0, torch.full_like(idx, -100), idx)   # tokens outside the allowed set are not trained
            prev[k, : len(w)] = idx.clamp(min=-1)
            lab[k, : len(w)] = idx
            lab[k, len(w)] = endc
        return prev, lab

    def loss(self, spans: list[str], word_ids: list[list[int]], smoothing: float = 0.05):
        """Teacher-forced cross-entropy; word_ids align one to one with span_words of every span."""
        H = self.encode(spans)
        mem, mpad, words = self.word_memory(H, spans)
        assert len(words) == len(word_ids), (len(words), len(word_ids))
        prev, lab = self.targets(word_ids)
        lg = self.logits(mem, mpad, prev).float()
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), lab.reshape(-1), ignore_index=-100,
                             label_smoothing=smoothing)
        with torch.no_grad():
            ok = (lg.argmax(-1) == lab) | (lab == -100)
            word_acc = ok.all(1).float().mean()
        return ce, word_acc

    # ---- soft embeddings for the language model -------------------------------------------------
    def _mix(self, lg, hard: bool):
        p = F.softmax(lg[..., :-1].float(), -1)
        if hard:
            return self.E[p.argmax(-1)], p
        return p @ self.E, p

    def forced_embeddings(self, spans: list[str], word_ids: list[list[int]], hard: bool = False):
        """Teacher-forced soft vectors per span, [n_tokens, D] each (training through the model)."""
        H = self.encode(spans)
        mem, mpad, words = self.word_memory(H, spans)
        prev, lab = self.targets(word_ids)
        x, _ = self._mix(self.logits(mem, mpad, prev), hard)
        out = [[] for _ in spans]
        for k, (i, _, _) in enumerate(words):
            n = min(len(word_ids[k]), self.max_tokens)
            out[i].append(x[k, :n])
        return [torch.cat(v) if v else x.new_zeros((0, x.shape[-1])) for v in out]

    @torch.no_grad()
    def decode(self, spans: list[str], hard: bool = False):
        """Greedy decoding. Per span: (vectors [n, D], print ids [n], top probabilities [n], word lengths)."""
        H = self.encode(spans)
        mem, mpad, words = self.word_memory(H, spans)
        W = len(words)
        endc = len(self.allowed)
        prev = torch.full((W, 0), -1, dtype=torch.long, device=H.device)
        done = torch.zeros(W, dtype=torch.bool, device=H.device)
        vecs, ids, conf = [], [], []
        for step in range(self.max_tokens):
            lg = self.logits(mem, mpad, prev)[:, -1].float()
            if step == 0:
                lg[:, endc] = -1e9            # every word has at least one token
            nxt = lg.argmax(-1)
            stop = (nxt == endc) | done
            x, p = self._mix(lg[:, None], hard)
            tokp = p[:, 0]
            top = tokp.argmax(-1)
            vecs.append(x[:, 0])
            ids.append(torch.where(stop, torch.full_like(top, -1), top))
            conf.append(tokp.max(-1).values)
            done = stop
            if done.all():
                break
            prev = torch.cat([prev, torch.where(done, torch.zeros_like(top), top)[:, None]], 1)
        V = torch.stack(vecs, 1)
        I = torch.stack(ids, 1)
        C = torch.stack(conf, 1)
        res = [([], [], [], []) for _ in spans]
        for k, (i, _, _) in enumerate(words):
            n = int((I[k] >= 0).sum())
            res[i][0].append(V[k, :n])
            res[i][1].extend(self.allowed[I[k, :n]].tolist())
            res[i][2].extend(C[k, :n].tolist())
            res[i][3].append(n)
        out = []
        for v, i_, c, n in res:
            out.append((torch.cat(v) if v else H.new_zeros((0, self.E.shape[1])), i_, c, n))
        return out


def load_encoder(path: str, emb: torch.Tensor, device="cuda") -> VocabEncoder:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    enc = VocabEncoder(torch.tensor(ck["allowed"]), emb, **ck["config"])
    enc.load_state_dict(ck["model"], strict=False)
    enc.set_embeddings(emb)
    return enc.to(device).eval()


# ---- prompts with braille spans read by the encoder ----------------------------------------------
@dataclass
class VExample:
    parts: list = field(default_factory=list)     # ("t", ids) or ("v", span index)
    spans: list = field(default_factory=list)     # normalized braille per span
    span_ids: list = field(default_factory=list)  # gold print ids per word per span, or None
    target: list = field(default_factory=list)
    teacher: list = field(default_factory=list)
    pairs: list = field(default_factory=list)     # (student position, teacher position) per word end


def build_vocab(segments: list, base_tok, target: str = "") -> VExample:
    """Like splice.build, for the encoder. When a span's print form has as many words as the braille
    span, its gold tokens are kept (teacher forcing and word-end pairs; student and teacher then use
    the same token sequence for that span)."""
    from .splice import Braille, _token_at, word_ends

    ex = VExample()
    teacher_str, pos = "", 0
    marks = []
    for k, seg in enumerate(segments):
        if isinstance(seg, Braille):
            br = normalize(seg.braille)
            gold = None
            if seg.text and len(seg.text.split()) == len(span_words(br)):
                gold = word_targets(base_tok, seg.text)
            ex.parts.append(("v", len(ex.spans)))
            ex.spans.append(br)
            ex.span_ids.append(gold)
            if gold is not None:
                marks.append((len(teacher_str), seg.text, pos, gold))
                pos += sum(len(w) for w in gold)
            else:
                pos = None        # decoded length unknown until run time; later positions unknown
            teacher_str += seg.text
        else:
            text = seg
            if k + 1 < len(segments) and isinstance(segments[k + 1], Braille):
                text = text.rstrip(" ")
            ids = base_tok(text, add_special_tokens=False)["input_ids"] if text else []
            if ids:
                ex.parts.append(("t", ids))
                if pos is not None:
                    pos += len(ids)
            teacher_str += seg
    enc = base_tok(teacher_str, add_special_tokens=False, return_offsets_mapping=True)
    ex.teacher = enc["input_ids"]
    for start, text, s_off, gold in marks:
        if s_off is None:
            continue
        ends, c = [], 0
        for w in gold:
            c += len(w)
            ends.append(c - 1)
        for be, pe in zip(ends, word_ends(text)):
            t = _token_at(enc["offset_mapping"], start + pe - 1)
            if t is not None:
                ex.pairs.append((s_off + be, t))
    if target:
        ex.target = base_tok(target, add_special_tokens=False)["input_ids"]
    return ex
