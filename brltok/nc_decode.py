"""Combining the noisy-channel scores and joint decoding (docs/design_noisy_channel.md).

An item is a list of spans; a span is a list of Words; each Word holds the observed braille and its
candidate readings. A candidate's score is

    w.r * reader - w.p * wordprior + w.ch * chain + w.br * B + w.llm * L

where reader, wordprior and chain depend on the word alone and B (braille prior: log probability of
the span's intended braille) and L (print prior: log probability of the item's print text) are
computed with the word replaced by the candidate and every other word at its current pick.

Per-word rescoring is one round with every other word at the reader's top reading. Joint decoding
repeats rounds, each re-picking all rescored words given the previous round's picks, until no pick
changes or `rounds` is reached. Only words marked `rescore` can change. The scorers are callables
from a list of strings to a list of floats, so any LLM and any braille LM fit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Cand:
    y: str
    reader: float
    wordprior: float = 0.0
    chain: float = -math.inf
    c: str = ""                 # best intended braille of y under the chain


@dataclass
class Word:
    b: str
    cands: list
    rescore: bool = True


@dataclass
class Weights:
    r: float = 1.0
    p: float = 0.0
    ch: float = 0.0
    br: float = 0.0
    llm: float = 0.0

    def local(self, c: Cand) -> float:
        s = 0.0
        if self.r:
            s += self.r * c.reader
        if self.p:
            s -= self.p * c.wordprior
        if self.ch:
            s += self.ch * c.chain
        return s


def print_text(item, picks, swap=None) -> str:
    spans = []
    for si, span in enumerate(item):
        ws = []
        for wi, w in enumerate(span):
            j = swap[2] if swap and swap[:2] == (si, wi) else picks[si][wi]
            ws.append(w.cands[j].y)
        spans.append(" ".join(ws))
    return "\n".join(spans)


def braille_text(span, pk, wi=None, j=None) -> str:
    return " ".join((w.cands[j].c if k == wi else w.cands[pk[k]].c) or w.b for k, w in enumerate(span))


def top_picks(items):
    return [[[0] * len(span) for span in item] for item in items]


def context_scores(items, picks, braille_scorer=None, llm_scorer=None):
    """Per item, span, word, candidate: (B, L) with the other words at `picks`. Words not marked
    rescore get None."""
    jobs, ptexts, btexts = [], [], []
    for ii, item in enumerate(items):
        for si, span in enumerate(item):
            for wi, w in enumerate(span):
                if not w.rescore or len(w.cands) < 2:
                    continue
                for j in range(len(w.cands)):
                    jobs.append((ii, si, wi, j))
                    if llm_scorer:
                        ptexts.append(print_text(item, picks[ii], (si, wi, j)))
                    if braille_scorer:
                        btexts.append(braille_text(span, picks[ii][si], wi, j))
    L = _score_unique(llm_scorer, ptexts) if llm_scorer else None
    B = _score_unique(braille_scorer, btexts) if braille_scorer else None
    out = [[[None] * len(span) for span in item] for item in items]
    for k, (ii, si, wi, j) in enumerate(jobs):
        if out[ii][si][wi] is None:
            out[ii][si][wi] = [(0.0, 0.0)] * len(items[ii][si][wi].cands)
        out[ii][si][wi][j] = (B[k] if B else 0.0, L[k] if L else 0.0)
    return out


def _score_unique(scorer, texts):
    uniq = sorted(set(texts))
    val = dict(zip(uniq, scorer(uniq)))
    return [val[t] for t in texts]


def pick(items, ctx, w: Weights, base=None):
    """New picks: argmax of the full score for rescored words; other words keep `base` (default top)."""
    base = base or top_picks(items)
    new = [[list(p) for p in item] for item in base]
    for ii, item in enumerate(items):
        for si, span in enumerate(item):
            for wi, word in enumerate(span):
                cs = ctx[ii][si][wi]
                if cs is None:
                    if word.rescore and len(word.cands) == 1:
                        new[ii][si][wi] = 0
                    continue
                new[ii][si][wi] = max(range(len(word.cands)),
                                      key=lambda j: w.local(word.cands[j]) + w.br * cs[j][0] + w.llm * cs[j][1])
    return new


def decode(items, w: Weights, braille_scorer=None, llm_scorer=None, rounds: int = 1, ctx0=None):
    """Returns (picks, rounds run). ctx0: precomputed context scores with top-reading context."""
    bs = braille_scorer if w.br else None
    ls = llm_scorer if w.llm else None
    picks = top_picks(items)
    ctx = ctx0 if ctx0 is not None else context_scores(items, picks, bs, ls)
    picks = pick(items, ctx, w)
    for r in range(1, rounds):
        if not (bs or ls):
            return picks, r
        ctx = context_scores(items, picks, bs, ls)
        new = pick(items, ctx, w, picks)
        if new == picks:
            return picks, r + 1
        picks = new
    return picks, rounds
