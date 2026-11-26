"""Braille error channel P(observed cells | intended cells) for one word (docs/design_noisy_channel.md).

A pair hidden Markov model mirroring the generators in brltok.noise. Each intended cell m is, with
the probabilities of one profile at error rate eps,
  deleted                                   D[m]
  emitted as one observed cell x            S[m, x]   (x == m with probability 1 - eps, plus events
                                                       that leave the cell unchanged)
  emitted unchanged and followed by x       I[m, x]   (an inserted cell: a repeat of m, or random)
so that D[m] + sum_x S[m, x] + sum_x I[m, x] = 1. The probability of an observed word sums over all
alignments by dynamic programming (`logp`). As in braille_noise, a word never loses all its cells:
the outcome where every cell is deleted keeps the last cell instead.

Profiles: "flip" (brltok.noise.flip_dots), "writing" and "ocr" (brltok.noise.PROFILES; the ocr
event mix is measured on Angelina), and "mix", the average of the three tables, for use when the
corrupting process is not known.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .cells import is_cell, mask
from .noise import PROFILES, _NEIGH, mirror_cell

NC = 64
FLOOR = 1e-9


def _event_dists(m: int, pr: dict) -> dict:
    """Distribution over output masks of each single-cell substitution event of braille_noise."""
    out = {}
    on = [d for d in range(6) if m >> d & 1]
    off = [d for d in range(6) if not m >> d & 1]
    v = np.zeros(NC)
    if len(on) <= 1:
        v[m] = 1
    else:
        w = np.array([pr.get("right_omit", 1.0) if d >= 3 else 1.0 for d in on])
        for d, p in zip(on, w / w.sum()):
            v[m & ~(1 << d)] += p
    out["omit"] = v
    v = np.zeros(NC)
    if off:
        for d in off:
            v[m | (1 << d)] += 1 / len(off)
    else:
        v[m] = 1
    out["add"] = v
    v = np.zeros(NC)
    moves = [(d, e) for d in on for e in _NEIGH[d] if not m >> e & 1]
    if moves:
        for d, e in moves:
            v[(m & ~(1 << d)) | (1 << e)] += 1 / len(moves)
    else:
        v[m] = 1
    out["neighbour"] = v
    v = np.zeros(NC)
    v[mirror_cell(m) or m] = 1
    out["mirror"] = v
    v = np.zeros(NC)
    v[1:] = 1 / 63
    out["multi"] = v
    return out


def profile_tables(profile: str, eps: float):
    """(D [64], S [64, 64], I [64, 64]) for one profile at per-cell error rate eps."""
    D, S, I = np.zeros(NC), np.zeros((NC, NC)), np.zeros((NC, NC))
    S[0, 0] = 1.0
    for m in range(1, NC):
        if profile == "flip":
            ok = [d for d in range(6) if m ^ (1 << d)]
            S[m, m] += 1 - eps
            for d in ok:
                S[m, m ^ (1 << d)] += eps / len(ok)
            continue
        pr = PROFILES[profile]
        ev = [k for k in ("omit", "add", "neighbour", "mirror", "insert", "delete", "multi") if pr.get(k)]
        W = sum(pr[k] for k in ev)
        dists = _event_dists(m, pr)
        S[m, m] += 1 - eps
        for k in ev:
            p = eps * pr[k] / W
            if k == "delete":
                D[m] += p
            elif k == "insert":
                r = pr.get("repeat", 0.5)
                I[m, m] += p * r
                I[m, 1:] += p * (1 - r) / 63
            else:
                S[m] += p * dists[k]
    return D, S, I


class Channel:
    """Log-space tables and the alignment sum for one channel."""

    def __init__(self, profile: str = "mix", eps: float = 0.1):
        self.profile, self.eps = profile, eps
        if profile == "mix":
            tabs = [profile_tables(p, eps) for p in ("flip", "writing", "ocr")]
            D, S, I = (sum(t[i] for t in tabs) / len(tabs) for i in range(3))
        else:
            D, S, I = profile_tables(profile, eps)
        with np.errstate(divide="ignore"):
            self.lD = np.log(np.maximum(D, FLOOR))
            self.lS = np.log(np.maximum(S, FLOOR))
            self.lI = np.log(np.maximum(I, FLOOR))
        self._cache = lru_cache(maxsize=2_000_000)(self._logp)

    @staticmethod
    def ids(word: str) -> tuple:
        return tuple(mask(ch) & 63 if is_cell(ch) else 0 for ch in word)

    def logp(self, observed: str, intended: str) -> float:
        """log P(observed word | intended word), summed over alignments."""
        return self._cache(self.ids(observed), self.ids(intended))

    def _logp(self, b: tuple, c: tuple) -> float:
        b = np.asarray(b, dtype=np.int64)
        m = len(b)
        f = np.full(m + 1, -np.inf)
        f[0] = 0.0
        for ci in c:
            g = f + self.lD[ci]
            if m >= 1:
                g[1:] = np.logaddexp(g[1:], f[:-1] + self.lS[ci, b])
            if m >= 2:
                rep = np.where(b[:-1] == ci, 0.0, -np.inf)
                g[2:] = np.logaddexp(g[2:], f[:-2] + rep + self.lI[ci, b[1:]])
            f = g
        # braille_noise never deletes the last remaining cell: the all-deleted outcome keeps the last cell
        if m == 0 and len(c):
            return -math.inf
        if m == 1 and len(c) and b[0] == c[-1]:
            return float(np.logaddexp(f[1], sum(self.lD[ci] for ci in c)))
        return float(f[m])


def logsumexp(xs) -> float:
    xs = [x for x in xs if x > -math.inf]
    if not xs:
        return -math.inf
    mx = max(xs)
    return mx + math.log(sum(math.exp(x - mx) for x in xs))
