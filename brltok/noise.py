"""Dot-level noise for braille text, as a stand-in for recognition errors."""
from __future__ import annotations

import random

from .cells import cell, is_cell, mask


def flip_dots(braille: str, p: float, rng: random.Random, n_dots: int = 6,
              keep_nonblank: bool = True) -> str:
    """Flip one random dot in each non-blank cell with probability p.

    With keep_nonblank, a flip that would erase the cell is redrawn, so word
    boundaries and the number of cells stay the same. Every corrupted cell is
    at Hamming distance 1 from the original.
    """
    if p <= 0:
        return braille
    out = []
    for ch in braille:
        if ch == " " or not is_cell(ch) or mask(ch) == 0 or rng.random() >= p:
            out.append(ch)
            continue
        m = mask(ch)
        while True:
            new = m ^ (1 << rng.randrange(n_dots))
            if new != 0 or not keep_nonblank:
                break
        out.append(cell(new) if new else " ")
    return "".join(out)


def drop_cells(braille: str, p: float, rng: random.Random) -> str:
    """Delete each non-blank cell with probability p."""
    return "".join(ch for ch in braille if ch == " " or rng.random() >= p)


def insert_cells(braille: str, p: float, rng: random.Random, n_dots: int = 6) -> str:
    """Insert a random non-blank cell after each cell with probability p."""
    out = []
    for ch in braille:
        out.append(ch)
        if ch != " " and rng.random() < p:
            out.append(cell(rng.randrange(1, 1 << n_dots)))
    return "".join(out)


def noisy_segments(segments, p_max: float, rng: random.Random):
    """Copy of splice segments with dots flipped in every braille span.

    One flip rate per call, drawn uniformly from [0, p_max], is shared by all spans, so an example
    is corrupted as a whole. Print segments and the print side of each span are left unchanged.
    """
    if p_max <= 0:
        return list(segments)
    from dataclasses import replace
    p = rng.uniform(0, p_max)
    return [replace(s, braille=flip_dots(s.braille, p, rng)) if hasattr(s, "braille") else s
            for s in segments]


# ---- braille-side error model (design note revision 2) ----------------------------------------------
# Dot positions as (column, row): dot 1..3 left column top to bottom, dots 4..6 right column.
_POS = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (1, 0), 4: (1, 1), 5: (1, 2)}
_NEIGH = {d: [e for e, (c2, r2) in _POS.items() if abs(c2 - c) + abs(r2 - r) == 1]
          for d, (c, r) in _POS.items()}

# Relative weights of error events per corrupted cell. "writing" follows the writing-error studies in
# the design note (dot errors dominate; omission more common than addition; neighbouring-dot
# substitutions; mirror errors rare); its exact values are assumptions. "ocr" is measured: the
# DotNeuralNet detector against gold cells on Angelina uploaded/test (scripts/realimg/angelina_channel.py,
# exp/realimg/angelina_channel.json), per 100 cells: delete 2.7 (per-image median; a heavy tail of
# images with box-matching failures is left out), insert 1.34 (34% repeat a neighbour), omit 0.40,
# multi 0.30 (after removing the gold dots-256 / detected dot-3 pair, a labelling convention), add 0.25,
# mirror 0.24, neighbour 0.12. Omitted dots are not rarer in the right column there (right_omit 1.0).
PROFILES = {
    "writing": {"omit": 0.40, "add": 0.20, "neighbour": 0.20, "mirror": 0.05, "insert": 0.08,
                "delete": 0.07, "repeat": 0.5, "right_omit": 0.7},
    "ocr": {"omit": 0.40, "add": 0.25, "neighbour": 0.12, "mirror": 0.24, "insert": 1.34,
            "delete": 2.70, "multi": 0.30, "repeat": 0.34, "right_omit": 1.0},
}


def mirror_cell(m: int) -> int:
    """Swap the left and right columns of a 6-dot mask (dot 1 <-> 4, 2 <-> 5, 3 <-> 6)."""
    return ((m & 0b000111) << 3) | ((m & 0b111000) >> 3) | (m & ~0b111111)


def _omit(m, rng, right_omit):
    on = [d for d in range(6) if m >> d & 1]
    if len(on) <= 1:
        return m
    w = [right_omit if d >= 3 else 1.0 for d in on]
    d = rng.choices(on, w)[0]
    return m & ~(1 << d)


def _add(m, rng):
    off = [d for d in range(6) if not m >> d & 1]
    return m | (1 << rng.choice(off)) if off else m


def _neighbour(m, rng):
    moves = [(d, e) for d in range(6) if m >> d & 1 for e in _NEIGH[d] if not m >> e & 1]
    if not moves:
        return m
    d, e = rng.choice(moves)
    return (m & ~(1 << d)) | (1 << e)


def braille_noise(braille: str, rate: float, rng: random.Random, profile: str = "writing") -> str:
    """Corrupt a braille string with braille-side errors, keeping word boundaries.

    Each non-blank cell is hit with probability `rate`; the event is drawn from PROFILES[profile]:
    omit one raised dot (never the last; right-column dots weighted by right_omit), add a dot, move a
    dot to a neighbouring position, mirror the cell, insert a cell after it (a repeat of it with
    probability `repeat`, else a random cell), delete it (never a word's last cell), or replace it by a
    random cell ("multi"). Blank cells are never created or removed, so braille words stay aligned with
    their print words; space errors are left to evaluation.
    """
    if rate <= 0:
        return braille
    pr = PROFILES[profile]
    events = [k for k in ("omit", "add", "neighbour", "mirror", "insert", "delete", "multi") if pr.get(k)]
    weights = [pr[k] for k in events]
    words = braille.split(" ")
    out_words = []
    for w in words:
        cells = list(w)
        out = []
        n_left = sum(1 for ch in cells if is_cell(ch) and mask(ch))
        for ch in cells:
            if not is_cell(ch) or mask(ch) == 0 or rng.random() >= rate:
                out.append(ch)
                continue
            m = mask(ch)
            ev = rng.choices(events, weights)[0]
            if ev == "omit":
                out.append(cell(_omit(m, rng, pr.get("right_omit", 1.0))))
            elif ev == "add":
                out.append(cell(_add(m, rng)))
            elif ev == "neighbour":
                out.append(cell(_neighbour(m, rng)))
            elif ev == "mirror":
                out.append(cell(mirror_cell(m) or m))
            elif ev == "multi":
                out.append(cell(rng.randrange(1, 64)))
            elif ev == "insert":
                out.append(ch)
                out.append(ch if rng.random() < pr.get("repeat", 0.5) else cell(rng.randrange(1, 64)))
            elif ev == "delete":
                if n_left > 1:
                    n_left -= 1
                else:
                    out.append(ch)
        out_words.append("".join(out))
    return " ".join(out_words)
