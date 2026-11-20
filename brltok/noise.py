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
