"""Braille cell utilities: Unicode cells, dot bitmasks, Braille ASCII.

A Unicode braille cell is U+2800 plus a bitmask, with dot 1 = 1, dot 2 = 2,
dot 3 = 4, dot 4 = 8, dot 5 = 16, dot 6 = 32, dot 7 = 64, dot 8 = 128.
"""
from __future__ import annotations

import numpy as np

BASE = 0x2800
BLANK = chr(BASE)

# North American Braille ASCII, indexed by the 6-dot bitmask.
_BRF = " A1B'K2L@CIF/MSP\"E3H9O6R^DJG>NTQ,*5<-U8V.%[$+X!&;:4\\0Z7(_?W]#Y)="
assert len(_BRF) == 64


def is_cell(ch: str) -> bool:
    """True for any Unicode braille pattern, including the blank cell."""
    return BASE <= ord(ch) <= BASE + 0xFF


def mask(ch: str) -> int:
    """Dot bitmask of a cell. An ASCII space counts as the blank cell."""
    if ch == " ":
        return 0
    if not is_cell(ch):
        raise ValueError(f"not a braille cell: {ch!r}")
    return ord(ch) - BASE


def cell(m: int) -> str:
    """Unicode cell for a dot bitmask."""
    if not 0 <= m <= 0xFF:
        raise ValueError(m)
    return chr(BASE + m)


def dots(ch: str) -> list[int]:
    """Raised dot numbers of a cell, for example dots('⠓') == [1, 2, 5]."""
    m = mask(ch)
    return [i + 1 for i in range(8) if m >> i & 1]


def from_dots(ds) -> str:
    """Cell with the given dot numbers raised."""
    return cell(sum(1 << (d - 1) for d in ds))


def normalize(braille: str) -> str:
    """Use an ASCII space for blank cells so that words split on whitespace."""
    return braille.replace(BLANK, " ")


def to_ascii(braille: str, lower: bool = True) -> str:
    """Unicode 6-dot braille to Braille ASCII (the BRF character set)."""
    out = []
    for ch in braille:
        if ch == " " or not is_cell(ch):
            out.append(ch)
            continue
        m = mask(ch)
        if m > 63:
            raise ValueError("Braille ASCII covers 6-dot cells only")
        a = _BRF[m]
        out.append(a.lower() if lower else a)
    return "".join(out)


def from_ascii(text: str) -> str:
    """Braille ASCII to Unicode braille. Blank cells come back as spaces."""
    out = []
    for ch in text:
        i = _BRF.find(ch.upper())
        if i < 0:
            raise ValueError(f"not a Braille ASCII character: {ch!r}")
        out.append(" " if i == 0 else cell(i))
    return "".join(out)


def bit_matrix(n_dots: int = 6) -> np.ndarray:
    """(2**n_dots, n_dots) array whose row m holds the dot bits of mask m."""
    n = 1 << n_dots
    return np.array([[m >> i & 1 for i in range(n_dots)] for m in range(n)], dtype=np.float32)


def hamming(a: str, b: str) -> int:
    """Number of dots that differ between two cells."""
    return bin(mask(a) ^ mask(b)).count("1")
