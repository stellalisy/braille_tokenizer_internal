"""Thin wrapper around liblouis for UEB forward and back translation.

Needs the liblouis Python bindings (`import louis`) and the UEB tables.
Braille is returned in Unicode with ASCII spaces between words.
"""
from __future__ import annotations

from dataclasses import dataclass

from .cells import BLANK, normalize

TABLES = {1: "en-ueb-g1.ctb", 2: "en-ueb-g2.ctb"}


def _louis():
    import louis  # imported lazily so the tokenizers work without liblouis

    return louis


def _tables(grade: int) -> list[str]:
    return ["unicode.dis", TABLES[grade]]


@dataclass
class Translation:
    text: str
    braille: str            # Unicode cells, ASCII spaces
    braille_to_text: list   # for each braille position, the text index it came from
    text_to_braille: list   # for each text index, the braille position it maps to


def forward(text: str, grade: int = 2) -> Translation:
    """Print to braille with position maps."""
    louis = _louis()
    out = louis.translate(_tables(grade), text)
    return Translation(text, normalize(out[0]), list(out[1]), list(out[2]))


def forward_string(text: str, grade: int = 2) -> str:
    return normalize(_louis().translateString(_tables(grade), text))


def back(braille: str, grade: int = 2) -> str:
    """Braille to print with the rule-based back-translator."""
    b = braille.replace(" ", BLANK)
    return _louis().backTranslateString(_tables(grade), b)


def version() -> str:
    return _louis().version()
