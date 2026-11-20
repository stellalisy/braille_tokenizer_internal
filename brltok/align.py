"""Rule units: braille segments and the print text each one stands for.

liblouis reports, for every braille position, the print index it was produced
from. Consecutive cells that share a print index were emitted together by one
rule (a contraction, a letter with its indicator, a punctuation sign), so a
run of such cells is one unit, and the print span up to the next unit is its
expansion.
"""
from __future__ import annotations

from dataclasses import dataclass

from .translate import Translation, forward

CAPITAL = "⠠"   # dot 6
NUMERIC = "⠼"   # dots 3456


@dataclass(frozen=True)
class Unit:
    cells: str   # one or more cells, or " " for a word break
    text: str    # print expansion


def units_from(tr: Translation, split_indicators: bool = False) -> list[Unit]:
    """Segment one translated sentence into units with their print expansions."""
    b, pos, text = tr.braille, tr.braille_to_text, tr.text
    n = len(b)
    spans = []  # (start, end, text_pos)
    i = 0
    while i < n:
        j = i + 1
        if b[i] != " ":
            while j < n and b[j] != " " and pos[j] == pos[i]:
                j += 1
        spans.append((i, j, pos[i]))
        i = j
    out = []
    for k, (s, e, p) in enumerate(spans):
        nxt = spans[k + 1][2] if k + 1 < len(spans) else len(text)
        if nxt < p:  # non-monotone map: fall back to an empty expansion
            nxt = p
        cells, exp = b[s:e], text[p:nxt]
        if cells == " ":
            out.append(Unit(" ", " "))
            continue
        if split_indicators and len(cells) > 1:
            head = ""
            if exp[:1].isupper():
                while len(cells) > 1 and cells[0] == CAPITAL:
                    head += cells[0]
                    cells = cells[1:]
            elif exp[:1].isdigit() and cells[0] == NUMERIC:
                head, cells = cells[0], cells[1:]
            for h in head:
                out.append(Unit(h, ""))
        out.append(Unit(cells, exp))
    return out


def units(text: str, grade: int = 2, split_indicators: bool = False) -> list[Unit]:
    return units_from(forward(text, grade), split_indicators)


def words(us: list[Unit]) -> list[list[Unit]]:
    """Group units into whitespace-delimited braille words."""
    out, cur = [], []
    for u in us:
        if u.cells == " ":
            if cur:
                out.append(cur)
                cur = []
        else:
            cur.append(u)
    if cur:
        out.append(cur)
    return out
