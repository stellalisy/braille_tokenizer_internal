"""Print expansions of braille tokens.

Each liblouis rule unit (brltok.align) covers some cells and stands for some print text. A token
from any tokenizer (T1 cell, T2 BPE, T3 rule unit) covers a run of cells. A token's expansion in
one sentence is the print text of the units whose first cell lies inside the token, so a unit that
a token cuts through goes to the token that holds its first cell. Over a corpus each token gets a
count of expansions, and the print-aligned initialization (E2) averages over them.
"""
from __future__ import annotations

import bisect
from collections import Counter, defaultdict

from .tokenizers import MARK, BrailleBPE

_STRIP = (MARK, BrailleBPE.EOW)


def token_cells(tok: str) -> str:
    for s in _STRIP:
        tok = tok.replace(s, "")
    return tok


def token_spans(tokens: list[str], braille: str) -> list[tuple[int, int, bool]]:
    """(start, end, starts_word) in `braille` for each token.

    Tokenizers that drop spaces (T2, T3) skip them; a space token (T1) covers one space.
    """
    spans, i = [], 0
    for t in tokens:
        cells = token_cells(t)
        if cells.strip():
            while i < len(braille) and braille[i] == " ":
                i += 1
        if braille[i:i + len(cells)] != cells:
            raise ValueError(f"token {t!r} does not match braille at {i}: {braille[i:i + 10]!r}")
        spans.append((i, i + len(cells), cells.strip() != "" and (i == 0 or braille[i - 1] == " ")))
        i += len(cells)
    return spans


def sentence_expansions(tokens: list[str], braille: str, units: list[list[str]]) -> list[tuple[str, bool]]:
    """(print expansion, starts_word) per token for one sentence."""
    spans = token_spans(tokens, braille)
    starts = [s for s, _, _ in spans]
    texts = [""] * len(tokens)
    pos = 0
    for cells, text in units:
        k = bisect.bisect_right(starts, pos) - 1
        if k >= 0 and spans[k][0] <= pos < spans[k][1] and (cells != " " or token_cells(tokens[k]) == " "):
            texts[k] += text
        pos += len(cells)
    return [(texts[k], spans[k][2]) for k in range(len(tokens))]


def id_tokens(tokenizer, ids) -> list[str]:
    """Token strings for ids, consistent with encode() (T3 spells unseen units cell by cell)."""
    if hasattr(tokenizer, "inv"):
        return [tokenizer.inv[i] for i in ids]
    if hasattr(tokenizer, "_tok"):
        return [tokenizer._tok.id_to_token(i) for i in ids]
    return [tokenizer.decode([i]) or "" for i in ids]


def corpus_expansions(tokenizer, records, grade: int):
    """Expansion counts per token id over records with g{grade} braille and units_g{grade}.

    Returns (counts, stats): counts[id] is a Counter of (text, starts_word), and stats gives the
    number of token occurrences and how many of them have a non-empty expansion.
    """
    counts = defaultdict(Counter)
    n, nonempty = 0, 0
    for r in records:
        br, units = r[f"g{grade}"], r[f"units_g{grade}"]
        ids = tokenizer.encode(br)
        toks = id_tokens(tokenizer, ids)
        for i, (text, sw) in zip(ids, sentence_expansions(toks, br, units)):
            counts[i][(text, sw)] += 1
            n += 1
            nonempty += text != ""
    return counts, {"tokens": n, "nonempty": nonempty}
