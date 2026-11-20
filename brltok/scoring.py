"""Answer scoring for multiple-choice QA (CommonsenseQA).

BrailleBench scores CommonsenseQA by exact match between the extracted answer text and the gold
choice text. That gives zero to a correct answer written as a letter ("B" or "B) library"), except
for choice A, whose letter is dropped as an article. `choice_correct` instead decides which choice
the response selects and compares it with the gold choice:

1. Take the span after the last "The answer is" marker, else the last non-empty line, up to the
   end of its first sentence.
2. A choice letter, alone or as a prefix ("B", "(B)", "B)", "B. library"), selects that choice.
   A Braille ASCII echo of a choice label (",b\"> library", the Grade 1/2 form of "B) library")
   counts as the letter too. When a prefixed letter and the text after it name different choices
   ("B) race track" where race track is C), the text wins, since it is what the model wrote out.
3. Otherwise the normalized span must equal a choice text.
4. Anything else selects nothing and scores 0.

A gold answer that appears as two choices (CommonsenseQA has a few duplicate choice texts)
accepts either letter.
"""
from __future__ import annotations

import re
import string

LETTERS = "ABCDE"
_MARKER = re.compile(r"the answer is\s*:?", re.IGNORECASE)
_LETTER = re.compile(r"^\(?([A-Ea-e])\)?(?:[\).:]\s*|\s*$|\s+(?=\S))")
_BRAILLE_LABEL = re.compile(r'^,?([a-e])">\s*')
_CHOICE = re.compile(r"^\s*\(?([A-E])\)\s*(.*)$")


def normalize(s: str) -> str:
    """FlashRAG normalization: lowercase, drop punctuation and articles, squeeze spaces."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def parse_choices(choices: list[str]) -> dict[str, str]:
    """["A) bank", ...] -> {"A": "bank", ...}."""
    out = {}
    for i, c in enumerate(choices):
        m = _CHOICE.match(c)
        out[m.group(1) if m else LETTERS[i]] = (m.group(2) if m else c).strip()
    return out


def answer_span(response: str) -> str:
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    marks = list(_MARKER.finditer(response))
    if marks:
        span = response[marks[-1].end():]
    else:
        lines = [l for l in response.splitlines() if l.strip()]
        span = lines[-1] if lines else ""
    span = span.strip().splitlines()[0] if span.strip() else ""
    # Stop at the first sentence end ("the air. The other choices ..."), but keep "E. new york".
    span = re.split(r"\.\s+(?=[A-Z])", span.replace("**", ""))[0]
    return span.strip().rstrip(".").strip()


def _by_text(text: str, opts: dict[str, str]) -> list[str]:
    return [k for k, v in opts.items() if normalize(v) == normalize(text)]


def selected_choices(response: str, choices: list[str]) -> set[str]:
    """Letters of the choice the response selects (two only for duplicate choice texts)."""
    opts = parse_choices(choices)
    span = answer_span(response)
    m = _BRAILLE_LABEL.match(span)
    if m:
        letter, rest = m.group(1).upper(), span[m.end():]
        return set(_by_text(rest, opts)) or {letter}
    m = _LETTER.match(span)
    if m and m.group(1).upper() in opts:
        letter, rest = m.group(1).upper(), span[m.end():]
        if not normalize(rest):
            return {letter}
        if span[m.end() - 1:m.end()].strip() or span[m.start():m.end()].rstrip().endswith((")", ".", ":")):
            return set(_by_text(rest, opts)) or {letter}
        # "a lot of money": the word "a", not the letter A; fall through to the text match.
    return set(_by_text(span, opts))


def selected_choice(response: str, choices: list[str]) -> str | None:
    sel = selected_choices(response, choices)
    return min(sel) if sel else None


def gold_choices(gold_answers: list[str], choices: list[str]) -> set[str]:
    opts = parse_choices(choices)
    golds = {normalize(g) for g in gold_answers}
    return {k for k, v in opts.items() if normalize(v) in golds}


def gold_choice(gold_answers: list[str], choices: list[str]) -> str | None:
    g = gold_choices(gold_answers, choices)
    return min(g) if g else None


def choice_correct(response: str, choices: list[str], gold_answers: list[str]) -> float:
    sel = selected_choices(response, choices)
    return float(bool(sel) and sel <= gold_choices(gold_answers, choices))
