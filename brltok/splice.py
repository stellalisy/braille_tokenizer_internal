"""Prompts with braille spans, for a frozen print model with a braille input layer.

A prompt is a list of segments: print text (str) or a braille span (Braille, with its print form
when known). The student sequence embeds print text with the base model's embeddings and each
braille span with the braille input layer. The teacher sequence is the same prompt in print.

Print text right before a braille span loses its trailing spaces in the student sequence, because
a word-start braille token already stands for a print token with a leading space (E2).

Words align one to one when a span has as many braille words as print words (UEB keeps print word
spacing). The last token of each braille word is then paired with the last token of the same print
word, and the distillation loss pulls the student's hidden state toward the teacher's there.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .cells import normalize
from .expansion import id_tokens, token_spans

PH0 = 0xE000      # private-use placeholders mark braille spans inside a rendered prompt


def ph(k: int) -> str:
    return chr(PH0 + k)


@dataclass
class Braille:
    braille: str
    text: str = ""          # print form; empty for evaluation inputs


@dataclass
class Example:
    parts: list = field(default_factory=list)    # ("t", ids) print or ("b", ids) braille, in order
    target: list = field(default_factory=list)   # print ids the student is trained to continue with
    teacher: list = field(default_factory=list)  # print ids of the prompt
    pairs: list = field(default_factory=list)    # (student position, teacher position) per aligned word

    @property
    def n_prompt(self) -> int:
        return sum(len(ids) for _, ids in self.parts)


def split_placeholders(s: str, spans: list) -> list:
    """Replace placeholder ph(k) in s by spans[k], returning a list of str and span segments."""
    out, buf = [], []
    for ch in s:
        k = ord(ch) - PH0
        if 0 <= k < len(spans):
            if buf:
                out.append("".join(buf))
                buf = []
            out.append(spans[k])
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out


def qa_segments(bb, dataset: str, grade: int, question: Braille, choices=None, chat=None) -> list:
    """BrailleBench's braille-input prompt (G1-EN or G2-EN, Unicode note) with braille spans spliced in.

    bb is the imported evaluate_braille module; chat maps a prompt string to the chat-formatted
    string. Choices are joined by spaces, as BrailleBench joins braille_choices.
    """
    choices = choices or []
    rec = {"braille_question": ph(0), "question": ph(0),
           "braille_choices": [ph(i + 1) for i in range(len(choices))]}
    prompt = bb.make_braille_prompt(dataset, rec, bb.CONFIGS[f"G{grade}-EN"], "unicode")
    if chat is not None:
        prompt = chat(prompt)
    return split_placeholders(prompt, [question] + list(choices))


def word_ends(s: str) -> list[int]:
    return [m.end() for m in re.finditer(r"\S+", s)]


def braille_word_end_tokens(btok, braille: str, ids: list[int]) -> list[int] | None:
    """Index of the token holding the last cell of each braille word, or None if spans fail (UNK)."""
    try:
        spans = token_spans(id_tokens(btok, ids), braille)
    except (ValueError, IndexError):
        return None
    out, j = [], 0
    for e in word_ends(braille):
        while j < len(spans) and not (spans[j][0] <= e - 1 < spans[j][1]):
            j += 1
        if j == len(spans):
            return None
        out.append(j)
    return out


def _token_at(offsets, c: int) -> int | None:
    for i, (a, b) in enumerate(offsets):
        if a <= c < b:
            return i
    return None


def build(segments: list, base_tok, btok, target: str = "") -> Example:
    ex = Example()
    teacher_str, braille_at = "", []    # char start of each braille span's print form
    for k, seg in enumerate(segments):
        if isinstance(seg, Braille):
            br = normalize(seg.braille)
            ids = btok.encode(br)
            ends = braille_word_end_tokens(btok, br, ids) if seg.text else None
            braille_at.append((len(teacher_str), seg, ex.n_prompt, ends))
            ex.parts.append(("b", ids))
            teacher_str += seg.text
        else:
            text = seg
            if k + 1 < len(segments) and isinstance(segments[k + 1], Braille):
                text = text.rstrip(" ")
            ids = base_tok(text, add_special_tokens=False)["input_ids"] if text else []
            if ids:
                ex.parts.append(("t", ids))
            teacher_str += seg
    enc = base_tok(teacher_str, add_special_tokens=False, return_offsets_mapping=True)
    ex.teacher = enc["input_ids"]
    for start, seg, s_off, ends in braille_at:
        if ends is None:
            continue
        pends = word_ends(seg.text)
        if len(pends) != len(ends):
            continue
        for be, pe in zip(ends, pends):
            t = _token_at(enc["offset_mapping"], start + pe - 1)
            if t is not None:
                ex.pairs.append((s_off + be, t))
    if target:
        ex.target = base_tok(target, add_special_tokens=False)["input_ids"]
    return ex
