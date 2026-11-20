"""Braille tokenizers.

T1  CellTokenizer       one token per cell
T2  BrailleBPE          BPE merges learned on braille, with word-start and word-end marks
T3  RuleUnitTokenizer   units taken from the translation rules (contractions, letters, signs)

All three read Unicode braille with ASCII spaces between words and share the
same small interface: tokens(), encode(), decode(), vocab_size, pad_id.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from .cells import BLANK, cell, is_cell, mask

PAD, UNK = "[PAD]", "[UNK]"
MARK = "▁"  # word-boundary mark, as in SentencePiece


class CellTokenizer:
    """T1. Ids are fixed by the dot pattern, so no training is needed."""

    name = "cell"
    pad_id, unk_id, _offset = 0, 1, 2

    def __init__(self, n_dots: int = 6):
        self.n_dots = n_dots
        self.vocab_size = self._offset + (1 << n_dots)

    def tokens(self, braille: str) -> list[str]:
        return list(braille)

    def encode(self, braille: str) -> list[int]:
        out = []
        for ch in braille:
            if ch == " " or ch == BLANK:
                out.append(self._offset)
            elif is_cell(ch) and mask(ch) < (1 << self.n_dots):
                out.append(self._offset + mask(ch))
            else:
                out.append(self.unk_id)
        return out

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            if i < self._offset:
                continue
            m = i - self._offset
            out.append(" " if m == 0 else cell(m))
        return "".join(out)

    def masks(self) -> list[int]:
        """Dot bitmask for every id, with -1 for the special tokens."""
        return [-1] * self._offset + list(range(1 << self.n_dots))


class BrailleBPE:
    """T2. BPE over cells. A word-start mark and a word-end suffix let a
    stand-alone wordsign and the same cell inside a word get different tokens."""

    name = "bpe"
    EOW = "</w>"

    def __init__(self, tok):
        self._tok = tok
        self.vocab_size = tok.get_vocab_size()
        self.pad_id = tok.token_to_id(PAD)
        self.unk_id = tok.token_to_id(UNK)

    @classmethod
    def train(cls, lines, vocab_size: int = 8000) -> "BrailleBPE":
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers

        tok = Tokenizer(models.BPE(unk_token=UNK, end_of_word_suffix=cls.EOW))
        tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement=MARK, prepend_scheme="always")
        alphabet = [cell(m) for m in range(1, 64)] + [MARK]
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, special_tokens=[PAD, UNK], initial_alphabet=alphabet,
            end_of_word_suffix=cls.EOW, show_progress=False)
        tok.train_from_iterator(lines, trainer)
        return cls(tok)

    def tokens(self, braille: str) -> list[str]:
        return self._tok.encode(braille).tokens

    def encode(self, braille: str) -> list[int]:
        return self._tok.encode(braille).ids

    def decode(self, ids) -> str:
        toks = [self._tok.id_to_token(i) for i in ids if i not in (self.pad_id, self.unk_id)]
        s = "".join(toks).replace(self.EOW, "").replace(MARK, " ")
        return s.strip()

    def save(self, path: str):
        self._tok.save(path)

    @classmethod
    def load(cls, path: str) -> "BrailleBPE":
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(path))


class RuleUnitTokenizer:
    """T3. Tokens are rule units: the cell groups that liblouis emits together.

    Building needs print text and liblouis. Tokenizing needs neither: a word is
    segmented by looking it up in a lexicon of segmentations seen in training,
    and unseen words fall back to longest match over the unit inventory, where
    a multi-cell unit only matches at a word position it was seen in.
    A token is the unit's cells with a mark in front if it starts a word and a
    mark behind if it ends one.
    """

    name = "rule"

    def __init__(self, vocab, lexicon, inventory, expansions=None, use_lexicon=True):
        self.vocab = vocab                      # token string -> id
        self.inv = {i: t for t, i in vocab.items()}
        self.lexicon = lexicon                  # braille word -> list of unit cells
        self.inventory = inventory              # unit cells -> set of position classes
        self.expansions = expansions or {}      # token string -> most common print text
        self.use_lexicon = use_lexicon
        self.max_len = max((len(u) for u in inventory), default=1)
        self.vocab_size = len(vocab)
        self.pad_id, self.unk_id = vocab[PAD], vocab[UNK]

    # position class of a unit inside its word: (starts_word, ends_word)
    @staticmethod
    def _tag(cells: str, first: bool, last: bool) -> str:
        return (MARK if first else "") + cells + (MARK if last else "")

    @classmethod
    def build(cls, records, min_count: int = 1, use_lexicon: bool = True) -> "RuleUnitTokenizer":
        """records: iterables of [cells, text] unit pairs, one list per sentence."""
        from .align import Unit, words

        seg_counts = defaultdict(Counter)
        tok_counts, exp_counts = Counter(), defaultdict(Counter)
        inventory = defaultdict(set)
        for rec in records:
            for w in words([Unit(c, t) for c, t in rec]):
                seg = tuple(u.cells for u in w)
                seg_counts["".join(seg)][seg] += 1
                for k, u in enumerate(w):
                    first, last = k == 0, k == len(w) - 1
                    tok = cls._tag(u.cells, first, last)
                    tok_counts[tok] += 1
                    exp_counts[tok][u.text.lower()] += 1
                    inventory[u.cells].add((first, last))
        vocab = {PAD: 0, UNK: 1}
        # every single cell in every position is always available as a fallback
        for m in range(1, 64):
            for first in (False, True):
                for last in (False, True):
                    vocab.setdefault(cls._tag(cell(m), first, last), len(vocab))
        for tok, c in tok_counts.most_common():
            if c >= min_count:
                vocab.setdefault(tok, len(vocab))
        lexicon = {w: list(c.most_common(1)[0][0]) for w, c in seg_counts.items()}
        expansions = {t: c.most_common(1)[0][0] for t, c in exp_counts.items()}
        return cls(vocab, lexicon, {u: p for u, p in inventory.items()}, expansions, use_lexicon)

    def segment(self, word: str) -> list[str]:
        """Split one braille word into unit cell strings."""
        if self.use_lexicon and word in self.lexicon:
            return self.lexicon[word]
        out, i, n = [], 0, len(word)
        while i < n:
            best = 1
            for L in range(min(self.max_len, n - i), 1, -1):
                u = word[i:i + L]
                if (i == 0, i + L == n) in self.inventory.get(u, ()):
                    best = L
                    break
            out.append(word[i:i + best])
            i += best
        return out

    def tokens(self, braille: str) -> list[str]:
        out = []
        for w in braille.split():
            seg = self.segment(w)
            for k, u in enumerate(seg):
                out.append(self._tag(u, k == 0, k == len(seg) - 1))
        return out

    def encode(self, braille: str) -> list[int]:
        out = []
        for t in self.tokens(braille):
            i = self.vocab.get(t)
            if i is not None:
                out.append(i)
                continue
            # unseen unit at this position: spell it out cell by cell
            first, last = t.startswith(MARK), t.endswith(MARK)
            cells = t.strip(MARK)
            for k, ch in enumerate(cells):
                tok = self._tag(ch, first and k == 0, last and k == len(cells) - 1)
                out.append(self.vocab.get(tok, self.unk_id))
        return out

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            t = self.inv.get(i, "")
            if t in (PAD, UNK):
                continue
            out.append((" " if t.startswith(MARK) else "") + t.strip(MARK))
        return "".join(out).strip()

    def save(self, path: str):
        with open(path, "w", encoding="utf8") as f:
            json.dump({
                "vocab": self.vocab, "lexicon": self.lexicon,
                "inventory": {u: sorted(map(list, p)) for u, p in self.inventory.items()},
                "expansions": self.expansions, "use_lexicon": self.use_lexicon,
            }, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "RuleUnitTokenizer":
        d = json.load(open(path, encoding="utf8"))
        inv = {u: {tuple(x) for x in p} for u, p in d["inventory"].items()}
        return cls(d["vocab"], d["lexicon"], inv, d["expansions"], d["use_lexicon"])
