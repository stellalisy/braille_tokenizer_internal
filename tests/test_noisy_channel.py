import math
import random
from collections import Counter

import torch

from brltok.cell_lm import CellLM
from brltok.channel import Channel, logsumexp, profile_tables
from brltok.nc_decode import Cand, Weights, Word, decode
from brltok.noise import braille_noise, flip_dots
from brltok.seq2seq import WordSeq2Seq, cell_input


def test_tables_normalize():
    for prof in ("flip", "writing", "ocr"):
        D, S, I = profile_tables(prof, 0.2)
        tot = D[1:] + S[1:].sum(1) + I[1:].sum(1)
        assert abs(tot - 1).max() < 1e-9, prof


def test_single_cell_sums_to_one():
    for prof in ("flip", "writing", "ocr", "mix"):
        ch = Channel(prof, 0.2)
        cells = [chr(0x2800 + m) for m in range(1, 64)]
        outs = [""] + cells + [a + b for a in cells for b in cells]
        p = sum(math.exp(ch.logp(o, "⠓")) for o in outs)
        assert abs(p - 1) < 1e-4, (prof, p)


def _empirical(fn, word, n, seed=0):
    rng = random.Random(seed)
    return Counter(fn(word, rng) for _ in range(n))


def test_channel_matches_generators():
    word, n = "⠓⠑⠇", 60000
    cases = [("writing", lambda w, r: braille_noise(w, 0.3, r, "writing")),
             ("ocr", lambda w, r: braille_noise(w, 0.3, r, "ocr")),
             ("flip", lambda w, r: flip_dots(w, 0.3, r))]
    for prof, fn in cases:
        ch = Channel(prof, 0.3)
        emp = _empirical(fn, word, n)
        for out, k in emp.most_common(12):
            p = math.exp(ch.logp(out, word))
            assert abs(k / n - p) < 4 * math.sqrt(p * (1 - p) / n) + 1e-3, (prof, out, k / n, p)


def test_clean_reading_is_most_likely():
    ch = Channel("mix", 0.1)
    assert ch.logp("⠓⠑⠇⠇⠕", "⠓⠑⠇⠇⠕") > ch.logp("⠓⠑⠇⠇⠕", "⠓⠑⠇⠕")
    assert ch.logp("⠓⠑⠇⠕", "⠓⠑⠇⠇⠕") > ch.logp("⠓⠑⠇⠕", "⠺⠕⠗⠇⠙")
    assert ch.logp("", "⠓") < 0 and logsumexp([0.0, -math.inf]) == 0.0


def test_seq2seq_nbest_scores_match_score():
    torch.manual_seed(0)
    m = WordSeq2Seq(66, 30, d=32, enc_layers=1, dec_layers=1, heads=2, max_out=6, cell_bits=True).eval()
    inputs = [cell_input("⠓⠑⠇"), cell_input("⠁")]
    nb = m.nbest(inputs, k=4)
    for inp, lst in zip(inputs, nb):
        assert len(lst) == 4
        lps = [lp for _, lp in lst]
        assert lps == sorted(lps, reverse=True)
        got = m.score([inp] * len(lst), [s for s, _ in lst])
        for (_, lp), g in zip(lst, got):
            assert abs(lp - g) < 1e-4
    ce, acc = m.loss(inputs, [[1, 2], [3]])
    assert torch.isfinite(ce)


def test_cell_lm_score_is_batch_independent():
    torch.manual_seed(0)
    lm = CellLM(d=32, layers=1, heads=2, max_len=64).eval()
    texts = ["⠓⠑⠇⠇⠕ ⠺⠕⠗⠇⠙", "⠁", "⠃⠗⠇"]
    together = lm.score(texts)
    alone = [lm.score([t])[0] for t in texts]
    assert all(abs(a - b) < 1e-4 for a, b in zip(together, alone))
    assert all(v < 0 for v in together)


def test_joint_decoding_fixes_what_per_word_misses():
    def item():
        return [[Word("x", [Cand("teh", -0.1), Cand("the", -1.0)]),
                 Word("y", [Cand("cot", -0.1), Cand("cat", -1.0)])]]

    def llm(texts):
        return [5.0 * (t == "the cat") + 1.0 * ("the" in t.split()) for t in texts]

    w = Weights(r=1.0, llm=1.0)
    per, _ = decode([item()], w, llm_scorer=llm, rounds=1)
    joint, r = decode([item()], w, llm_scorer=llm, rounds=4)
    assert per == [[[1, 0]]]          # "the cot"
    assert joint == [[[1, 1]]] and r <= 4   # "the cat"


def test_unrescored_words_keep_top_reading():
    items = [[[Word("x", [Cand("a", -0.01), Cand("b", -5.0)], rescore=False)]]][0]
    picks, _ = decode([items], Weights(r=0.0, llm=1.0), llm_scorer=lambda ts: [float(t == "b") for t in ts])
    assert picks == [[[0]]]
