import random

from brltok.cells import cell, mask
from brltok.noise import PROFILES, braille_noise, mirror_cell
from brltok.vocab_encoder import span_words


def test_mirror():
    assert mirror_cell(0b000001) == 0b001000           # dot 1 -> dot 4
    assert mirror_cell(mirror_cell(0b101101)) == 0b101101
    assert cell(mirror_cell(mask("\u2811"))) == "\u280a"  # e (dots 1,5) <-> i (dots 2,4)


def test_word_count_kept_and_rate():
    rng = random.Random(0)
    b = "\u2803\u2817\u2801\u280a\u2807\u2807\u2811 \u281e\u2811\u282d\u281e \u2801"
    for prof in PROFILES:
        for _ in range(200):
            n = braille_noise(b, 0.5, rng, prof)
            assert len(span_words(n)) == len(span_words(b))
            assert all(mask(c) for c in n if c != " ")
    assert braille_noise(b, 0.0, rng) == b
    changed = sum(braille_noise(b, 0.1, random.Random(i)) != b for i in range(300))
    assert 100 < changed < 300
