import random

import pytest

from brltok import cells, noise


def test_mask_and_dots():
    assert cells.mask("⠁") == 1
    assert cells.dots("⠓") == [1, 2, 5]
    assert cells.from_dots([1, 2, 5]) == "⠓"
    assert cells.cell(0) == cells.BLANK
    assert cells.mask(" ") == 0


def test_ascii_round_trip():
    s = "⠦⠠⠺⠑ ⠩ ⠝⠐⠑ ⠁⠛⠗⠑⠑ ⠁⠃ ⠓⠍⠂⠴"
    a = cells.to_ascii(s)
    assert a == '8,we % n"e agree ab hm10'
    assert cells.from_ascii(a) == s


def test_ascii_rejects_eight_dot():
    with pytest.raises(ValueError):
        cells.to_ascii(cells.cell(0x80))


def test_bit_matrix():
    m = cells.bit_matrix()
    assert m.shape == (64, 6)
    assert m[0b100101].tolist() == [1, 0, 1, 0, 0, 1]


def test_flip_dots_is_one_dot_away():
    rng = random.Random(0)
    s = "⠠⠮ ⠅ ⠷ ⠡⠝⠂ ⠼⠁⠃ ⠷ ⠮⠍⠲" * 20
    out = noise.flip_dots(s, 0.3, rng)
    assert len(out) == len(s)
    changed = 0
    for a, b in zip(s, out):
        if a == " ":
            assert b == " "            # word boundaries are kept
        else:
            assert b != " " and cells.hamming(a, b) <= 1
            changed += a != b
    assert 0.2 < changed / len(s.replace(" ", "")) < 0.4
    assert noise.flip_dots(s, 0.0, rng) == s
