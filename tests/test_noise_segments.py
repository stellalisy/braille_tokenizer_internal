import random

from brltok.noise import noisy_segments
from brltok.splice import Braille


def test_noisy_segments_keeps_print_and_length():
    segs = ["Q: ", Braille("\u2801\u2803 \u2809", "ab c"), " end"]
    out = noisy_segments(segs, 1.0, random.Random(0))
    assert out[0] == "Q: " and out[2] == " end"
    assert out[1].text == "ab c"
    assert len(out[1].braille) == 4 and out[1].braille[2] == " "
    assert segs[1].braille == "\u2801\u2803 \u2809"   # input untouched


def test_noisy_segments_zero_is_identity():
    segs = ["x", Braille("\u2801", "a")]
    assert noisy_segments(segs, 0.0, random.Random(0)) == segs


def test_noisy_segments_deterministic():
    segs = [Braille("\u2801\u2803\u2809\u2819" * 5, "")]
    a = noisy_segments(segs, 0.5, random.Random(3))
    b = noisy_segments(segs, 0.5, random.Random(3))
    assert a == b
