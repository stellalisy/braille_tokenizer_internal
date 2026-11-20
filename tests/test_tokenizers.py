import pytest

from brltok.tokenizers import MARK, BrailleBPE, CellTokenizer, RuleUnitTokenizer

SENT = "⠠⠮ ⠅ ⠷ ⠡⠝⠂ ⠼⠁⠃ ⠷ ⠮⠍⠲"   # The knowledge of children, 12 of them.

# rule units for two short sentences, as scripts/build_data.py stores them
RECORDS = [
    [["⠠⠮", "The"], [" ", " "], ["⠅", "knowledge"], [" ", " "], ["⠷", "of"], [" ", " "],
     ["⠡⠝", "children"], ["⠂", ","], [" ", " "], ["⠼⠁", "1"], ["⠃", "2"], [" ", " "],
     ["⠷", "of"], [" ", " "], ["⠮", "the"], ["⠍", "m"], ["⠲", "."]],
    [["⠅", "k"], ["⠔", "in"], ["⠙", "d"], [" ", " "], ["⠞", "t"], ["⠑", "e"], ["⠡", "ch"],
     ["⠝", "n"], ["⠊", "i"], ["⠉", "c"], ["⠊", "i"], ["⠁", "a"], ["⠝", "n"]],
]


def test_cell_tokenizer_round_trip():
    t = CellTokenizer()
    ids = t.encode(SENT)
    assert len(ids) == len(SENT)
    assert t.vocab_size == 66
    assert t.decode(ids) == SENT
    assert t.encode("x") == [t.unk_id]
    masks = t.masks()
    assert masks[:2] == [-1, -1] and masks[ids[0]] == 0b100000   # capital indicator, dot 6


def test_bpe_marks_word_boundaries():
    lines = [SENT, "⠅⠔⠙ ⠅ ⠅⠔⠙ ⠅"] * 50
    t = BrailleBPE.train(lines, vocab_size=120)
    assert t.decode(t.encode(SENT)) == SENT
    toks = t.tokens("⠅ ⠅⠔⠙")
    # a stand-alone wordsign and the same cell opening a longer word are different tokens
    assert toks[0] == MARK + "⠅" + t.EOW
    assert toks[0] != toks[1]


def test_bpe_save_load(tmp_path):
    t = BrailleBPE.train([SENT] * 20, vocab_size=100)
    p = str(tmp_path / "bpe.json")
    t.save(p)
    assert BrailleBPE.load(p).encode(SENT) == t.encode(SENT)


def test_rule_units_from_lexicon():
    t = RuleUnitTokenizer.build(RECORDS)
    toks = t.tokens(SENT)
    assert toks[:3] == [MARK + "⠠⠮" + MARK, MARK + "⠅" + MARK, MARK + "⠷" + MARK]
    assert MARK + "⠡⠝" in toks                      # shortform kept as one unit
    assert t.decode(t.encode(SENT)) == SENT
    assert t.expansions[MARK + "⠅" + MARK] == "knowledge"
    assert t.expansions[MARK + "⠅"] == "k"


def test_rule_units_longest_match_respects_position():
    t = RuleUnitTokenizer.build(RECORDS, use_lexicon=False)
    # "⠡⠝" was only seen opening a word, so inside a word it stays two units
    assert t.segment("⠞⠑⠡⠝⠊⠉⠊⠁⠝") == list("⠞⠑⠡⠝⠊⠉⠊⠁⠝")
    assert t.segment("⠡⠝⠂") == ["⠡⠝", "⠂"]


def test_rule_units_unseen_word_falls_back_to_cells():
    t = RuleUnitTokenizer.build(RECORDS)
    ids = t.encode("⠵⠵⠵")
    assert t.unk_id not in ids and len(ids) == 3
    assert t.decode(ids) == "⠵⠵⠵"


def test_rule_units_save_load(tmp_path):
    t = RuleUnitTokenizer.build(RECORDS)
    p = str(tmp_path / "rule.json")
    t.save(p)
    u = RuleUnitTokenizer.load(p)
    assert u.encode(SENT) == t.encode(SENT)
    assert u.segment("⠡⠝⠂") == t.segment("⠡⠝⠂")


def test_alignment_with_liblouis():
    pytest.importorskip("louis")
    from brltok import translate
    from brltok.align import units

    us = units("The knowledge of children, 12 of them.")
    assert [(u.cells, u.text) for u in us[:3]] == [("⠠⠮", "The"), (" ", " "), ("⠅", "knowledge")]
    assert "".join(u.cells for u in us) == SENT
    assert "".join(u.text for u in us) == "The knowledge of children, 12 of them."
    split = units("The knowledge", split_indicators=True)
    assert [u.cells for u in split] == ["⠠", "⠮", " ", "⠅"]
    assert translate.back(SENT) == "The knowledge of children, 12 of them."
