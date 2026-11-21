import types

import torch

from brltok.composed import ComposedEmbedding, token_features
from brltok.splice import Braille, build, braille_word_end_tokens, ph, qa_segments, split_placeholders
from brltok.tokenizers import BrailleBPE, CellTokenizer, RuleUnitTokenizer

# "I had." in Grade 2
BR, PR = "⠠⠊ ⠸⠓⠲", "I had."
UNITS = [["⠠⠊", "I"], [" ", " "], ["⠸⠓", "had"], ["⠲", "."]]


class CharTok:
    """Character tokenizer with offsets, standing in for a Hugging Face fast tokenizer."""
    def __call__(self, s, add_special_tokens=False, return_offsets_mapping=False):
        out = {"input_ids": [ord(c) % 97 for c in s]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(s))]
        return out


def test_split_placeholders():
    segs = split_placeholders("Q: " + ph(0) + " C: " + ph(1) + ph(1), ["x", "y"])
    assert segs == ["Q: ", "x", " C: ", "y", "y"]


def test_word_end_tokens_t1_and_t3():
    t1 = CellTokenizer()
    assert braille_word_end_tokens(t1, BR, t1.encode(BR)) == [1, 5]
    t3 = RuleUnitTokenizer.build([UNITS])
    ids = t3.encode(BR)
    assert len(ids) == 3 and braille_word_end_tokens(t3, BR, ids) == [0, 2]
    assert braille_word_end_tokens(t1, "⣿", t1.encode("⣿")) is None     # 8-dot cell is UNK


def test_build_pairs_and_positions():
    t1 = CellTokenizer()
    ex = build(["Braille: ", Braille(BR, PR), "\nPrint:"], CharTok(), t1, target=" " + PR)
    kinds = [k for k, _ in ex.parts]
    assert kinds == ["t", "b", "t"]
    assert len(ex.parts[0][1]) == len("Braille:")           # trailing space dropped before braille
    assert ex.teacher == CharTok()("Braille: I had.\nPrint:")["input_ids"]
    # student: 8 print chars, then cells; word ends at cells 1 and 5
    # teacher: "Braille: " is 9 chars, words "I" (end 9) and "had." (end 14)
    assert ex.pairs == [(8 + 1, 9), (8 + 5, 14)]
    assert ex.n_prompt == 8 + len(BR) + len("\nPrint:")
    assert ex.target == CharTok()(" I had.")["input_ids"]


def test_build_skips_misaligned_and_eval_spans():
    t1 = CellTokenizer()
    assert build([Braille(BR, "I had it.")], CharTok(), t1).pairs == []   # 2 braille words, 3 print
    ex = build([Braille(BR)], CharTok(), t1)                             # evaluation input
    assert ex.pairs == [] and ex.teacher == []


def test_qa_segments_uses_braillebench_template():
    def make_braille_prompt(ds, rec, cfg, fmt):
        assert cfg == "G2" and fmt == "unicode"
        return f"note {ds}\nQuestion: {rec['braille_question']}\nChoices: {' '.join(rec['braille_choices'])}"
    bb = types.SimpleNamespace(make_braille_prompt=make_braille_prompt, CONFIGS={"G2-EN": "G2"})
    q, a, b = Braille("⠁"), Braille("⠃"), Braille("⠉")
    segs = qa_segments(bb, "commonsenseqa", 2, q, [a, b], chat=lambda s: "<u>" + s + "</u>")
    assert segs == ["<u>note commonsenseqa\nQuestion: ", q, "\nChoices: ", a, " ", b, "</u>"]


def test_composed_embedding_shapes_and_fit():
    torch.manual_seed(0)
    t2 = BrailleBPE.train(["⠠⠊ ⠸⠓⠲", "⠸⠓ ⠊⠞⠲"] * 20, vocab_size=80)
    cells, flags = token_features(t2)
    assert cells[t2.pad_id] == "" and cells[t2.unk_id] == ""
    emb = ComposedEmbedding(cells, flags, dim=16, hidden=32)
    assert emb(torch.tensor([[0, 5]])).shape == (1, 2, 16)
    target = torch.randn(t2.vocab_size, 16)
    rows = torch.ones(t2.vocab_size, dtype=torch.bool)
    stats = emb.fit(target, rows, steps=300, lr=3e-3)
    assert stats["mean_cos"] > 0.5
    assert torch.allclose(emb.table()[t2.pad_id], target[t2.pad_id])     # specials copy the target
    # one dot apart starts closer than many dots apart (before fitting)
    torch.manual_seed(0)
    t1 = CellTokenizer()
    c1, f1 = token_features(t1)
    e1 = ComposedEmbedding(c1, f1, dim=16, hidden=32)
    tab = e1.table().detach()
    a, b, c = (t1.encode(x)[0] for x in "⠁⠃⠿")
    assert (tab[a] - tab[b]).norm() < (tab[a] - tab[c]).norm()
