import torch

from brltok.splice import Braille
from brltok.vocab_encoder import (SPACE, VocabEncoder, build_vocab, cell_ids, span_words,
                                  word_targets)


class FakeTok:
    """One token per character, id = ord(c) % 40."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        out = {"input_ids": [ord(c) % 40 for c in text]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


def tiny(seed=0):
    torch.manual_seed(seed)
    emb = torch.randn(50, 16)
    return VocabEncoder(torch.arange(40), emb, d=32, enc_layers=1, dec_layers=1, heads=2, max_len=64,
                        max_tokens=8, dropout=0.0)


def test_span_words_and_cells():
    b = "⠁⠃ ⠉"
    assert span_words(b) == [(0, 2), (3, 4)]
    assert cell_ids(b) == [1, 3, SPACE, 9]


def test_overfit_and_decode():
    tok = FakeTok()
    spans = ["⠁⠃ ⠉", "⠙⠑"]
    texts = ["ab c", "de"]
    ids = [w for t in texts for w in word_targets(tok, t)]
    enc = tiny()
    opt = torch.optim.Adam(enc.parameters(), lr=3e-3)
    for _ in range(300):
        loss, acc = enc.loss(spans, ids, smoothing=0.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert acc.item() == 1.0
    enc.eval()
    out = enc.decode(spans)
    assert out[0][1] == ids[0] + ids[1] and out[1][1] == ids[2]
    assert out[0][3] == [len(ids[0]), len(ids[1])]
    v = out[0][0]
    assert v.shape == (len(ids[0]) + len(ids[1]), 16)
    # convex mixes of embedding rows stay within the largest row norm
    assert v.norm(dim=-1).max() <= enc.E.norm(dim=-1).max() + 1e-4
    hard = enc.decode(spans, hard=True)
    assert torch.allclose(hard[1][0], enc.E[enc.to_idx[torch.tensor(ids[2])]])


def test_forced_embeddings_lengths_and_grad():
    tok = FakeTok()
    enc = tiny()
    spans = ["⠁⠃ ⠉"]
    ids = word_targets(tok, "ab c")
    x = enc.forced_embeddings(spans, ids)
    assert x[0].shape == (sum(len(w) for w in ids), 16)
    x[0].sum().backward()
    assert enc.cell.weight.grad is not None


def test_build_vocab_pairs():
    tok = FakeTok()
    segs = ["Q: ", Braille("⠁⠃ ⠉", "ab c"), " end"]
    ex = build_vocab(segs, tok, target=" x")
    # student: "Q:" (2) + span tokens " ab"(3) " c"(2) + " end"(4); teacher is the print string
    assert ex.parts[0] == ("t", tok("Q:")["input_ids"])
    assert ex.parts[1] == ("v", 0)
    assert ex.span_ids[0] == [tok(" ab")["input_ids"], tok(" c")["input_ids"]]
    assert ex.teacher == tok("Q: ab c end")["input_ids"]
    # word ends: student positions 2+2=4 and 2+4=6; teacher chars "b" at 4 and "c" at 6
    assert ex.pairs == [(4, 4), (6, 6)]
    assert ex.target == tok(" x")["input_ids"]


def test_build_vocab_without_print():
    tok = FakeTok()
    ex = build_vocab(["Q: ", Braille("⠁⠃ ⠉"), " end"], tok)
    assert ex.span_ids == [None] and ex.pairs == []


def test_local_reader_ignores_context():
    torch.manual_seed(0)
    emb = torch.randn(50, 16)
    enc = VocabEncoder(torch.arange(40), emb, d=32, enc_layers=2, dec_layers=1, heads=2, max_len=64,
                       max_tokens=8, dropout=0.0, local=True).eval()
    # the states of word "⠁⠃" are identical whatever surrounds it and wherever it sits
    a = enc.encode(["⠁⠃"])[0, :2]
    b = enc.encode(["⠉⠙⠑ ⠁⠃ ⠋"])[0, 4:6]
    assert torch.allclose(a, b, atol=1e-5)
    glob = VocabEncoder(torch.arange(40), emb, d=32, enc_layers=2, dec_layers=1, heads=2, max_len=64,
                        max_tokens=8, dropout=0.0, local=False).eval()
    assert not torch.allclose(glob.encode(["⠁⠃"])[0, :2], glob.encode(["⠉⠙⠑ ⠁⠃ ⠋"])[0, 4:6], atol=1e-3)
