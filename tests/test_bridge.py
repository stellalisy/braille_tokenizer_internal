from collections import Counter

import pytest
import torch

from brltok.bridge import BrailleInput, freeze, mean_init, param_hash, print_aligned_init
from brltok.embed import TableEmbedding
from brltok.expansion import corpus_expansions, sentence_expansions, token_spans
from brltok.tokenizers import CellTokenizer

# "I had." in Grade 2: capital + i, the wordsign for "had", and a period
REC = {"g2": "⠠⠊ ⠸⠓⠲", "units_g2": [["⠠⠊", "I"], [" ", " "], ["⠸⠓", "had"], ["⠲", "."]]}


def test_token_spans_cells_and_words():
    toks = list(REC["g2"])
    spans = token_spans(toks, REC["g2"])
    assert spans[0] == (0, 1, True) and spans[1] == (1, 2, False)
    assert spans[2] == (2, 3, False)      # the space cell
    assert spans[3] == (3, 4, True)


def test_token_spans_skip_spaces_and_marks():
    toks = ["▁⠠⠊</w>", "▁⠸⠓", "⠲</w>"]
    assert token_spans(toks, REC["g2"]) == [(0, 2, True), (3, 5, True), (5, 6, False)]
    with pytest.raises(ValueError):
        token_spans(["▁⠁"], REC["g2"])


def test_unit_goes_to_token_with_its_first_cell():
    exp = sentence_expansions(list(REC["g2"]), REC["g2"], REC["units_g2"])
    assert exp == [("I", True), ("", False), (" ", False), ("had", True), ("", False), (".", False)]
    exp = sentence_expansions(["▁⠠⠊</w>", "▁⠸⠓", "⠲</w>"], REC["g2"], REC["units_g2"])
    assert exp == [("I", True), ("had", True), (".", False)]


def test_corpus_expansions_counts():
    tok = CellTokenizer()
    counts, stats = corpus_expansions(tok, [REC, REC], 2)
    cap = tok.encode("⠠")[0]
    assert counts[cap] == Counter({("I", True): 2})
    assert stats == {"tokens": 12, "nonempty": 8}


class TinyTok:
    """Character tokenizer standing in for a Hugging Face tokenizer."""
    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": [ord(c) % 50 for c in s]}


def test_print_aligned_init_weights_by_count():
    base = torch.randn(50, 8)
    counts = {2: Counter({("a", False): 3, ("b", False): 1}), 3: Counter({("", False): 5})}
    table, fb = print_aligned_init(counts, TinyTok(), base, 5)
    want = (3 * base[ord("a") % 50] + base[ord("b") % 50]) / 4
    assert torch.allclose(table[2], want)
    assert fb == 4                       # ids 0, 1, 3 (empty only) and 4 fall back to E1
    sp = print_aligned_init({2: Counter({("a", True): 1})}, TinyTok(), base, 3)[0][2]
    assert torch.allclose(sp, base[[ord(" ") % 50, ord("a") % 50]].mean(0))


def test_mean_init_near_mean():
    base = torch.randn(1000, 4)
    t = mean_init(base, 3)
    assert torch.allclose(t.mean(0), base.mean(0), atol=0.05)


def test_freeze_proof_base_hash_unchanged_after_step():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    base = freeze(GPT2LMHeadModel(GPT2Config(vocab_size=50, n_embd=16, n_layer=2, n_head=2)))
    before = param_hash(base)
    bi = BrailleInput(TableEmbedding(10, 16))
    w0 = bi.embed.weight.detach().clone()
    opt = torch.optim.AdamW(bi.parameters(), lr=1e-2)
    x, s, e = bi.splice(base, TinyTok(), "ab", [3, 4, 5], "cd")
    assert x.shape == (1, 7, 16) and (s, e) == (2, 5)
    labels = torch.full((1, 7), -100)
    labels[0, 5:] = torch.tensor([ord("c") % 50, ord("d") % 50])
    loss = base(inputs_embeds=x, labels=labels).loss
    loss.backward()
    opt.step()
    assert param_hash(base) == before
    assert not torch.equal(bi.embed.weight, w0)
    assert all(p.grad is None for p in base.parameters())
