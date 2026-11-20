import pytest

torch = pytest.importorskip("torch")

from brltok.cells import cell  # noqa: E402
from brltok.ctc import BackTranslator, greedy_decode  # noqa: E402
from brltok.embed import DotEmbedding, TableEmbedding  # noqa: E402
from brltok.tokenizers import CellTokenizer  # noqa: E402


def test_dot_embedding_neighbors_start_close():
    torch.manual_seed(0)
    t = CellTokenizer()
    e = DotEmbedding(t.masks(), 32)
    a, b, c = (t.encode(cell(m))[0] for m in (0b000001, 0b000011, 0b111110))
    v = e(torch.tensor([a, b, c]))
    near, far = (v[0] - v[1]).norm(), (v[0] - v[2]).norm()
    assert near < far                      # one dot apart is closer than five dots apart
    assert e(torch.tensor([t.pad_id])).abs().sum() == 0


def test_dot_embedding_residual_makes_it_trainable_per_cell():
    t = CellTokenizer()
    e = DotEmbedding(t.masks(), 16)
    ids = torch.tensor([[5, 6]])
    e(ids).sum().backward()
    assert e.residual.weight.grad[5].abs().sum() > 0
    assert DotEmbedding(t.masks(), 16, residual=False).residual is None


def test_backtranslator_shapes_and_decode():
    t = CellTokenizer()
    m = BackTranslator(TableEmbedding(t.vocab_size, 32), 32, n_out=10, frames=4, n_layers=1)
    ids = torch.tensor([t.encode("⠠⠮ ⠅"), t.encode("⠅") + [t.pad_id] * 3])
    logp = m(ids, ids == t.pad_id)
    assert logp.shape == (2, 16, 10)
    assert torch.allclose(logp.exp().sum(-1), torch.ones(2, 16), atol=1e-4)
    chars = ["<blank>"] + list("abcdefghi")
    fake = torch.full((1, 6, 10), -9.0)
    for k, i in enumerate([1, 1, 0, 1, 2, 2]):
        fake[0, k, i] = 0.0
    assert greedy_decode(fake, torch.tensor([6]), chars) == ["aab"]
