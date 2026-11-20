"""Train and evaluate a toy back-translation model for one tokenizer and embedding.

    python scripts/train_backtranslate.py --tok cell --embed table --steps 3000 --out runs/cell_table

Reports character error rate and exact match on the clean test set and on
copies of it with dot noise.
"""
import argparse
import json
import os
import random
import time

import torch
from rapidfuzz.distance import Levenshtein

from brltok import noise
from brltok.ctc import BackTranslator, greedy_decode
from brltok.embed import DotEmbedding, TableEmbedding
from brltok.tokenizers import BrailleBPE, CellTokenizer, RuleUnitTokenizer


def load(path, limit=None):
    rows = [json.loads(l) for l in open(path, encoding="utf8")]
    return rows[:limit] if limit else rows


def make_tokenizer(a):
    if a.tok == "cell":
        return CellTokenizer()
    if a.tok == "bpe":
        return BrailleBPE.load(f"{a.tokdir}/bpe{a.bpe_size}.json")
    if a.tok == "rule":
        return RuleUnitTokenizer.load(f"{a.tokdir}/rule.json")
    raise ValueError(a.tok)


def make_embedding(a, tok, dim):
    if a.embed == "table":
        return TableEmbedding(tok.vocab_size, dim, tok.pad_id)
    if a.tok != "cell":
        raise ValueError("dot embeddings need the cell tokenizer")
    return DotEmbedding(tok.masks(), dim, residual=a.embed == "dots", mlp=a.embed == "dotsmlp")


def chunks(braille, text, n):
    """Split a sentence pair into aligned pieces of at most n words.

    UEB keeps print word spacing, so braille and print words pair up one to one.
    Returns None when the word counts differ.
    """
    bw, tw = braille.split(), text.split()
    if len(bw) != len(tw):
        return None
    if n <= 0:
        return [(braille, text)]
    return [(" ".join(bw[i:i + n]), " ".join(tw[i:i + n])) for i in range(0, len(bw), n)]


def batches(items, bs, shuffle, rng):
    """items: list of (ids, target_ids). Buckets by length to keep padding low."""
    order = list(range(len(items)))
    if shuffle:
        rng.shuffle(order)
        chunks = [sorted(order[i:i + bs * 50], key=lambda k: len(items[k][0]))
                  for i in range(0, len(order), bs * 50)]
        order = [k for c in chunks for k in c]
    out = [order[i:i + bs] for i in range(0, len(order), bs)]
    if shuffle:
        rng.shuffle(out)
    return out


def collate(items, idx, pad_id):
    L = max(len(items[k][0]) for k in idx)
    ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
    for r, k in enumerate(idx):
        ids[r, :len(items[k][0])] = torch.tensor(items[k][0])
    lens = torch.tensor([len(items[k][0]) for k in idx])
    tgt = torch.tensor([c for k in idx for c in items[k][1]])
    tlens = torch.tensor([len(items[k][1]) for k in idx])
    return ids, ids == pad_id, lens, tgt, tlens


@torch.no_grad()
def evaluate(model, tok, chars, braille, refs, chunk_words=0, bs=200):
    """Decode each sentence piece by piece and score the joined output."""
    model.eval()
    pieces, owner = [], []
    for k, b in enumerate(braille):
        words = b.split()
        n = chunk_words if chunk_words > 0 else max(1, len(words))
        for i in range(0, max(1, len(words)), n):
            pieces.append(" ".join(words[i:i + n]))
            owner.append(k)
    items = [(tok.encode(b) or [tok.unk_id], []) for b in pieces]
    outs = [None] * len(items)
    order = sorted(range(len(items)), key=lambda k: len(items[k][0]))
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        ids, pad, lens, _, _ = collate(items, idx, tok.pad_id)
        out = greedy_decode(model(ids, pad), lens * model.frames, chars)
        for k, h in zip(idx, out):
            outs[k] = h.strip()
    parts = [[] for _ in braille]
    for k, h in zip(owner, outs):
        parts[k].append(h)
    hyps = [" ".join(p) for p in parts]
    edits = sum(Levenshtein.distance(h, r) for h, r in zip(hyps, refs))
    cer = edits / sum(len(r) for r in refs)
    exact = sum(h == r for h, r in zip(hyps, refs)) / len(refs)
    return cer, exact, hyps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--tokdir", default="runs/tokenizers")
    ap.add_argument("--tok", choices=["cell", "bpe", "rule"], default="cell")
    ap.add_argument("--bpe-size", type=int, default=8000)
    ap.add_argument("--embed", choices=["table", "dots", "dotslinear", "dotsmlp"], default="table")
    ap.add_argument("--frames", type=int, default=0, help="output frames per token, 0 = default for the tokenizer")
    ap.add_argument("--chunk-words", type=int, default=8,
                    help="train and decode on pieces of this many words, 0 = whole sentences")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-noise", type=float, default=0.0, help="dot-flip rate applied to training input")
    ap.add_argument("--noise-copies", type=int, default=4)
    ap.add_argument("--eval-noise", default="0.01,0.02,0.05,0.1")
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)

    train = load(f"{a.data}/train.jsonl", a.train_limit or None)
    dev, test = load(f"{a.data}/dev.jsonl"), load(f"{a.data}/test.jsonl")
    tok = make_tokenizer(a)
    frames = a.frames or {"cell": 4, "rule": 5, "bpe": 12}[a.tok]

    chars = ["<blank>"] + sorted({c for r in train for c in r["text"]})
    cid = {c: i for i, c in enumerate(chars)}
    enc_t = lambda s: [cid[c] for c in s if c in cid]

    # training items; with noise, several corrupted copies of the input are tokenized up front
    copies = []
    for c in range(a.noise_copies if a.train_noise > 0 else 1):
        nrng = random.Random(1000 + c)
        items, total = [], 0
        for r in train:
            b = noise.flip_dots(r["g2"], a.train_noise, nrng) if a.train_noise > 0 else r["g2"]
            for pb, pt in chunks(b, r["text"], a.chunk_words) or []:
                total += 1
                ids, t = tok.encode(pb), enc_t(pt)
                if ids and len(ids) * frames >= len(t) + 4 and len(ids) < 500:
                    items.append((ids, t))
        copies.append(items)
    kept = len(copies[0]) / max(1, total)
    log_pieces = len(copies[0])

    model = BackTranslator(make_embedding(a, tok, a.dim), a.dim, len(chars), frames, a.layers)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    warm = 300
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.1, 1 - s / a.steps))
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    log = {"args": vars(a), "params": n_params, "frames": frames, "kept_train": kept, "train_pieces": log_pieces,
           "vocab": tok.vocab_size, "curve": []}
    print(f"params {n_params:,} vocab {tok.vocab_size} frames {frames} pieces {log_pieces} kept {kept:.4f}", flush=True)
    step, t0, done = 0, time.time(), False
    while not done:
        items = copies[rng.randrange(len(copies))]
        for idx in batches(items, a.bs, True, rng):
            model.train()
            ids, pad, lens, tgt, tlens = collate(items, idx, tok.pad_id)
            logp = model(ids, pad)
            loss = ctc(logp.transpose(0, 1), tgt, lens * frames, tlens)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 250 == 0 or step == a.steps:
                cer, ex, _ = evaluate(model, tok, chars, [r["g2"] for r in dev[:200]],
                                      [r["text"] for r in dev[:200]], a.chunk_words)
                el = time.time() - t0
                log["curve"].append({"step": step, "loss": round(loss.item(), 4),
                                     "dev_cer": round(cer, 4), "dev_exact": round(ex, 4),
                                     "seconds": round(el, 1)})
                print(f"step {step} loss {loss.item():.3f} dev CER {cer:.4f} exact {ex:.3f} {el:.0f}s", flush=True)
            if step >= a.steps:
                done = True
                break

    refs = [r["text"] for r in test]
    res = {}
    cer, ex, hyps = evaluate(model, tok, chars, [r["g2"] for r in test], refs, a.chunk_words)
    res["0"] = {"cer": round(cer, 4), "exact": round(ex, 4)}
    for p in [float(x) for x in a.eval_noise.split(",") if x]:
        nrng = random.Random(4242)
        noisy = [noise.flip_dots(r["g2"], p, nrng) for r in test]
        c, e, _ = evaluate(model, tok, chars, noisy, refs, a.chunk_words)
        res[str(p)] = {"cer": round(c, 4), "exact": round(e, 4)}
    log["test"] = res
    log["train_seconds"] = round(time.time() - t0, 1)
    log["examples"] = [{"ref": r, "hyp": h} for r, h in list(zip(refs, hyps))[:5]]
    json.dump(log, open(f"{a.out}/result.json", "w"), indent=1, ensure_ascii=False)
    torch.save(model.state_dict(), f"{a.out}/model.pt")
    print("test", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
