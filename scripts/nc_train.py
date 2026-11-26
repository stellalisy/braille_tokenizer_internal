"""Train one per-word model of the noisy-channel reader (docs/design_noisy_channel.md).

  --task reader      braille word -> print characters. Same data stream as revision 2's
                     g{g}_scratch_local_nonw_mix (48 sentences + 192 words per step, 30% non-word
                     items, half the spans corrupted at U[0, 10%] with a profile drawn from flip,
                     writing, ocr), split into words: the reader is word-local, so a sentence is the
                     same as its words read one by one. Same held-out word list as --heldout-from.
  --task translator  print word -> braille cells, clean pairs from the same stream.
  --task wordprior   constant input -> print word, the print-word distribution of the same stream.

Checkpoints every --save-every steps to <out>/ckpt.pt and resumes from it; final model <out>/model.pt.

    python scripts/nc_train.py --task reader --grade 1 --out exp/nc/g1/reader
"""
import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time

import torch

sys.path.insert(0, ".")

DEFAULTS = {"reader": dict(d=512, enc_layers=4, dec_layers=2, steps=20000, lr=5e-4),
            "translator": dict(d=256, enc_layers=3, dec_layers=2, steps=8000, lr=5e-4),
            "wordprior": dict(d=256, enc_layers=1, dec_layers=3, steps=8000, lr=5e-4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(DEFAULTS), required=True)
    ap.add_argument("--grade", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data/phaseB")
    ap.add_argument("--vocab", default="data/vocab")
    ap.add_argument("--heldout-from", default=None, help="run whose heldout_words.json to reuse")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--d", type=int, default=None)
    ap.add_argument("--enc-layers", type=int, default=None)
    ap.add_argument("--dec-layers", type=int, default=None)
    ap.add_argument("--sent-batch", type=int, default=48)
    ap.add_argument("--word-batch", type=int, default=192)
    ap.add_argument("--nonword-share", type=float, default=0.3)
    ap.add_argument("--noise", type=float, default=0.10)
    ap.add_argument("--noise-share", type=float, default=0.5)
    ap.add_argument("--smoothing", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for k, v in DEFAULTS[a.task].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    from brltok.cells import normalize
    from brltok.noise import braille_noise, flip_dots
    from brltok.seq2seq import N_CELL_IN, CharVocab, WordSeq2Seq, cell_input, save
    from brltok.vocab_encoder import span_words

    g = a.grade

    def rows(path, limit=None):
        out = []
        for line in open(path, encoding="utf8"):
            r = json.loads(line)
            b = normalize(r[f"g{g}"])
            if len(span_words(b)) == len(r["text"].split()):
                out.append((b, r["text"]))
            if limit and len(out) >= limit:
                break
        return out

    tr = rows(f"{a.data}/train.jsonl")
    dv = rows(f"{a.data}/dev.jsonl", 1000)
    words = [json.loads(l) for l in open(f"{a.vocab}/words_g{g}.jsonl", encoding="utf8")]
    if a.heldout_from:
        held = set(json.load(open(f"{a.heldout_from}/heldout_words.json")))
    else:
        corpus = set()
        for _, t in tr:
            corpus.update(re.findall(r"[a-z]+(?:'[a-z]+)?", t.lower()))
        rng0 = random.Random(a.seed)
        held = {w["text"].lower() for w in words if w["text"].lower() not in corpus and rng0.random() < 0.02}
    json.dump(sorted(held), open(f"{a.out}/heldout_words.json", "w"))
    wtr = [(normalize(w["braille"]), w["text"]) for w in words if w["text"].lower() not in held]
    wdv = [(normalize(w["braille"]), w["text"]) for w in words if w["text"].lower() in held][:2000]
    nw_sent, nw_word = [], []
    for line in open(f"{a.vocab}/nonwords_g{g}.jsonl", encoding="utf8"):
        r = json.loads(line)
        (nw_sent if r["kind"].startswith("sent") else nw_word).append((normalize(r["braille"]), r["text"]))

    chars = sorted({c for _, t in tr + wtr + nw_sent + nw_word for c in t if c != " "})
    cv = CharVocab(chars)
    if a.task == "reader":
        config = dict(n_in=N_CELL_IN, n_out=len(chars), cell_bits=True)
    elif a.task == "translator":
        config = dict(n_in=len(chars) + 1, n_out=64, max_in=40, max_out=64)
    else:
        config = dict(n_in=1, n_out=len(chars))
    config.update(d=a.d, enc_layers=a.enc_layers, dec_layers=a.dec_layers)
    meta = {"task": a.task, "grade": g, "chars": chars}
    model = WordSeq2Seq(**config).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{a.task}: {n_params / 1e6:.1f}M params, {len(chars)} chars; sentences {len(tr)}, words {len(wtr)}, "
          f"held-out {len(wdv)}, non-word {len(nw_sent)}+{len(nw_word)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)

    def lr_at(s):
        if s < a.warmup:
            return (s + 1) / a.warmup
        return 0.5 * (1 + math.cos(math.pi * (s - a.warmup) / max(1, a.steps - a.warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    step, history = 0, []
    ck = f"{a.out}/ckpt.pt"
    if os.path.exists(ck):
        st = torch.load(ck, map_location="cuda", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, history = st["step"], st["history"]
        print(f"resumed at step {step}", flush=True)

    def noisy(b, r):
        if a.task != "reader" or r is None or r.random() >= a.noise_share:
            return b
        prof = r.choice(["flip", "writing", "ocr"])
        rate = r.uniform(0, a.noise)
        return flip_dots(b, rate, r) if prof == "flip" else braille_noise(b, rate, r, prof)

    def pairs_to_xy(pairs, r=None):
        X, Y = [], []
        for b, t in pairs:
            nb = noisy(b, r)
            for (s, e), w in zip(span_words(nb), t.split()):
                bw = nb[s:e]
                if a.task == "reader":
                    X.append(cell_input(bw))
                    Y.append(cv.encode(w))
                elif a.task == "translator":
                    X.append(cv.encode(w))
                    Y.append(cell_input(bw))
                else:
                    X.append([0])
                    Y.append(cv.encode(w))
        return X, Y

    @torch.no_grad()
    def evaluate():
        model.eval()
        res = {}
        er = random.Random(123)
        for name, rs in (("dev_sent", dv), ("heldout_words", wdv)):
            for p in ((0.0, 0.05) if a.task == "reader" else (0.0,)):
                X, Y = pairs_to_xy([(flip_dots(b, p, er) if p else b, t) for b, t in rs])
                tl = ta = n = 0
                for k in range(0, len(X), 2048):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        l, acc = model.loss(X[k: k + 2048], Y[k: k + 2048], 0.0)
                    m = len(X[k: k + 2048])
                    tl, ta, n = tl + l.item() * m, ta + acc.item() * m, n + m
                res[f"{name}_p{int(p * 100)}"] = {"ce": round(tl / n, 4), "word_acc": round(100 * ta / n, 2)}
        model.train()
        return res

    rng = random.Random(a.seed * 1000 + step)
    t0 = time.time()
    model.train()
    if step == 0:
        history.append({"step": 0, **evaluate()})
    ns, nw = round(a.sent_batch * a.nonword_share), round(a.word_batch * a.nonword_share)
    while step < a.steps:
        pairs = (rng.sample(tr, a.sent_batch - ns) + rng.sample(nw_sent, ns)
                 + rng.sample(wtr, a.word_batch - nw) + rng.sample(nw_word, nw))
        X, Y = pairs_to_xy(pairs, rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, acc = model.loss(X, Y, a.smoothing)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        sched.step()
        step += 1
        if step % 100 == 0:
            print(f"step {step} ce {loss.item():.4f} word_acc {100 * acc.item():.1f} gn {gn:.2f} "
                  f"words {len(X)} {time.time() - t0:.0f}s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            r = evaluate()
            history.append({"step": step, "train_ce": round(loss.item(), 4), **r})
            print(f"step {step}", r, flush=True)
        if step % a.save_every == 0 or step == a.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "history": history}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
            rng = random.Random(a.seed * 1000 + step)
    save(f"{a.out}/model.pt", model, config, meta)
    json.dump({"args": vars(a), "config": config, "params": n_params, "history": history,
               "seconds": round(time.time() - t0, 1), "command": "python " + " ".join(sys.argv),
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "data_manifest": json.load(open(f"{a.vocab}/manifest.json")).get("git_commit"),
               "slurm_job": os.environ.get("SLURM_JOB_ID")}, open(f"{a.out}/train_log.json", "w"), indent=1)
    print("done", json.dumps(history[-1]), flush=True)


if __name__ == "__main__":
    main()
