"""Stage A: train the print-vocabulary encoder to read braille words into print tokens.

No language model in the loop; only its tokenizer and frozen input embeddings are used. Batches mix
training sentences (data/phaseB/train.jsonl, word-aligned) and single words from the word list
(data/vocab/words_g{g}.jsonl). Half of the spans get dot flips at a rate drawn from U[0, --noise]
(brltok.noise.flip_dots); the label is always the clean print.

A seeded 2% of the word list, restricted to words absent from the training sentences, is held out
as unseen dev words (saved to <out>/heldout_words.json). Checkpoints every --save-every steps; a run
resumes from <out>/ckpt.pt. Final model: <out>/encoder.pt.

    python scripts/vocab_train_a.py --grade 1 --out exp/vocab/g1_a_s0
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grade", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data/phaseB")
    ap.add_argument("--vocab", default="data/vocab")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--sent-batch", type=int, default=48)
    ap.add_argument("--word-batch", type=int, default=192)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--noise", type=float, default=0.10)
    ap.add_argument("--noise-share", type=float, default=0.5)
    ap.add_argument("--smoothing", type=float, default=0.05)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--enc-layers", type=int, default=4)
    ap.add_argument("--dec-layers", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = "cuda"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from brltok.cells import normalize
    from brltok.noise import flip_dots
    from brltok.vocab_encoder import VocabEncoder, span_words, word_targets

    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    m = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, torch_dtype=torch.bfloat16)
    emb = m.get_input_embeddings().weight.detach().float()
    del m
    allowed = torch.tensor(json.load(open(f"{a.vocab}/allowed_ids.json")))
    g = a.grade

    def sent_rows(path, limit=None):
        out = []
        for line in open(path, encoding="utf8"):
            r = json.loads(line)
            b = normalize(r[f"g{g}"])
            if len(span_words(b)) == len(r["text"].split()):
                out.append((b, r["text"]))
            if limit and len(out) >= limit:
                break
        return out

    tr = sent_rows(f"{a.data}/train.jsonl")
    dv = sent_rows(f"{a.data}/dev.jsonl", 1000)
    corpus = set()
    for _, t in tr:
        corpus.update(re.findall(r"[a-z]+(?:'[a-z]+)?", t.lower()))
    words = [json.loads(l) for l in open(f"{a.vocab}/words_g{g}.jsonl", encoding="utf8")]
    rng = random.Random(a.seed)
    held = {w["text"].lower() for w in words if w["text"].lower() not in corpus and rng.random() < 0.02}
    wtr = [(normalize(w["braille"]), w["text"]) for w in words if w["text"].lower() not in held]
    wdv = [(normalize(w["braille"]), w["text"]) for w in words if w["text"].lower() in held][:2000]
    json.dump(sorted(held), open(f"{a.out}/heldout_words.json", "w"))
    print(f"sentences {len(tr)} dev {len(dv)}; words {len(wtr)} held-out {len(wdv)}", flush=True)

    config = dict(d=a.d, enc_layers=a.enc_layers, dec_layers=a.dec_layers)
    enc = VocabEncoder(allowed, emb, **config).to(dev)
    n_params = sum(p.numel() for p in enc.parameters())
    opt = torch.optim.AdamW(enc.parameters(), lr=a.lr, weight_decay=0.01)

    def lr_at(s):
        if s < a.warmup:
            return (s + 1) / a.warmup
        return 0.5 * (1 + math.cos(math.pi * (s - a.warmup) / max(1, a.steps - a.warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    step, history = 0, []
    ck = f"{a.out}/ckpt.pt"
    if os.path.exists(ck):
        st = torch.load(ck, map_location=dev, weights_only=False)
        enc.load_state_dict(st["model"], strict=False)
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, history = st["step"], st["history"]
        print(f"resumed at step {step}", flush=True)

    def noisy(b, r):
        if r.random() < a.noise_share:
            return flip_dots(b, r.uniform(0, a.noise), r)
        return b

    def batch_loss(pairs, r=None):
        spans = [noisy(b, r) if r else b for b, _ in pairs]
        ids = [w for _, t in pairs for w in word_targets(tok, t)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return enc.loss(spans, ids, a.smoothing)

    @torch.no_grad()
    def evaluate():
        enc.eval()
        res = {}
        er = random.Random(123)
        for name, rows in (("dev_sent", dv), ("heldout_words", wdv)):
            for p in (0.0, 0.05):
                tot_l = tot_a = n = 0
                for k in range(0, len(rows), 128):
                    chunk = rows[k: k + 128]
                    spans = [flip_dots(b, p, er) if p else b for b, _ in chunk]
                    ids = [w for _, t in chunk for w in word_targets(tok, t)]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        l, acc = enc.loss(spans, ids, 0.0)
                    tot_l, tot_a, n = tot_l + l.item(), tot_a + acc.item(), n + 1
                res[f"{name}_p{int(p * 100)}"] = {"ce": round(tot_l / n, 4), "word_acc": round(100 * tot_a / n, 2)}
        enc.train()
        return res

    rng = random.Random(a.seed * 1000 + step)
    t0 = time.time()
    enc.train()
    if step == 0:
        history.append({"step": 0, **evaluate()})
    while step < a.steps:
        pairs = rng.sample(tr, a.sent_batch) + rng.sample(wtr, a.word_batch)
        loss, acc = batch_loss(pairs, rng)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0).item()
        opt.step()
        sched.step()
        step += 1
        if step % 100 == 0:
            print(f"step {step} ce {loss.item():.4f} word_acc {100 * acc.item():.1f} gn {gn:.2f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            r = evaluate()
            history.append({"step": step, "train_ce": round(loss.item(), 4), **r})
            print(f"step {step}", r, flush=True)
        if step % a.save_every == 0 or step == a.steps:
            torch.save({"model": enc.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "history": history}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
            rng = random.Random(a.seed * 1000 + step)
    torch.save({"model": enc.state_dict(), "allowed": allowed.tolist(), "config": config}, f"{a.out}/encoder.pt")
    json.dump({"args": vars(a), "config": config, "params": n_params, "history": history,
               "seconds": round(time.time() - t0, 1), "command": "python " + " ".join(sys.argv),
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "slurm_job": os.environ.get("SLURM_JOB_ID")}, open(f"{a.out}/train_log.json", "w"), indent=1)
    print("done", json.dumps(history[-1]), flush=True)


if __name__ == "__main__":
    main()
