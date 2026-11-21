"""Phase B stage 1: train a braille input layer into a frozen print model.

Only the braille embedding module is trained. The base model (tokenizer, embeddings, body, head)
is frozen and its parameter hash is checked at start and end. Two example types, mixed per batch:

  transcription  "Braille:" + braille sentence + "\\nPrint:"  ->  " " + print sentence
  qa             BrailleBench braille-input prompt (chat template, thinking off) with the question
                 and choices as braille spans  ->  the frozen model's own print answer

Loss = language-modeling loss on the target + lambda * distillation loss. The distillation loss is
the squared error between the student's hidden state at the last token of each braille word and
the frozen model's hidden state at the last token of the same print word (same prompt in print),
divided by the squared norm of the latter, averaged over words and over the chosen layers.

Checkpoints (trainable state, optimizer, schedule, RNG, step) are saved every --save-every steps,
and a run restarts from its last checkpoint. The final embedding table goes to <out>/table.pt.

    python scripts/phaseB_train.py --grade 1 --tok t3 --emb e2 --out exp/phaseB/train/g1_t3_e2_s0
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter

import torch
import torch.nn.functional as F

from brltok.bridge import freeze, mean_init, param_hash, print_aligned_init
from brltok.composed import ComposedEmbedding, token_features
from brltok.splice import Braille, build, qa_segments
from brltok.tokenizers import BrailleBPE, CellTokenizer, RuleUnitTokenizer

PREFIX, SUFFIX = "Braille: ", "\nPrint:"


def load_counts(path):
    out = {}
    for l in open(path, encoding="utf8"):
        d = json.loads(l)
        out[d["id"]] = Counter({(t, sw): n for t, sw, n in d["exp"]})
    return out


def load_tok(data, g, name):
    if name == "t1":
        return CellTokenizer()
    if name == "t2":
        return BrailleBPE.load(f"{data}/tok/g{g}_t2.json")
    return RuleUnitTokenizer.load(f"{data}/tok/g{g}_t3.json")


class Table(torch.nn.Module):
    def __init__(self, init):
        super().__init__()
        self.weight = torch.nn.Parameter(init.clone().float())

    def table(self):
        return self.weight

    def forward(self, ids):
        return self.weight[ids]


def make_embedding(a, btok, base_tok, base_emb, log):
    counts = load_counts(f"{a.data}/tok/g{a.grade}_{a.tok}_expansions.jsonl")
    e2, fb = print_aligned_init(counts, base_tok, base_emb, btok.vocab_size, seed=a.seed)
    log["e2_fallback"] = fb
    if a.emb == "e1":
        return Table(mean_init(base_emb, btok.vocab_size, seed=a.seed))
    if a.emb == "e2":
        return Table(e2)
    cells, flags = token_features(btok)
    m = ComposedEmbedding(cells, flags, base_emb.shape[1], hidden=a.e3_hidden)
    rows = torch.zeros(btok.vocab_size, dtype=torch.bool)
    for i, c in counts.items():
        if any(t.strip() for (t, _), n in c.items()):
            rows[i] = True
    m = m.to(a.device)
    log["e3_fit"] = m.fit(e2, rows, steps=a.e3_fit_steps, seed=a.seed)
    print("e3 fit", log["e3_fit"], flush=True)
    return m


def examples_transcription(recs, g):
    for r in recs:
        yield [PREFIX, Braille(r[f"g{g}"], r["text"]), SUFFIX], " " + r["text"]


def qa_example(bb, chat, r, g, eos):
    ch = [Braille(b, p) for b, p in zip(r.get(f"g{g}_choices", []), r.get("choices", []))]
    segs = qa_segments(bb, r["ds"], g, Braille(r[f"g{g}_question"], r["question"]), ch, chat=chat)
    return segs, r["response"] + eos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grade", type=int, required=True)
    ap.add_argument("--tok", choices=["t1", "t2", "t3"], required=True)
    ap.add_argument("--emb", choices=["e1", "e2", "e3"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="data/phaseB")
    ap.add_argument("--qa", default="data/phaseB/qa")
    ap.add_argument("--bb", default=os.path.expandvars("$BRL_SCRATCH/ext/braillebench"))
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--qa-share", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lam", type=float, default=1.0, help="distillation weight")
    ap.add_argument("--layers", default="4,8,14")
    ap.add_argument("--max-target", type=int, default=1024)
    ap.add_argument("--e3-hidden", type=int, default=512)
    ap.add_argument("--e3-fit-steps", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--n-dev", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-ckpt", action="store_true")
    a = ap.parse_args()
    a.device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = [int(x) for x in a.layers.split(",") if x]
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    sys.path.insert(0, os.path.join(a.bb, "src"))
    import evaluate_braille as bb
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, torch_dtype=torch.bfloat16).to(a.device)
    freeze(model)
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()           # checkpointing is active only in train mode; Qwen3 has no dropout
    h0 = param_hash(model)
    base = model.get_input_embeddings()
    base_emb = base.weight.detach().float().cpu()
    btok = load_tok(a.data, a.grade, a.tok)
    log = {"args": {k: v for k, v in vars(a).items()}, "base_hash": h0}
    emb = make_embedding(a, btok, tok, base_emb, log).to(a.device)
    n_params = sum(p.numel() for p in emb.parameters() if p.requires_grad)
    log["trainable_params"] = n_params

    def chat(p):
        return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)

    eos = tok.decode([tok.convert_tokens_to_ids("<|im_end|>")])
    g = a.grade
    tr = [json.loads(l) for l in open(f"{a.data}/train.jsonl", encoding="utf8")]
    tr = [{"text": r["text"], f"g{g}": r[f"g{g}"]} for r in tr]
    dev = [json.loads(l) for l in open(f"{a.data}/dev.jsonl", encoding="utf8")][: a.n_dev]
    qa_tr = [json.loads(l) for l in open(f"{a.qa}/train.jsonl", encoding="utf8")]
    qa_dev = [json.loads(l) for l in open(f"{a.qa}/dev.jsonl", encoding="utf8")]
    qa_dev = [r for r in qa_dev if r["ds"] == "gsm8k"][:100] + [r for r in qa_dev if r["ds"] == "commonsenseqa"][:100]

    def ex_tr(r):
        return build([PREFIX, Braille(r[f"g{g}"], r["text"]), SUFFIX], tok, btok, " " + r["text"])

    def ex_qa(r):
        segs, tgt = qa_example(bb, chat, r, g, eos)
        ex = build(segs, tok, btok, tgt)
        ex.target = ex.target[: a.max_target]
        return ex

    def embed(ex, table):
        xs = []
        for kind, ids in ex.parts + [("t", ex.target)]:
            if not ids:
                continue
            t = torch.tensor(ids, device=a.device)
            xs.append(base(t) if kind == "t" else table[t].to(torch.bfloat16))
        return torch.cat(xs)

    def losses(batch, train=True):
        table = emb.table()
        xs = [embed(ex, table) for ex in batch]
        T = max(x.shape[0] for x in xs)
        d = xs[0].shape[1]
        X = torch.zeros(len(xs), T, d, device=a.device, dtype=torch.bfloat16)
        M = torch.zeros(len(xs), T, device=a.device, dtype=torch.long)
        for i, x in enumerate(xs):
            X[i, : x.shape[0]], M[i, : x.shape[0]] = x, 1
        out = model(inputs_embeds=X, attention_mask=M, output_hidden_states=bool(layers and a.lam), use_cache=False)
        lm_num, lm_den = 0.0, 0
        for i, ex in enumerate(batch):
            P, n = ex.n_prompt, len(ex.target)
            logits = out.logits[i, P - 1: P - 1 + n].float()
            lm_num = lm_num + F.cross_entropy(logits, torch.tensor(ex.target, device=a.device), reduction="sum")
            lm_den += n
        lm = lm_num / lm_den
        dist = torch.zeros((), device=a.device)
        n_pairs = sum(len(ex.pairs) for ex in batch)
        if layers and a.lam and n_pairs:
            Tt = max(len(ex.teacher) for ex in batch)
            ids = torch.full((len(batch), Tt), tok.pad_token_id or 0, device=a.device)
            mt = torch.zeros_like(ids)
            for i, ex in enumerate(batch):
                ids[i, : len(ex.teacher)] = torch.tensor(ex.teacher, device=a.device)
                mt[i, : len(ex.teacher)] = 1
            with torch.no_grad():
                th = model(input_ids=ids, attention_mask=mt, output_hidden_states=True, use_cache=False).hidden_states
            bi = torch.tensor([i for i, ex in enumerate(batch) for _ in ex.pairs], device=a.device)
            sp = torch.tensor([s for ex in batch for s, _ in ex.pairs], device=a.device)
            tp = torch.tensor([t for ex in batch for _, t in ex.pairs], device=a.device)
            for L in layers:
                hs = out.hidden_states[L][bi, sp].float()
                ht = th[L][bi, tp].float()
                dist = dist + ((hs - ht).pow(2).sum(-1) / ht.pow(2).sum(-1).clamp(min=1e-6)).mean()
            dist = dist / len(layers)
        return lm, dist, n_pairs

    params = [p for p in emb.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)

    def lr_at(s):
        if s < a.warmup:
            return (s + 1) / a.warmup
        return 0.5 * (1 + math.cos(math.pi * (s - a.warmup) / max(1, a.steps - a.warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    step, history = 0, []
    ck = f"{a.out}/ckpt.pt"
    if os.path.exists(ck):
        st = torch.load(ck, map_location=a.device, weights_only=False)
        emb.load_state_dict(st["emb"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, history = st["step"], st["history"]
        random.setstate(st["py_rng"])
        torch.set_rng_state(st["torch_rng"])
        print(f"resumed at step {step}", flush=True)

    @torch.no_grad()
    def evaluate():
        res = {}
        for name, rows, mk in (("dev_transcription", dev, ex_tr), ("dev_qa", qa_dev, ex_qa)):
            lm_s, d_s, n = 0.0, 0.0, 0
            for k in range(0, len(rows), 8):
                lm, dist, _ = losses([mk(r) for r in rows[k: k + 8]], train=False)
                lm_s, d_s, n = lm_s + lm.item(), d_s + dist.item(), n + 1
            res[name] = {"lm": round(lm_s / n, 4), "dist": round(d_s / n, 4)}
        return res

    rng = random.Random(a.seed * 1000 + step)
    if step == 0:
        r0 = evaluate()
        history.append({"step": 0, **r0})
        print("step 0", r0, flush=True)
    t0, gpu_t0 = time.time(), time.time()
    n_qa = round(a.batch * a.qa_share)
    while step < a.steps:
        batch = [ex_qa(r) for r in rng.sample(qa_tr, n_qa)] + [ex_tr(r) for r in rng.sample(tr, a.batch - n_qa)]
        lm, dist, n_pairs = losses(batch)
        loss = lm + a.lam * dist
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step()
        sched.step()
        step += 1
        if step % 25 == 0:
            print(f"step {step} lm {lm.item():.4f} dist {dist.item():.4f} pairs {n_pairs} gn {gn:.3f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            r = evaluate()
            history.append({"step": step, "train_lm": round(lm.item(), 4), "train_dist": round(dist.item(), 4), **r})
            print(f"step {step}", r, flush=True)
        if step % a.save_every == 0 or step == a.steps:
            torch.save({"emb": emb.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "history": history, "py_rng": random.getstate(),
                        "torch_rng": torch.get_rng_state()}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
            rng = random.Random(a.seed * 1000 + step)   # resume draws the same batches as a run that never stopped

    with torch.no_grad():
        torch.save(emb.table().detach().float().cpu(), f"{a.out}/table.pt")
    h1 = param_hash(model)
    assert h1 == h0, "base model changed"
    log.update(history=history, base_hash_end=h1, base_unchanged=True, steps=step,
               seconds=round(time.time() - gpu_t0, 1), slurm_job=os.environ.get("SLURM_JOB_ID"),
               command="python " + " ".join(sys.argv),
               git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               tokenizer_vocab=btok.vocab_size)
    json.dump(log, open(f"{a.out}/train_log.json", "w"), indent=1)
    print("done", json.dumps(history[-1]), flush=True)


if __name__ == "__main__":
    main()
