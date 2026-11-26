"""Train the braille prior P(c), a causal cell language model (brltok/cell_lm.py), on the liblouis
braille of the training sentences (data/phaseB/train.jsonl, field g{g}). Reports dev cross-entropy in
bits per cell; keeps the checkpoint with the best dev loss as <out>/model.pt.

    python scripts/nc_train_cell_lm.py --grade 1 --out exp/nc/g1/cell_lm
"""
import argparse
import json
import math
import os
import random
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
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    from brltok.cell_lm import CellLM
    from brltok.cells import normalize

    g = a.grade
    tr = [normalize(json.loads(l)[f"g{g}"]) for l in open(f"{a.data}/train.jsonl", encoding="utf8")]
    dv = [normalize(json.loads(l)[f"g{g}"]) for l in open(f"{a.data}/dev.jsonl", encoding="utf8")][:1000]
    config = dict(d=a.d, layers=a.layers, max_len=a.max_len)
    lm = CellLM(**config).cuda()
    n_params = sum(p.numel() for p in lm.parameters())
    print(f"cell LM {n_params / 1e6:.1f}M params; {len(tr)} train sentences", flush=True)
    opt = torch.optim.AdamW(lm.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / a.warmup if s < a.warmup else
        0.5 * (1 + math.cos(math.pi * (s - a.warmup) / max(1, a.steps - a.warmup))))
    step, history, best = 0, [], float("inf")
    ck = f"{a.out}/ckpt.pt"
    if os.path.exists(ck):
        st = torch.load(ck, map_location="cuda", weights_only=False)
        lm.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, history, best = st["step"], st["history"], st["best"]
        print(f"resumed at step {step}", flush=True)

    @torch.no_grad()
    def evaluate():
        lm.eval()
        lp = lm.score(dv)
        n = sum(len(s) + 1 for s in dv)
        lm.train()
        return -sum(lp) / n / math.log(2)

    rng = random.Random(a.seed * 1000 + step)
    t0 = time.time()
    lm.train()
    while step < a.steps:
        batch = rng.sample(tr, a.batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = lm.loss(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 100 == 0:
            print(f"step {step} ce {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            bpc = evaluate()
            history.append({"step": step, "train_ce": round(loss.item(), 4), "dev_bits_per_cell": round(bpc, 4)})
            print(history[-1], flush=True)
            if bpc < best:
                best = bpc
                torch.save({"model": lm.state_dict(), "config": config}, f"{a.out}/model.pt")
            torch.save({"model": lm.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "step": step, "history": history, "best": best}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
            rng = random.Random(a.seed * 1000 + step)
    json.dump({"args": vars(a), "config": config, "params": n_params, "history": history, "best_dev_bits_per_cell": best,
               "command": "python " + " ".join(sys.argv),
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "slurm_job": os.environ.get("SLURM_JOB_ID")}, open(f"{a.out}/train_log.json", "w"), indent=1)
    print("done best dev bits/cell", round(best, 4), flush=True)


if __name__ == "__main__":
    main()
