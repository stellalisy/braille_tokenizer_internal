"""E1 to E3 of docs/design_noisy_channel.md on dev sentences.

The first --n-dev dev sentences whose braille and print word counts match, each corrupted at --rate
with every profile (flip, writing, ocr; seeds "nc-dev|{profile}|{i}"). Sentences [0, n/2) tune the
weights; [n/2, n) report. Weights are shared across profiles (tuned on the mean).

Score of a candidate (brltok/nc_decode.py):  s * likelihood + br * braille prior + llm * print prior
  likelihoods (E2): reader; reader - gamma * wordprior; chain (channel mix at eps 0.05, 0.1, 0.2);
                    chain + lam * reader; chain with the profile-matched channel (upper bound)
  priors (E1):      none; braille prior only; print prior (LLM) only; both
E1 and E2 use per-word rescoring (every other word at its top reading). E3 reruns the best setting of
each prior with joint decoding (up to 4 rounds).

    python scripts/nc_dev.py --root exp/nc/g1
"""
import argparse
import itertools
import json
import os
import random
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, ".")

EPS = (0.05, 0.1, 0.2)
S = (0.5, 1.0, 2.0, 4.0)
BR = (0.0, 0.25, 0.5, 1.0, 2.0)
GAMMA = (0.25, 0.5, 0.75, 1.0)
LAM = (0.25, 0.5, 1.0, 2.0)


def likelihoods():
    """name -> (channel key or None, r, p, ch) ; channel key 'matched' means the profile's own channel."""
    out = {"reader": (None, 1.0, 0.0, 0.0)}
    for gm in GAMMA:
        out[f"reader-prior{gm}"] = (None, 1.0, gm, 0.0)
    for e in EPS:
        out[f"chain-mix{e}"] = (f"mix{e}", 0.0, 0.0, 1.0)
        for lam in LAM:
            out[f"chain-mix{e}+reader{lam}"] = (f"mix{e}", lam, 0.0, 1.0)
    out["chain-matched"] = ("matched", 0.0, 0.0, 1.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n-dev", type=int, default=400)
    ap.add_argument("--rate", type=float, default=0.10)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--tau", type=float, default=0.99)
    ap.add_argument("--llm", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--llm-revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    ap.add_argument("--llm-mem", type=float, default=0.35)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--out", default=None, help="default <root>/dev")
    a = ap.parse_args()
    out_dir = a.out or f"{a.root}/dev"
    os.makedirs(out_dir, exist_ok=True)
    from brltok.cells import normalize
    from brltok.channel import Channel
    from brltok.nc_decode import Weights, context_scores, decode, top_picks
    from brltok.nc_pipeline import LLMScorer, Pipeline, use_channel
    from brltok.noise import braille_noise, flip_dots
    from brltok.vocab_encoder import span_words

    t0 = time.time()
    pipe = Pipeline(a.root, k=a.k, tau=a.tau)
    g = pipe.grade
    llm = LLMScorer(a.llm, a.llm_revision, a.llm_mem)
    rows = []
    for line in open("data/phaseB/dev.jsonl", encoding="utf8"):
        r = json.loads(line)
        b = normalize(r[f"g{g}"])
        if len(span_words(b)) == len(r["text"].split()):
            rows.append((b, r["text"].split()))
        if len(rows) >= a.n_dev:
            break
    half = len(rows) // 2
    liks = likelihoods()
    feats, data = {}, {}
    for prof in ("flip", "writing", "ocr"):
        items, golds = [], []
        for i, (b, gw) in enumerate(rows):
            rng = random.Random(f"nc-dev|{prof}|{i}")
            nb = flip_dots(b, a.rate, rng) if prof == "flip" else braille_noise(b, a.rate, rng, prof)
            items.append([nb])
            golds.append(gw)
        chans = {f"mix{e}": Channel("mix", e) for e in EPS}
        chans["matched"] = Channel(prof, a.rate)
        built = pipe.build(items, chans)
        use_channel(built, "mix0.1")      # intended braille of the context words
        ctx = context_scores(built, top_picks(built), pipe.braille_score, llm)
        data[prof] = (built, golds, ctx)
        # arrays over rescored words: [n, K] features, gold index (-1 if absent), split flag
        R, P, B, L, gi, split = [], [], [], [], [], []
        CH = {key: [] for key in chans}
        fixed = {"tune": [0, 0], "report": [0, 0]}
        top1 = {"tune": [0, 0], "report": [0, 0]}
        orac = {"tune": [0, 0], "report": [0, 0]}
        K = a.k
        for ii, (item, gw) in enumerate(zip(built, golds)):
            sp = "tune" if ii < half else "report"
            for wi, (w, gword) in enumerate(zip(item[0], gw)):
                ys = [c.y for c in w.cands]
                top1[sp][0] += ys[0] == gword
                top1[sp][1] += 1
                orac[sp][0] += gword in ys
                orac[sp][1] += 1
                cs = ctx[ii][0][wi]
                if cs is None:
                    fixed[sp][0] += ys[0] == gword
                    fixed[sp][1] += 1
                    continue
                pad = K - len(ys)
                R.append([c.reader for c in w.cands] + [-np.inf] * pad)
                P.append([c.wordprior for c in w.cands] + [0.0] * pad)
                B.append([x[0] for x in cs] + [0.0] * pad)
                L.append([x[1] for x in cs] + [0.0] * pad)
                for key in chans:
                    CH[key].append([c.chains[key][0] for c in w.cands] + [-np.inf] * pad)
                gi.append(ys.index(gword) if gword in ys else -1)
                split.append(sp == "tune")
        feats[prof] = dict(R=np.array(R), P=np.array(P), B=np.array(B), L=np.array(L),
                           CH={k_: np.array(v) for k_, v in CH.items()}, gi=np.array(gi), tune=np.array(split),
                           fixed=fixed, top1=top1, oracle=orac)
        print(f"{prof}: {len(gi)} rescored words; top1 {top1}; oracle {orac}; {time.time() - t0:.0f}s", flush=True)

    def acc(prof, lik, s, br, llm_w, part):
        f = feats[prof]
        key, r, p, ch = liks[lik]
        loc = np.zeros_like(f["R"])
        if r:
            loc = loc + r * np.where(np.isfinite(f["R"]), f["R"], -1e9)
        if p:
            loc = loc - p * f["P"]
        if ch:
            loc = loc + ch * np.where(np.isfinite(f["CH"][key]), f["CH"][key], -1e9)
        valid = np.isfinite(f["R"])
        sc = np.where(valid, s * loc + br * f["B"] + llm_w * f["L"], -np.inf)
        m = f["tune"] if part == "tune" else ~f["tune"]
        ok = (sc.argmax(1) == f["gi"])[m].sum()
        fx = f["fixed"][part]
        return 100 * (ok + fx[0]) / (m.sum() + fx[1])

    priors = {"none": [(1.0, 0.0, 0.0)],
              "braille": [(s, br, 0.0) for s in S for br in BR if br],
              "llm": [(s, 0.0, 1.0) for s in S],
              "both": [(s, br, 1.0) for s in S for br in BR if br]}
    table, best_cfg = {}, {}
    for lik in liks:
        for pr, grid in priors.items():
            scored = [(np.mean([acc(p_, lik, *w, "tune") for p_ in feats]), w) for w in grid]
            tune_acc, w = max(scored, key=lambda x: x[0])
            rep = {p_: round(acc(p_, lik, *w, "report"), 2) for p_ in feats}
            table[f"{lik}|{pr}"] = {"weights": dict(zip(("s", "br", "llm"), w)), "tune_mean": round(tune_acc, 2),
                                    "report": rep, "report_mean": round(float(np.mean(list(rep.values()))), 2)}
    for pr in priors:
        cands = {k_: v for k_, v in table.items() if k_.endswith(f"|{pr}") and "matched" not in k_}
        best_cfg[pr] = max(cands, key=lambda k_: cands[k_]["tune_mean"])
        print(pr, best_cfg[pr], table[best_cfg[pr]], flush=True)
    base = {p_: {"top1": round(100 * feats[p_]["top1"]["report"][0] / feats[p_]["top1"]["report"][1], 2),
                 f"oracle{a.k}": round(100 * feats[p_]["oracle"]["report"][0] / feats[p_]["oracle"]["report"][1], 2)}
            for p_ in feats}
    print("reader baseline (report split)", base, flush=True)

    # E3: joint decoding with the best setting of each prior
    joint = {}
    for pr, cfg in best_cfg.items():
        lik, _ = cfg.split("|")
        key, r, p, ch = liks[lik]
        wt = table[cfg]["weights"]
        W = Weights(r=wt["s"] * r, p=wt["s"] * p, ch=wt["s"] * ch, br=wt["br"], llm=wt["llm"])
        res = {}
        for prof, (built, golds, ctx) in data.items():
            use_channel(built, key or "mix0.1")
            rep_items = built[half:]
            rep_golds = golds[half:]
            rep_ctx = ctx[half:]
            out = {}
            for mode, rounds in (("per_word", 1), ("joint", a.rounds)):
                picks, nr = decode(rep_items, W, pipe.braille_score, llm, rounds=rounds,
                                   ctx0=rep_ctx if rounds == 1 else None)
                ok = n = 0
                for item, pk, gw in zip(rep_items, picks, rep_golds):
                    for w, j, gword in zip(item[0], pk[0], gw):
                        ok += w.cands[j].y == gword
                        n += 1
                out[mode] = round(100 * ok / n, 2)
                if mode == "joint":
                    out["rounds"] = nr
            res[prof] = out
        joint[pr] = {"config": cfg, **res}
        print("E3", pr, joint[pr], flush=True)

    json.dump({"table": table, "best": best_cfg, "joint": joint, "reader_baseline": base,
               "n_sentences": len(rows), "tune": [0, half], "report": [half, len(rows)], "rate": a.rate,
               "rescored_words": {p_: int(len(feats[p_]["gi"])) for p_ in feats},
               "llm": a.llm, "llm_revision": a.llm_revision, "k": a.k, "tau": a.tau,
               "command": "python " + " ".join(sys.argv),
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "slurm_job": os.environ.get("SLURM_JOB_ID"), "seconds": round(time.time() - t0)},
              open(f"{out_dir}/e123_{a.llm.split('/')[-1]}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
