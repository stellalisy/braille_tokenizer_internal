"""Decode the QA test inputs with the noisy-channel reader and write them as print inputs for
scripts/phaseA_eval.py (docs/design_noisy_channel.md, E3 and E4).

Each braille span of an item (the question, and for CommonsenseQA each choice) is read word by word;
the item's spans are decoded jointly. Three readings per condition, written as pipe-style records
(question and choices replaced by the reading, everything else copied) to
<out>/{ds}/pipe_nc{top,pw,joint}_{cond}.jsonl:
  top    the reader's top reading (no priors)
  pw     per-word rescoring with the best "both" setting of <dev json>
  joint  joint decoding with the same setting
phaseA_eval scores these with the plain print prompt (BrailleBench EN-EN), so any LLM can answer.

    python scripts/nc_decode_inputs.py --root exp/nc/g1 --dev exp/nc/g1/dev/e123_Qwen3-1.7B.json --out exp/nc/g1/inputs_qwen
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dev", required=True, help="nc_dev.py result with the tuned weights")
    ap.add_argument("--prior", default="both", help="which tuned setting of the dev json to use")
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", default="data/v2inputs")
    ap.add_argument("--datasets", default="gsm8k,commonsenseqa")
    ap.add_argument("--conds", default="p0,p10,w10,o10")
    ap.add_argument("--llm", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--llm-revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    ap.add_argument("--llm-mem", type=float, default=0.35)
    ap.add_argument("--rounds", type=int, default=4)
    a = ap.parse_args()
    from brltok.cells import normalize
    from brltok.channel import Channel
    from brltok.nc_decode import Weights, decode, top_picks
    from brltok.nc_pipeline import LLMScorer, Pipeline, use_channel

    sys.path.insert(0, "scripts")
    from nc_dev import likelihoods

    dev = json.load(open(a.dev))
    cfg = dev["best"][a.prior]
    lik = cfg.split("|")[0]
    key, r, p, ch = likelihoods()[lik]
    assert key != "matched"
    wt = dev["table"][cfg]["weights"]
    W = Weights(r=wt["s"] * r, p=wt["s"] * p, ch=wt["s"] * ch, br=wt["br"], llm=wt["llm"])
    eps = float(key[3:]) if key else 0.1
    pipe = Pipeline(a.root, k=dev["k"], tau=dev["tau"])
    g = pipe.grade
    llm = LLMScorer(a.llm, a.llm_revision, a.llm_mem)
    t0 = time.time()
    stats = {}
    for ds in a.datasets.split(","):
        os.makedirs(f"{a.out}/{ds}", exist_ok=True)
        src = os.path.abspath(f"{a.inputs}/{ds}/print.jsonl")
        try:
            os.symlink(src, f"{a.out}/{ds}/print.jsonl")
        except FileExistsError:
            pass
        for cond in a.conds.split(","):
            recs = [json.loads(l) for l in open(f"{a.inputs}/{ds}/g{g}_unicode_{cond}.jsonl", encoding="utf8")]
            items = [[normalize(r["braille_question"])] + [normalize(c) for c in r.get("braille_choices", [])]
                     for r in recs]
            built = pipe.build(items, {"mix": Channel("mix", eps)})
            use_channel(built, "mix")
            readings = {"top": top_picks(built)}
            readings["pw"], _ = decode(built, W, pipe.braille_score, llm, rounds=1)
            readings["joint"], nr = decode(built, W, pipe.braille_score, llm, rounds=a.rounds)
            for name, picks in readings.items():
                with open(f"{a.out}/{ds}/pipe_nc{name}_g{g}_{cond}.jsonl", "w", encoding="utf8") as f:
                    for rec, item, pk in zip(recs, built, picks):
                        spans = [" ".join(w.cands[j].y for w, j in zip(sp, ps)) for sp, ps in zip(item, pk)]
                        out = dict(rec)
                        out["question"] = spans[0]
                        if rec.get("braille_choices"):
                            out["choices"] = spans[1:]
                        f.write(json.dumps(out, ensure_ascii=False) + "\n")
            nw = sum(len(sp) for it in built for sp in it)
            stats[f"{ds}/{cond}"] = {"words": nw, "rescored": sum(w.rescore for it in built for sp in it for w in sp),
                                     "changed_pw": sum(j != 0 for pk in readings["pw"] for ps in pk for j in ps),
                                     "changed_joint": sum(j != 0 for pk in readings["joint"] for ps in pk for j in ps),
                                     "joint_rounds": nr}
            print(ds, cond, stats[f"{ds}/{cond}"], f"{time.time() - t0:.0f}s", flush=True)
    json.dump({"inputs_dir": a.inputs, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()},
              open(f"{a.out}/manifest.json", "w"))
    json.dump({"config": cfg, "weights": vars(W), "eps": eps, "llm": a.llm, "llm_revision": a.llm_revision,
               "stats": stats, "command": "python " + " ".join(sys.argv),
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "slurm_job": os.environ.get("SLURM_JOB_ID"), "seconds": round(time.time() - t0)},
              open(f"{a.out}/decode_meta_{a.conds.replace(',', '_')}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
