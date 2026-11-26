"""E5 of docs/design_noisy_channel.md: real photos (data/realimg/obr_v1.jsonl, detected cells of the
401 BrailleBench images) read by the noisy-channel reader, by recognizer-quality band.

Images of this grade (or book images assigned to it, as in scripts/realimg/realimg_score.py) are read
with: the reader's top reading; per-word and joint decoding with the best "both" setting of the dev
json, using the dev-tuned mix channel; and joint decoding with the ocr channel (event mix measured on
Angelina) at eps 0.1 and 0.2. Revision 2 token reader and liblouis readings come from
<token run>/realimg_score.json. Scored with realimg_score.score (word accuracy over a difflib alignment).
Bands: cell error of the detection against liblouis forward translation of the label.

    python scripts/nc_realimg.py --root exp/nc/g1 --dev exp/nc/g1/dev/e123_Qwen3-1.7B.json \
        --token exp/vocab/v2/g1_scratch_local_nonw_mix
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/realimg")

BANDS = ((0, 10), (10, 25), (25, 50), (50, 1000))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--obr", default="data/realimg/obr_v1.jsonl")
    ap.add_argument("--llm", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--llm-revision", default="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")
    ap.add_argument("--rounds", type=int, default=4)
    a = ap.parse_args()
    from brltok.channel import Channel
    from brltok.nc_decode import Weights, decode, top_picks
    from brltok.nc_pipeline import LLMScorer, Pipeline, use_channel
    from nc_dev import likelihoods
    from realimg_score import score

    dev = json.load(open(a.dev))
    cfg = dev["best"]["both"]
    key, r, p, ch = likelihoods()[cfg.split("|")[0]]
    wt = dev["table"][cfg]["weights"]
    W = Weights(r=wt["s"] * r, p=wt["s"] * p, ch=wt["s"] * ch, br=wt["br"], llm=wt["llm"])
    eps = float(key[3:]) if key else 0.1
    pipe = Pipeline(a.root, k=dev["k"], tau=dev["tau"])
    g = pipe.grade
    llm = LLMScorer(a.llm, a.llm_revision, 0.35)
    prev = {(x["set"], x["id"]): x for x in json.load(open(f"{a.token}/realimg_score.json"))["rows"]}
    rows = [x for x in prev.values() if x["grade_used"] == g and x["obr"].strip()]
    from brltok.cells import normalize
    items = [[normalize(x["obr"])] for x in rows]
    chans = {"mix": Channel("mix", eps), "ocr0.1": Channel("ocr", 0.1), "ocr0.2": Channel("ocr", 0.2)}
    built = pipe.build(items, chans)
    reads = {}
    use_channel(built, "mix")
    reads["nc_top"] = top_picks(built)
    reads["nc_pw"], _ = decode(built, W, pipe.braille_score, llm, rounds=1)
    reads["nc_joint"], _ = decode(built, W, pipe.braille_score, llm, rounds=a.rounds)
    if ch:
        for c in ("ocr0.1", "ocr0.2"):
            use_channel(built, c)
            reads[f"nc_joint_{c}"], _ = decode(built, W, pipe.braille_score, llm, rounds=a.rounds)
    tot = {}
    out_rows = []
    for x, item, i in zip(rows, built, range(len(rows))):
        err = 100 * x["cell_edits"] / max(1, x["cells_ref"])
        band = next(f"{lo}-{hi}" for lo, hi in BANDS if lo <= err < hi)
        texts = {"liblouis": x["obr_liblouis"], "token_r2": x["obr_encoder"]}
        for name, picks in reads.items():
            texts[name] = " ".join(w.cands[j].y for w, j in zip(item[0], picks[i][0]))
        for name, t in texts.items():
            s = score(t, x["gt"])
            for k in (f"all/{name}", f"band{band}/{name}"):
                v = tot.setdefault(k, [0, 0, 0])
                v[0] += s[0]
                v[1] += s[1]
                v[2] += 1 if name == "liblouis" else 0
        out_rows.append({"set": x["set"], "id": x["id"], "gt": x["gt"], "cell_err": round(err, 1), **texts})
    summ = {k: {"word_acc": round(100 * v[0] / max(1, v[1]), 1), "label_words": v[1]} for k, v in sorted(tot.items())}
    for k, v in summ.items():
        print(k, v)
    os.makedirs(f"{a.root}/realimg", exist_ok=True)
    json.dump({"summary": summ, "images": {b: tot.get(f"band{b}/liblouis", [0, 0, 0])[2] for b in (f"{lo}-{hi}" for lo, hi in BANDS)},
               "rows": out_rows, "config": cfg, "eps": eps, "llm": a.llm, "command": "python " + " ".join(sys.argv),
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "slurm_job": os.environ.get("SLURM_JOB_ID")},
              open(f"{a.root}/realimg/e5_{a.llm.split('/')[-1]}.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
