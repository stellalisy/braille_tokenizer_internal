"""Evaluation inputs with the braille-side noise profiles (design note revision 2), for the model runs.

Same record format as scripts/phaseA_build_inputs.py, so scripts/phaseB_eval.py (encoder) and
scripts/phaseA_eval.py (print and pipeline) read them unchanged. Writes to <out>/<ds>/:
  g{1,2}_unicode_{w,o}{P}   clean BrailleBench Unicode braille corrupted with brltok.noise.braille_noise,
                            profile writing (w) or ocr (o), rate P% of non-blank cells
  pipe_g{1,2}_{w,o}{P}      liblouis back-translation of the same corrupted braille, asked as EN-EN
and links print, print_clean, g*_unicode_p{0,10} and pipe_g*_p{0,10} from the Phase A inputs, so one
--inputs directory holds every condition. Noise is seeded per (dataset, grade, profile, rate, item).

    python scripts/v2_build_inputs.py --bb $BRL_SCRATCH/ext/braillebench --out data/v2inputs
"""
import argparse
import json
import os
import random
import subprocess

from brltok import translate
from brltok.cells import BLANK, normalize
from brltok.noise import braille_noise

RATES = [5, 10]
PROFILES = {"w": "writing", "o": "ocr"}
SEED = 4343


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phaseA", default="data/phaseA")
    ap.add_argument("--datasets", default="gsm8k,commonsenseqa")
    a = ap.parse_args()
    counts = {}
    for ds in a.datasets.split(","):
        d = os.path.join(a.out, ds)
        os.makedirs(d, exist_ok=True)
        for name in ["print", "print_clean"] + [f"{k}{g}_{t}p{p}" for g in (1, 2) for p in (0, 10)
                                                for k, t in (("g", "unicode_"), ("pipe_g", ""))]:
            src = os.path.abspath(f"{a.phaseA}/{ds}/{name}.jsonl")
            dst = f"{d}/{name}.jsonl"
            assert os.path.exists(src), src
            if not os.path.lexists(dst):
                os.symlink(src, dst)
        asc = [json.loads(l) for l in open(f"{a.bb}/data/braille/{ds}/grade1/ascii.jsonl", encoding="utf8")]
        for g in (1, 2):
            uni = [json.loads(l) for l in open(f"{a.bb}/data/braille/{ds}/grade{g}/unicode.jsonl", encoding="utf8")]
            for tag, prof in PROFILES.items():
                for p in RATES:
                    ru_all, rp_all = [], []
                    for i, (u, s) in enumerate(zip(uni, asc)):
                        seed = f"{SEED}|{ds}|g{g}|{prof}|{p}|{i}"
                        rng = random.Random(seed)
                        q = braille_noise(normalize(u["braille_question"]), p / 100, rng, prof)
                        ch = [braille_noise(normalize(c), p / 100, rng, prof) for c in u.get("braille_choices") or []]
                        meta = {"noise_rate": p / 100, "noise_profile": prof, "noise_seed": seed, "grade": g}
                        ru = dict(u, braille_question=q.replace(" ", BLANK), **meta)
                        rp = dict(s, **meta)
                        rp["question"] = " ".join(translate.back(q, g).split())
                        rp["source_question"] = s.get("clean_question") or s["question"]
                        if ch:
                            ru["braille_choices"] = [c.replace(" ", BLANK) for c in ch]
                            rp["choices"] = [" ".join(translate.back(c, g).split()) for c in ch]
                        ru_all.append(ru)
                        rp_all.append(rp)
                    for name, rs in ((f"g{g}_unicode_{tag}{p}", ru_all), (f"pipe_g{g}_{tag}{p}", rp_all)):
                        with open(f"{d}/{name}.jsonl", "w", encoding="utf8") as f:
                            for r in rs:
                                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                        counts[f"{ds}/{name}"] = len(rs)
                    print(ds, g, prof, p, flush=True)
    bb = subprocess.check_output(["git", "-C", a.bb, "rev-parse", "HEAD"], text=True).strip()
    json.dump({"command": f"python scripts/v2_build_inputs.py --bb {a.bb} --out {a.out}",
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "liblouis": translate.version(), "braillebench_commit": bb, "seed": SEED, "rates": RATES,
               "profiles": PROFILES, "counts": counts}, open(f"{a.out}/manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
