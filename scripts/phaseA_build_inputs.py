"""Build the Phase A evaluation inputs from the BrailleBench release (CPU only).

Writes one JSONL per condition in the BrailleBench record format, so its own prompt builder,
answer extraction and scoring apply unchanged:

  print                       English question (BrailleBench EN-EN)
  print_clean                 the preprocessed English text the braille was rendered from
  g{1,2}_{ascii,unicode}_p{P} braille question (and choices), P% of cells with one dot flipped
  pipe_g{1,2}_p{P}            liblouis back-translation of the same noisy braille, asked as EN-EN

Noise is drawn once per (dataset, grade, rate, item) on the Unicode cells, so the ASCII and Unicode
conditions and the pipeline see exactly the same corrupted cells.

    python scripts/phaseA_build_inputs.py --bb $BRL_SCRATCH/ext/braillebench --out exp/phaseA/inputs
"""
import argparse
import json
import os
import random
import subprocess

from brltok import noise, translate
from brltok.cells import BLANK, normalize, to_ascii

RATES = [0.0, 0.02, 0.05, 0.10]
SEED = 4242


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf8")]


def rng_for(ds, grade, p, i):
    # str seeds are hashed deterministically (sha512) by random.Random
    return random.Random(f"{SEED}|{ds}|g{grade}|{p}|{i}")


_BRF_LOWER = str.maketrans("@[\\]^", "`{|}~")


def bb_ascii(noisy_u, orig_u, orig_a):
    """Braille ASCII for noisy Unicode, in BrailleBench's lowercase BRF convention.

    Unchanged positions keep the released ASCII character; flipped cells are converted.
    """
    assert len(noisy_u) == len(orig_u) == len(orig_a), "length mismatch"
    return "".join(a if n == o else to_ascii(n).translate(_BRF_LOWER)
                   for n, o, a in zip(noisy_u, orig_u, orig_a))


def corrupt(u, p, rng):
    """Flip dots in Unicode braille u (U+2800 blanks); returns Unicode with U+2800 blanks."""
    return noise.flip_dots(normalize(u), p, rng).replace(" ", BLANK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", default="gsm8k,commonsenseqa")
    a = ap.parse_args()
    counts = {}
    for ds in a.datasets.split(","):
        d = os.path.join(a.out, ds)
        os.makedirs(d, exist_ok=True)
        base = load(f"{a.bb}/data/braille/{ds}/grade1/ascii.jsonl")
        with open(f"{d}/print.jsonl", "w", encoding="utf8") as f:
            for r in base:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[f"{ds}/print"] = len(base)
        # BrailleBench renders braille from `clean_question`, whose preprocessing strips paired `$`
        # as LaTeX delimiters. print_clean asks the print prompt on that exact text.
        with open(f"{d}/print_clean.jsonl", "w", encoding="utf8") as f:
            for r in base:
                f.write(json.dumps(dict(r, question=r.get("clean_question") or r["question"]),
                                   ensure_ascii=False) + "\n")
        counts[f"{ds}/print_clean"] = len(base)
        for g in (1, 2):
            uni = load(f"{a.bb}/data/braille/{ds}/grade{g}/unicode.jsonl")
            asc = load(f"{a.bb}/data/braille/{ds}/grade{g}/ascii.jsonl")
            assert len(uni) == len(asc) == len(base)
            # Our cell-level conversion must agree with the released ASCII wherever both are cells.
            # (Whitespace is ignored: a few sources carry tabs or NBSPs that liblouis passes through.)
            bad = sum("".join(to_ascii(normalize(u["braille_question"])).translate(_BRF_LOWER).split())
                      != "".join(s["braille_question"].split())
                      for u, s in zip(uni, asc))
            assert bad == 0, f"{ds} g{g}: {bad} ASCII conversion mismatches"
            for p in RATES:
                tag = f"p{int(round(p * 100))}"
                rows = {"ascii": [], "unicode": [], "pipe": []}
                for i, (u, s) in enumerate(zip(uni, asc)):
                    rng = rng_for(ds, g, p, i)
                    q = corrupt(u["braille_question"], p, rng)
                    ch = [corrupt(c, p, rng) for c in u.get("braille_choices") or []]
                    meta = {"noise_rate": p, "noise_seed": f"{SEED}|{ds}|g{g}|{p}|{i}", "grade": g}
                    ru = dict(u, braille_question=q, **meta)
                    ra = dict(s, braille_question=bb_ascii(q, u["braille_question"], s["braille_question"]), **meta)
                    if ch:
                        ru["braille_choices"] = ch
                        ra["braille_choices"] = [bb_ascii(c, oc, oa) for c, oc, oa in
                                                 zip(ch, u["braille_choices"], s["braille_choices"])]
                    rp = dict(s, **meta)
                    rp["question"] = " ".join(translate.back(normalize(q), g).split())
                    if ch:
                        rp["choices"] = [" ".join(translate.back(normalize(c), g).split()) for c in ch]
                    rp["source_question"] = s.get("clean_question") or s["question"]
                    rows["unicode"].append(ru)
                    rows["ascii"].append(ra)
                    rows["pipe"].append(rp)
                if p == 0:
                    assert all(r == dict(x, **{k: r[k] for k in ("noise_rate", "noise_seed", "grade")})
                               for r, x in zip(rows["ascii"], asc)), "clean ASCII differs from release"
                    assert all(r == dict(x, **{k: r[k] for k in ("noise_rate", "noise_seed", "grade")})
                               for r, x in zip(rows["unicode"], uni)), "clean Unicode differs from release"
                for kind, rs in rows.items():
                    name = f"pipe_g{g}_{tag}" if kind == "pipe" else f"g{g}_{kind}_{tag}"
                    with open(f"{d}/{name}.jsonl", "w", encoding="utf8") as f:
                        for r in rs:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    counts[f"{ds}/{name}"] = len(rs)
                    if kind == "pipe":
                        exact = sum(" ".join(r["question"].split()) == " ".join(r["source_question"].split())
                                    for r in rs) / len(rs)
                        counts[f"{ds}/{name}/back_exact"] = round(exact, 4)
                print(ds, g, tag, "done", flush=True)
    bb = subprocess.check_output(["git", "-C", a.bb, "rev-parse", "HEAD"], text=True).strip()
    json.dump({"command": f"python scripts/phaseA_build_inputs.py --bb {a.bb} --out {a.out}",
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "liblouis": translate.version(), "braillebench_commit": bb, "seed": SEED,
               "rates": RATES, "noise": "brltok.noise.flip_dots, one dot per corrupted cell, never blanks a cell",
               "counts": counts}, open(f"{a.out}/manifest.json", "w"), indent=1)
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
